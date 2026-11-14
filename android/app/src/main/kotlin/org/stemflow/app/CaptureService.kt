package org.stemflow.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Process
import android.service.quicksettings.TileService
import android.content.ComponentName
import android.util.Log
import android.view.KeyEvent
import java.io.DataInputStream
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Live bass isolation: captured audio -> separator (Rust core, GPU) -> mix -> speaker.
 *
 * Audio comes from one of two sources:
 *  - "helper" (default): Helper.java, running as the shell uid, reroutes the source
 *    app's playback to us over a loopback socket. Works for Spotify and silences the
 *    original. The helper is started outside the app (adb for now, Shizuku later).
 *  - "projection": Android's playback capture, after the screen-share consent. Only
 *    for apps that allow capture, and the original keeps playing.
 *
 * Threads: reader (source -> hops) -> worker (separator) -> player (mix + limiter).
 * Levels are read at playback, so the controls act at once, not a latency later.
 *
 * Control over adb (same receiver the notification buttons use):
 *   adb shell am broadcast -a org.stemflow.app.CMD -p org.stemflow.app --es snap full
 *   ... --ef balance -0.5     ... --ef bass 1.5 --ef band 0
 */
class CaptureService : Service() {
    companion object {
        const val TAG = "stemflow"
        const val RATE = 44100
        const val HELPER_PORT = 28711
        const val CMD = "org.stemflow.app.CMD"
        const val HOP_S = 2.0f // the 7.8 s model takes ~1.6 s per run on a Pixel 10a
        const val LOOKAHEAD_S = 1.0f
        const val XFADE_S = 0.25f
        const val BLOCK = RATE / 50 // 20 ms, as the desktop's limiter assumes

        @Volatile var running = false
        @Volatile var status = "Off"
        // The engine holds ~1 GB and takes seconds to build: one per process, reused.
        private var engine = 0L
        private var engineModel = ""
    }

    private class Stems(val bass: FloatArray, val band: FloatArray)

    private val hops = LinkedBlockingQueue<FloatArray>()
    private val stems = LinkedBlockingQueue<Stems>()
    private var projection: MediaProjection? = null
    private var record: AudioRecord? = null
    @Volatile private var serverSocket: ServerSocket? = null
    @Volatile private var source = ""
    @Volatile private var levelDb = -120.0
    @Volatile private var computeS = 0f
    @Volatile private var late = 0
    @Volatile private var skipped = 0

    private val commands = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            i.getStringExtra("snap")?.let { Control.snap(it) }
            if (i.hasExtra("balance")) Control.setBalance(i.getFloatExtra("balance", 0f))
            if (i.hasExtra("bass") || i.hasExtra("band")) {
                Control.setLevels(i.getFloatExtra("bass", Control.bass), i.getFloatExtra("band", Control.band))
            }
            if (i.getBooleanExtra("stop", false)) stopSelf()
            Control.save(c)
            Log.i(TAG, "mix: ${Control.describe()}")
            getSystemService(NotificationManager::class.java).notify(1, notification())
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun action(label: String, key: String, value: String, id: Int): Notification.Action {
        val pi = PendingIntent.getBroadcast(
            this, id, Intent(CMD).setPackage(packageName).putExtra(key, value),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Action.Builder(null, label, pi).build()
    }

    private fun notification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, "live")
            .setContentTitle("stemflow · ${Control.describe()}")
            .setContentText(status.lineSequence().first())
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(action("Bass solo", "snap", "solo", 1))
            .addAction(action("Full mix", "snap", "full", 2))
            .addAction(action("No bass", "snap", "nobass", 3))
            .build()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running) return START_NOT_STICKY
        val code = intent?.getIntExtra("code", 0) ?: 0
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>("data")
        source = if (data != null) "projection" else "helper"

        Control.load(this)
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel("live", "Bass isolation", NotificationManager.IMPORTANCE_LOW)
        )
        status = "Starting"
        // The projection type must be in place before asking for the projection.
        startForeground(
            1, notification(),
            if (data != null) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            else ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK,
        )
        if (data != null) {
            val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val mp = mpm.getMediaProjection(code, data) ?: run { stopSelf(); return START_NOT_STICKY }
            mp.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() { stopSelf() }
            }, Handler(Looper.getMainLooper()))
            projection = mp
        }
        registerReceiver(commands, IntentFilter(CMD), Context.RECEIVER_EXPORTED)
        running = true
        tileChanged()
        thread(name = "stemflow-worker") { work() }
        return START_NOT_STICKY
    }

    private fun tileChanged() =
        TileService.requestListeningState(this, ComponentName(this, StemTile::class.java))

    private fun fail(msg: String) {
        Log.e(TAG, msg)
        status = msg
        stopSelf()
    }

    /** Worker thread: load the model, start the other threads, then separate hops. */
    private fun work() {
        // Newest .tflite in the app's files (adb: push to /data/local/tmp, then
        // `run-as org.stemflow.app cp` it into files/).
        val model = listOfNotNull(filesDir, getExternalFilesDir(null))
            .flatMap { it.listFiles()?.toList() ?: emptyList() }
            .filter { it.name.endsWith(".tflite") }
            .maxByOrNull { it.lastModified() }
            ?: return fail("No model in $filesDir")
        try {
            if (engine == 0L || engineModel != model.path) {
                status = "Loading ${model.name}…"
                Log.i(TAG, status)
                val t = System.nanoTime()
                // A changed model leaks the old engine; fine until models can be switched.
                engine = Native.engineNew(model.path, Native.GPU_FP16, HOP_S, LOOKAHEAD_S, XFADE_S)
                engineModel = model.path
                Log.i(TAG, "model ready in %.1f s".format((System.nanoTime() - t) / 1e9))
            }
        } catch (e: Throwable) {
            return fail("Model failed: ${e.message}")
        }
        if (!running) return
        val hop = Native.engineHop(engine)
        thread(name = "stemflow-player") { play() }
        thread(name = "stemflow-reader") { if (source == "helper") readHelper(hop) else readProjection(hop) }

        val hopS = hop.toFloat() / RATE
        while (running) {
            var x = hops.poll(200, TimeUnit.MILLISECONDS) ?: continue
            // Fallen behind (a slow run, a warm phone): skip ahead to the newest hop
            // rather than stay late for good. It is heard as a jump in the music.
            while (true) {
                x = hops.poll() ?: break
                skipped++
            }
            val bass = FloatArray(2 * hop)
            val band = FloatArray(2 * hop)
            if (x.all { it > -1e-4f && it < 1e-4f }) {
                // The source is paused: play silence and leave the GPU alone.
                stems.put(Stems(bass, band))
                status = "%s · paused".format(model.name.removeSuffix(".tflite"))
                continue
            }
            computeS = try {
                Native.engineProcess(engine, x, bass, band)
            } catch (e: Throwable) {
                return fail("Separator failed: ${e.message}")
            }
            if (computeS > hopS) late++
            stems.put(Stems(bass, band))
            val latency = hopS + LOOKAHEAD_S + computeS + hops.size * hopS
            status = "%s · %s\n%.2f s per %.1f s hop · ~%.1f s behind%s\ninput %.0f dBFS".format(
                model.name.removeSuffix(".tflite"), if (source == "helper") "helper" else "screen capture",
                computeS, hopS, latency, if (skipped > 0) " · skipped $skipped" else "", levelDb,
            )
            Log.i(TAG, "run %.3fs late=%d skipped=%d in=%.1fdBFS %s".format(computeS, late, skipped, levelDb, Control.describe()))
        }
    }

    /** Cuts the incoming stream into hops, and meters it. */
    private inner class Hopper(private val hop: Int) {
        private var buf = FloatArray(2 * hop)
        private var fill = 0
        private var sumSq = 0.0
        private var n = 0

        fun push(x: FloatArray, len: Int) {
            var i = 0
            while (i < len) {
                val take = minOf(len - i, buf.size - fill)
                System.arraycopy(x, i, buf, fill, take)
                fill += take
                i += take
                if (fill == buf.size) {
                    hops.put(buf)
                    buf = FloatArray(2 * hop)
                    fill = 0
                }
            }
            for (k in 0 until len) sumSq += x[k] * x[k]
            n += len
            if (n >= RATE) {
                val rms = sqrt(sumSq / n)
                levelDb = if (rms > 0) 20 * log10(rms) else -120.0
                sumSq = 0.0
                n = 0
            }
        }
    }

    private fun readHelper(hop: Int) {
        val hopper = Hopper(hop)
        try {
            ServerSocket(HELPER_PORT, 1, InetAddress.getLoopbackAddress()).use { server ->
                serverSocket = server
                status = "Waiting for the capture helper"
                Log.i(TAG, "$status on port $HELPER_PORT")
                server.accept().use { sock ->
                    Log.i(TAG, "helper connected")
                    status = "Buffering…"
                    val input = DataInputStream(sock.getInputStream())
                    val bytes = ByteArray(RATE / 10 * 4) // 100 ms of s16le stereo
                    val x = FloatArray(bytes.size / 2)
                    while (running) {
                        input.readFully(bytes)
                        val sb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        for (i in x.indices) x[i] = sb.get(i) / 32768f
                        hopper.push(x, x.size)
                    }
                }
            }
        } catch (e: IOException) {
            Log.i(TAG, "helper stream ended: $e")
        }
        if (running) fail("Capture helper disconnected")
    }

    private fun readProjection(hop: Int) {
        val config = AudioPlaybackCaptureConfiguration.Builder(projection!!)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .excludeUid(Process.myUid()) // never capture our own playback
            .build()
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(RATE)
            .setChannelMask(AudioFormat.CHANNEL_IN_STEREO)
            .build()
        val rec = try {
            AudioRecord.Builder()
                .setAudioPlaybackCaptureConfig(config)
                .setAudioFormat(format)
                .setBufferSizeInBytes(RATE * 2 * 4)
                .build()
        } catch (e: Exception) {
            return fail("Capture failed: $e")
        }
        record = rec
        rec.startRecording()
        status = "Buffering…"
        val hopper = Hopper(hop)
        val x = FloatArray(RATE / 10 * 2)
        while (running) {
            val got = rec.read(x, 0, x.size, AudioRecord.READ_BLOCKING)
            if (got <= 0) break
            hopper.push(x, got)
        }
    }

    /** Player thread: 20 ms blocks at the current levels, limiter one block ahead. */
    private fun play() {
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(RATE)
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .build()
        val track = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(format)
            .setBufferSizeInBytes(RATE * 2 * 4 / 4)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track.play()
        val mixer = Native.mixerNew()
        var bass = FloatArray(0)
        var band = FloatArray(0)
        var pos = 0 // frames already played from the front of bass/band
        val out = FloatArray(2 * BLOCK)
        try {
            while (running) {
                var avail = bass.size / 2 - pos
                if (avail < 2 * BLOCK) {
                    val st = if (avail < BLOCK) stems.poll(200, TimeUnit.MILLISECONDS) else stems.poll()
                    if (st != null) {
                        bass = bass.copyOfRange(2 * pos, bass.size) + st.bass
                        band = band.copyOfRange(2 * pos, band.size) + st.band
                        pos = 0
                        avail = bass.size / 2
                    }
                }
                if (avail < BLOCK) continue
                val ahead = minOf(BLOCK, avail - BLOCK)
                Native.mixerMix(mixer, bass, band, pos, BLOCK, ahead, Control.bass, Control.band, out)
                track.write(out, 0, out.size, AudioTrack.WRITE_BLOCKING)
                pos += BLOCK
            }
        } finally {
            track.release()
        }
    }

    override fun onDestroy() {
        running = false
        if (!status.contains("failed") && !status.startsWith("No model") && !status.contains("disconnected")) status = "Off"
        try { unregisterReceiver(commands) } catch (_: Exception) {}
        try { serverSocket?.close() } catch (_: Exception) {} // the helper exits on this
        record?.let { try { it.stop() } catch (_: Exception) {}; it.release() }
        projection?.stop()
        hops.clear()
        stems.clear()
        if (source == "helper") {
            // Dropping the helper's mix moves the source back to the speaker, and
            // Spotify pauses itself when its route changes: press play for it.
            val am = applicationContext.getSystemService(AudioManager::class.java)
            Handler(Looper.getMainLooper()).postDelayed({
                for (a in intArrayOf(KeyEvent.ACTION_DOWN, KeyEvent.ACTION_UP)) {
                    am.dispatchMediaKeyEvent(KeyEvent(a, KeyEvent.KEYCODE_MEDIA_PLAY))
                }
            }, 1200)
        }
        Control.save(this)
        tileChanged()
        super.onDestroy()
    }
}
