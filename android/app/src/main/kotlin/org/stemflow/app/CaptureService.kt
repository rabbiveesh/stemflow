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
import android.media.session.PlaybackState
import android.os.Process
import android.os.SystemClock
import android.service.quicksettings.TileService
import android.content.ComponentName
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
 * Looping (src/looper.rs): the player keeps the last minutes of stems and plays
 * from the loop instead of live while one plays. What a loop does (finding the bars,
 * extending, pausing the player, bookmarks) is the desktop's deck (src/loopdeck, in
 * Rust): this service feeds it where the player and the speaker are (`markAnchor`,
 * `deckHeard`), and the screens send it the loop panel's lines (Loops.kt).
 *
 * Control over adb (same receiver the notification buttons use):
 *   adb shell am broadcast -a org.stemflow.app.CMD -p org.stemflow.app --es snap full
 *   ... --ef balance -0.5     ... --ef bass 1.5 --ef band 0
 *   ... --ei loop 4   ... --ez live true   ... --es save NAME
 *   ... --es loopcmd "move 0 4"   (any loop panel line, src/loopdeck/command.rs)
 */
class CaptureService : Service() {
    companion object {
        const val TAG = "stemflow"
        const val RATE = 44100
        const val HELPER_PORT = 28711
        const val CMD = "org.stemflow.app.CMD"
        const val XFADE_S = 0.25f
        const val BLOCK = RATE / 50 // 20 ms, as the desktop's limiter assumes
        const val CLICK_LEVEL = 0.35f
        /** Seconds of input kept waiting for the separator while a loop plays. */
        const val MAX_BACKLOG_S = 60
        /** The loop deck (src/loopdeck), 0 until its beat model is loaded. */
        @Volatile var deck = 0L; private set
        /** Its looper, for the audio threads' looper* calls and the loop view. */
        @Volatile var looper = 0L; private set
        /** "" or what the app itself has to say about looping (loading it, re-stemming). */
        @Volatile var loopStatus = ""
        // Where the speaker is in the live stream: live frame = AudioTrack frame + offset
        // (the offset changes when a loop has played in between).
        @Volatile private var track: AudioTrack? = null
        @Volatile private var trackOffset = 0L

        @Volatile private var writtenFrames = 0L
        /** Stem frames the player took from the separator, and the looper's total after
         *  its last push: the same numbers once the looper is up (it is padded). */
        @Volatile private var polledFrames = 0L
        @Volatile private var pushedFrames = 0L

        /** The track frame being heard now (its timestamp, so the device and Bluetooth
         * delay are included), or -1. */
        private fun heardTrackFrame(): Long {
            val t = track ?: return -1
            val ts = android.media.AudioTimestamp()
            return if (t.getTimestamp(ts)) {
                ts.framePosition + (System.nanoTime() - ts.nanoTime) * RATE / 1_000_000_000L
            } else {
                t.playbackHeadPosition.toLong()
            }
        }

        /** The live stem frame being heard now, or -1. */
        private fun heardFrame(): Long {
            val f = heardTrackFrame()
            return if (f < 0) -1 else f + trackOffset
        }

        /**
         * Frames written to the speaker but not heard yet: the track's buffer, the
         * device, and on Bluetooth the headphones' delay. The screen draws the playhead
         * this far behind where the loop is being read.
         */
        fun unheardFrames(): Long {
            val f = heardTrackFrame()
            return if (f < 0) 0 else (writtenFrames - f).coerceIn(0, 2L * RATE)
        }

        /** The buffer broke here: frames skipped or dropped, the model switched. */
        private fun cut(why: String) {
            val d = deck
            if (d != 0L) Native.deckCut(d, why)
        }

        /** A bookmark tapped while off: recalled once looping is up and audio flows. */
        @Volatile private var pendingRecall = 0L
        @Volatile private var pendingSince = 0L

        fun recallWhenOn(created: Long) {
            pendingRecall = created
            pendingSince = System.currentTimeMillis()
            Log.i(TAG, "recall $created: once stemflow is on")
        }

        /**
         * Play or pause the player: through its media session when we can read it (a
         * media key goes nowhere once Android has dropped the media button session,
         * which it does after the player sits paused a while), else as a media key.
         */
        fun mediaKey(c: Context, code: Int) {
            val tc = NowPlaying.controller(c)?.transportControls
            val name = KeyEvent.keyCodeToString(code)
            if (tc != null) when (code) {
                KeyEvent.KEYCODE_MEDIA_PLAY -> { tc.play(); Log.i(TAG, "player: play"); return }
                KeyEvent.KEYCODE_MEDIA_PAUSE -> { tc.pause(); Log.i(TAG, "player: pause"); return }
            }
            Log.i(TAG, "player: $name as a media key${if (tc == null) " (no session to talk to)" else ""}")
            val am = c.applicationContext.getSystemService(AudioManager::class.java)
            for (a in intArrayOf(KeyEvent.ACTION_DOWN, KeyEvent.ACTION_UP)) am.dispatchMediaKeyEvent(KeyEvent(a, code))
        }

        @Volatile private var restemWanted = false
        @Volatile private var restemBusy = false
        private var restemCheck = 0L

        /**
         * The threads that use the looper (or the engine) besides the worker's own:
         * switching off waits for them before freeing either, else they crash on it.
         */
        private val helpers = java.util.concurrent.ConcurrentLinkedQueue<Thread>()
        private fun background(name: String, f: () -> Unit): Thread =
            Thread({ try { f() } finally { helpers.remove(Thread.currentThread()) } }, name)
                .also { helpers.add(it); it.start() }

        /** The last hop of input was silence: the source is paused (or between songs). */
        @Volatile var sourceSilent = false

        /** Is the source playing? Its media session says, else whether we hear it. */
        fun sourcePlaying(c: Context): Boolean =
            NowPlaying.controller(c)?.let { NowPlaying.playing(it) } ?: (running && !sourceSilent)

        /** The notification's Pause/Play: through the deck when it's up (it keeps a
         *  loop's own pause apart), else straight to the player. */
        fun playPause(c: Context) {
            val playing = sourcePlaying(c)
            Log.i(TAG, "play/pause: source was ${if (playing) "playing" else "paused"}")
            if (deck != 0L) Loops.send(c, "playpause")
            else mediaKey(c, if (playing) KeyEvent.KEYCODE_MEDIA_PAUSE else KeyEvent.KEYCODE_MEDIA_PLAY)
            sourceSilent = playing // until the next hop says otherwise, so the button flips now
        }

        @Volatile var running = false
        @Volatile var status = "Off"
        /** From switching on (or picking another model) until that model is ready or failed. */
        @Volatile var loading = false
        /** Path of the model that is loaded, "" if none; the UI compares it with the one picked. */
        @Volatile var loadedModel = ""; private set
        // The engine holds ~1 GB. Only the worker thread touches it: it loads the
        // model when isolation is switched on and frees it when it is switched off.
        private var engine = 0L
        private var engineModel = ""
            set(v) { field = v; loadedModel = v }
    }

    /** One hop of stems; `second` is all zeros for a one-stem model. */
    private class Stems(val bass: FloatArray, val band: FloatArray, val second: FloatArray, val plain: Boolean = false)

    private val chunks = LinkedBlockingQueue<FloatArray>() // 100 ms each, from the reader
    private val stems = LinkedBlockingQueue<Stems>()
    private var projection: MediaProjection? = null
    private var record: AudioRecord? = null
    @Volatile private var serverSocket: ServerSocket? = null
    @Volatile private var source = ""
    @Volatile private var levelDb = -120.0
    @Volatile private var computeS = 0f
    @Volatile private var late = 0
    @Volatile private var skipped = 0
    @Volatile private var delayFrames = 0
    private var loaderThread: Thread? = null
    private var playerThread: Thread? = null

    private val commands = object : BroadcastReceiver() {
        override fun onReceive(c: Context, i: Intent) {
            // The stem first: it brings in that target's own levels, which a snap or
            // level in the same command then overrides. ("karaoke" is the old name.)
            if (i.hasExtra("karaoke")) Control.setTarget(c, if (i.getBooleanExtra("karaoke", false)) "vocals" else "bass")
            i.getStringExtra("stem")?.let { if (it in Models.STEMS) Control.setTarget(c, it) }
            i.getStringExtra("snap")?.let { Control.snap(it) }
            // debugging: the player to this song position (ms), playing
            i.getStringExtra("seek")?.toLongOrNull()?.let { ms ->
                NowPlaying.controller(c)?.let { cut("seek to $ms ms (adb)"); it.transportControls.seekTo(ms); it.transportControls.play() }
            }
            // debugging: the looper's newest minute as raw f32 stereo, 44.1 kHz
            i.getStringExtra("dump")?.let {
                val h = looper
                if (h != 0L) thread {
                    val v = Native.looperRecent(h)
                    val b = java.nio.ByteBuffer.allocate(4 * v.size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    b.asFloatBuffer().put(v)
                    java.io.File(c.filesDir, "recent.f32").writeBytes(b.array())
                    Log.i(TAG, "dumped %.1f s".format(v.size / 2.0 / RATE))
                }
            }
            if (i.hasExtra("balance")) Control.setBalance(i.getFloatExtra("balance", 0f))
            if (i.hasExtra("bass") || i.hasExtra("band")) {
                Control.setLevels(i.getFloatExtra("bass", Control.bass), i.getFloatExtra("band", Control.band))
            }
            i.getStringExtra("model")?.let { id -> Models.all(c).firstOrNull { it.id == id }?.let { Models.select(c, it) } }
            // A package name, or "" for whatever is playing; used at the next start.
            if (i.hasExtra("meter")) Loops.send(c, "loop meter ${i.getIntExtra("meter", 0).coerceIn(0, 12)}")
            if (i.hasExtra("feel")) Loops.send(c, "loop feel ${i.getFloatExtra("feel", 1f)}")
            i.getStringExtra("beatmodel")?.let { Models.selectBeatModel(c, it) } // next start
            if (i.hasExtra("loop")) Loops.send(c, "loop ${i.getIntExtra("loop", 4)}") // bars
            if (i.getBooleanExtra("live", false)) Loops.send(c, "loop live")
            // any panel line: --es loopcmd "move 0 4"
            i.getStringExtra("loopcmd")?.let { Loops.send(c, "loop $it")?.let { e -> Log.w(TAG, "loopcmd: $e") } }
            if (i.hasExtra("playpause")) playPause(c)
            // debugging: save the loop as a bookmark
            i.getStringExtra("save")?.let { name -> Loops.send(c, "loop save $name") }
            Control.saveBeats(c)
            i.getStringExtra("source")?.let { pkg -> Sources.select(c, Sources.apps(c).firstOrNull { it.pkg == pkg }) }
            if (i.getBooleanExtra("stop", false)) stopSelf()
            Control.save(c)
            Log.i(TAG, "mix: ${Control.describe()} · ${Control.describeBeats()}")
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
            .setSmallIcon(R.drawable.ic_stemflow)
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(action(if (sourceSilent) "Play" else "Pause", "playpause", "1", 4))
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

        Log.init(this)
        Log.i(TAG, "--- switching on ---")
        Control.load(this)
        Control.loadBeats(this)
        // frame numbers start again with this run
        captureSilenced = false
        polledFrames = 0L
        pushedFrames = 0L
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel("live", "Bass isolation", NotificationManager.IMPORTANCE_LOW)
        )
        status = "Starting…"
        loading = true
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
        loading = false
        stopSelf()
    }

    /** Load `m` if it isn't the loaded model; false if that failed. */
    private fun load(m: Models.Model): Boolean {
        Models.applyMono(this, m)
        if (engine != 0L && engineModel == m.file.path) return true
        status = "Loading ${m.label}…"
        loading = true
        Log.i(TAG, status)
        try {
            val t = System.nanoTime()
            unload() // first, so two models are never in memory at once
            engine = Native.engineNew(m.file.path, Native.GPU_FP16, m.hop, m.lookahead, XFADE_S)
            engineModel = m.file.path
            delayFrames = Native.engineDelay(engine)
            Log.i(TAG, "model ready in %.1f s".format((System.nanoTime() - t) / 1e9))
            loading = false
            status = "Starting…"
            return true
        } catch (e: Throwable) {
            fail("Model failed: ${e.message}")
            return false
        }
    }

    /** A loop is kept: separate it again with the model just loaded, so it plays (and
     *  goes on into the tail) in the new stems. The loop keeps playing meanwhile. */
    private fun restemLoop(m: Models.Model, full: Boolean = true) {
        val h = looper
        if (h == 0L || engine == 0L) return
        loopStatus = if (full) "Re-stemming the loop with ${m.name}…" else "Separating the loop…"
        val t = System.nanoTime()
        loopStatus = try {
            if (Native.looperRestem(h, engine, full)) Log.i(TAG, "loop ${if (full) "re-stemmed" else "separated"} in %.1f s".format((System.nanoTime() - t) / 1e9))
            ""
        } catch (e: Throwable) {
            Log.e(TAG, "re-stem failed: ${e.message}")
            restemCheck = System.nanoTime() + 10_000_000_000L // not again at once
            "Couldn't re-stem the loop: ${e.message}"
        }
    }

    private fun unload() {
        if (engine != 0L) Native.engineFree(engine)
        engine = 0L
        engineModel = ""
    }

    /** Worker thread: load the model, start the other threads, then separate hops. */
    private fun work() {
        try {
            separate()
        } finally {
            loaderThread?.join()
            playerThread?.join()
            val d = deck
            looper = 0L
            deck = 0L
            // They end soon; one that doesn't keeps
            // the looper and the engine: leaking them beats freeing them under it.
            val deadline = System.nanoTime() + 10_000_000_000L
            for (t in helpers) t.join(((deadline - System.nanoTime()) / 1_000_000).coerceAtLeast(1))
            val stuck = helpers.filter { it.isAlive }.map { it.name }
            if (stuck.isEmpty()) {
                if (d != 0L) Native.deckFree(d) // and its looper
                unload()
            } else {
                Log.e(TAG, "off: $stuck still running; not freeing the looper or the model")
                engine = 0L
                engineModel = ""
            }
            loopStatus = ""
        }
    }

    /** The source went quiet or came back: the notification's Pause/Play follows. */
    private fun silence(quiet: Boolean, frame: Long) {
        sourceSilent = quiet
        silentWhilePlaying(quiet, frame)
        if (quiet == notifiedSilent) return
        notifiedSilent = quiet
        getSystemService(NotificationManager::class.java).notify(1, notification())
    }
    private var notifiedSilent = false

    /**
     * Exact silence coming in while the player says it is playing (for over a second,
     * so not the moment after pressing play): our capture was silenced, which Android
     * does to background capture while another app has the mic, say. It isn't the
     * song: no anchors meanwhile, and a break either side, so loops keep clear of it.
     */
    private fun silentWhilePlaying(quiet: Boolean, frame: Long) {
        val now = quiet && NowPlaying.controller(this)?.let {
            NowPlaying.playing(it) && SystemClock.elapsedRealtime() - (NowPlaying.reportedAt(it) ?: Long.MAX_VALUE) > 1_000
        } == true
        if (now == captureSilenced) return
        captureSilenced = now
        if (now) {
            silencedFrom = frame
            silencedAt = SystemClock.elapsedRealtime()
            val ctl = NowPlaying.controller(this)
            Log.w(TAG, "input SILENT while %s plays (at %s ms): capture silenced? mic users: %s; buffer frame %d".format(
                ctl?.packageName, ctl?.let { NowPlaying.positionMs(it) }, micUsers(), frame))
            cut("input silent while playing")
        } else {
            val h = looper
            if (h != 0L) Native.looperMarkGap(h, silencedFrom, frame)
            cut("input back after silence")
            Log.w(TAG, "input back after %d ms of silence while playing (frames %d–%d marked as a gap)".format(
                SystemClock.elapsedRealtime() - silencedAt, silencedFrom, frame))
        }
    }
    @Volatile private var captureSilenced = false
    private var silencedFrom = 0L
    private var silencedAt = 0L

    /** Who is recording now (what we can see of it), for the log. */
    private fun micUsers(): String = try {
        getSystemService(AudioManager::class.java).activeRecordingConfigurations.joinToString {
            "source %d%s".format(it.audioSource, if (it.isClientSilenced) " (silenced)" else "")
        }.ifEmpty { "none visible" }
    } catch (e: Exception) { "? (${e.message})" }

    private var lastStateLog = 0L
    /** Debug builds only: the per-hop line and the state line every 10 s are for chasing bugs, not to ship. */
    private val debuggable by lazy { applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0 }

    /** Every few seconds: where the buffer, the player and the speaker are. */
    private fun logState(frame: Long) {
        val h = looper
        val ctl = NowPlaying.controller(this)
        Log.i(TAG, "state: player frame %d, buffer %d (pushed %d, polled %d), written %d, heard %d, unheard %d ms; queues: chunks %d, stems %d; player %s at %s ms (reported %s ms ago); loop %s".format(
            frame, if (h != 0L) Native.looperTotal(h) else -1, pushedFrames, polledFrames, writtenFrames, heardFrame(),
            unheardFrames() * 1000 / RATE, chunks.size, stems.size,
            ctl?.playbackState?.state, ctl?.let { NowPlaying.positionMs(it) },
            ctl?.let { NowPlaying.reportedAt(it) }?.let { SystemClock.elapsedRealtime() - it },
            Loops.state()?.let { "playing ${it.optBoolean("playing")}, ${it.optString("status")}" } ?: "none"))
    }

    /**
     * Buffer frame `frame` is where the player is now: the deck reads the player's
     * song position against it, for the song map (src/songmap.rs).
     */
    private fun markAnchor(frame: Long) {
        val d = deck
        if (d == 0L) return
        Native.deckInput(d, frame)
        val r = pendingRecall
        if (r != 0L) {
            pendingRecall = 0L
            if (System.currentTimeMillis() - pendingSince < 60_000) {
                Log.i(TAG, "recall $r: stemflow is on")
                Loops.send(this, "loop recall $r")
            } else Log.i(TAG, "recall $r: dropped, asked for over a minute ago")
        }
        val now = System.nanoTime()
        if (debuggable && now - lastStateLog > 10_000_000_000L) {
            lastStateLog = now
            logState(frame)
        }
    }

    private fun separate() {
        // About 0.16 s of each run is CPU (STFT/iSTFT): ask for a fast core.
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        var model = Models.current(this) ?: return fail("No model in $filesDir")
        if (!load(model)) return
        if (!running) return
        startLooper()
        playerThread = thread(name = "stemflow-player") { play() }
        thread(name = "stemflow-reader") { if (source == "helper") readHelper() else readProjection() }

        var hop = Native.engineHop(engine)
        var buf = FloatArray(2 * hop)
        var fill = 0
        var processed = 0L // input frames given to the separator (= stem frames out)
        var capturing = false // the last chunk went through without the model
        while (running) {
            // The model picked in the UI (or the stem) changed: swap it in. Audio that
            // arrives meanwhile is dropped by the catch-up below.
            val want = Models.current(this)
            if (want != null && want.file.path != engineModel && !restemBusy) {
                model = want
                cut("model switch to ${want.name}")
                Log.i(TAG, "model switch at input %d; buffer at %d".format(processed, looper.let { if (it != 0L) Native.looperTotal(it) else -1 }))
                if (!load(model)) return
                restemLoop(model)
                hop = Native.engineHop(engine)
                buf = FloatArray(2 * hop)
                fill = 0
            }
            // A loop to separate (made or extended over what was only captured): while
            // only capturing, on its own thread, so capture (and listening ahead)
            // carries on; the engine is free meanwhile. The separator below waits for
            // it before using the engine again.
            val now = System.nanoTime()
            if (!restemBusy && now - restemCheck > 500_000_000L) {
                restemCheck = now
                val h = looper
                if (h != 0L && Native.looperNeedsStems(h)) restemWanted = true
            }
            if (restemWanted && !restemBusy) {
                restemWanted = false
                restemBusy = true
                val m = model
                val work = {
                    try { restemLoop(m, full = false) } finally { restemBusy = false }
                }
                if (Control.stemsOnLoop) background("stemflow-restem") {
                    Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
                    work()
                } else work()
            }
            if (Control.stemsOnLoop) {
                // Capture only: the chunk goes to the buffer and the speaker as it is
                // (in the band's place, played at full level): no model, no lag.
                val c = chunks.poll(200, TimeUnit.MILLISECONDS) ?: continue
                if (!capturing) {
                    // the separator's last `delay` frames are dropped: the buffer jumps
                    cut("separating -> capture only")
                    Log.i(TAG, "capture only from input %d (delay %d dropped)".format(processed, delayFrames))
                }
                capturing = true
                processed += c.size / 2
                silence(c.all { it > -1e-4f && it < 1e-4f }, processed)
                markAnchor(processed + chunks.size * (RATE / 10))
                val z = FloatArray(c.size)
                stems.put(Stems(z, c, z, plain = true))
                status = "%s · capturing\nStems only when looping\ninput %.0f dBFS".format(model.label, levelDb)
                continue
            }
            if (restemBusy) { Thread.sleep(50); continue }
            if (capturing) {
                // back to separating: start the stream afresh
                cut("capture only -> separating")
                Log.i(TAG, "separating from input %d (delay %d of start-up)".format(processed, delayFrames))
                capturing = false
                Native.engineReset(engine)
                buf = FloatArray(2 * hop)
                fill = 0
            }
            val hopS = hop.toFloat() / RATE
            // Fallen behind (a slow run, a warm phone, a model load): skip ahead rather
            // than stay late for good. It is heard as a jump in the music.
            val perHop = 2 * hop / (RATE / 10 * 2)
            // Not while a loop plays: nobody is hearing the live stream then, and what
            // would be dropped is the song the loop's buffer needs. It catches up after.
            val h = looper
            val keep = if (h != 0L && Native.looperIsPlaying(h)) MAX_BACKLOG_S * 10 else 2 * perHop
            if (chunks.size > keep) {
                val n = (chunks.size - keep + perHop - 1) / perHop
                cut("skipped ${n * perHop} chunks behind")
                Log.w(TAG, "behind: skipping %d chunks (%d queued, keep %d)".format(n * perHop, chunks.size, keep))
            }
            while (chunks.size > keep) {
                repeat(perHop) { chunks.poll() }
                skipped++
            }
            val c = chunks.poll(200, TimeUnit.MILLISECONDS) ?: continue
            var i = 0
            while (i < c.size) {
                val take = minOf(c.size - i, buf.size - fill)
                System.arraycopy(c, i, buf, fill, take)
                fill += take
                i += take
                if (fill < buf.size) break
                fill = 0
                val x = buf
                buf = FloatArray(2 * hop)
                processed += hop
                // Where the player is now = the newest input received (this hop and any
                // chunks queued after it), whose stems land `delay` frames later.
                val quiet = x.all { it > -1e-4f && it < 1e-4f }
                silence(quiet, processed + delayFrames)
                markAnchor(processed + chunks.size * (RATE / 10) + delayFrames)
                val bass = FloatArray(2 * hop)
                val band = FloatArray(2 * hop)
                val second = FloatArray(2 * hop)
                if (quiet) {
                    // The source is paused: play silence and leave the GPU alone.
                    stems.put(Stems(bass, band, second))
                    status = "${model.label} · paused"
                    continue
                }
                computeS = try {
                    Native.engineProcess(engine, x, bass, band, second)
                } catch (e: Throwable) {
                    return fail("Separator failed: ${e.message}")
                }
                if (computeS > hopS) late++
                stems.put(Stems(bass, band, second))
                val latency = hopS + model.lookahead + computeS + chunks.size * 0.1f
                status = "%s · %s\n%.2f s per %.1f s hop · ~%.1f s behind%s\ninput %.0f dBFS".format(
                    model.label, if (source == "helper") "helper" else "screen capture",
                    computeS, hopS, latency, if (skipped > 0) " · skipped $skipped" else "", levelDb,
                )
                if (debuggable) Log.i(TAG, "run %.3fs late=%d skipped=%d in=%.1fdBFS unheard=%d ms %s".format(computeS, late, skipped, levelDb, unheardFrames() * 1000 / RATE, Control.describe()))
            }
        }
    }

    /**
     * Load the looper's beat model on a low-priority thread: the model's CPU threads
     * are created here and keep this priority, so finding bars never starves the
     * separator. Without a beat model, everything else runs as before.
     */
    private fun startLooper() {
        val f = Models.beatModel(this) ?: return Unit.also { Log.i(TAG, "no beat model in $filesDir") }
        loaderThread = thread(name = "stemflow-looper") {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            try {
                val t = System.nanoTime()
                val l = Native.looperNew(f.path, 2)
                // the deck takes the looper over, and reaches the player through its session
                val d = Native.deckNew(l, DeckSource(this), filesDir.path, Control.meter, Control.feel)
                Loops.restore(d)
                looper = Native.deckLooper(d)
                deck = d
                Log.i(TAG, "looper (%s) ready in %.1f s".format(f.name, (System.nanoTime() - t) / 1e9))
            } catch (e: Throwable) {
                Log.e(TAG, "beat model failed: ${e.message}")
                loopStatus = "Looping unavailable: ${e.message}"
            }
        }
    }

    /** Queues the incoming stream in 100 ms chunks, and meters it. */
    private inner class Feeder {
        private var sumSq = 0.0
        private var n = 0

        fun push(x: FloatArray, len: Int) {
            chunks.put(x.copyOf(len))
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

    private fun readHelper() {
        val hopper = Feeder()
        try {
            ServerSocket(HELPER_PORT, 1, InetAddress.getLoopbackAddress()).use { server ->
                serverSocket = server
                if (Shizu.granted()) {
                    status = "Starting the capture helper…"
                    // The chosen player, or all media playback except our own output.
                    val app = Sources.current(this)
                    Log.i(TAG, "capturing ${app?.pkg ?: "all media"}")
                    Shizu.start(
                        this, HELPER_PORT,
                        if (app != null) intArrayOf(app.uid) else IntArray(0), intArrayOf(Process.myUid()),
                    )
                } else {
                    status = "Waiting for the capture helper\n${Shizu.state()}"
                }
                Log.i(TAG, "${status.lineSequence().first()} (port $HELPER_PORT)")
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

    private fun readProjection() {
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
        val hopper = Feeder()
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
        CaptureService.track = track
        val mixer = Native.mixerNew()
        var bass = FloatArray(0)
        var band = FloatArray(0)
        var sec = FloatArray(0)
        var pos = 0 // frames already played from the front of bass/band
        var played = 0L // live stem frames consumed since the start
        var written = 0L // frames written to the track
        val out = FloatArray(2 * BLOCK)
        // a loop block plus the limiter's lookahead, and its click
        val lb = FloatArray(4 * BLOCK)
        val lr = FloatArray(4 * BLOCK)
        val ls = FloatArray(4 * BLOCK)
        val lc = FloatArray(2 * BLOCK)
        try {
            var looping = false // the last block came from the looper
            while (running) {
                var avail = bass.size / 2 - pos
                if (avail < 2 * BLOCK) {
                    // While looping, never wait for live stems: a model switch or a re-stem
                    // can hold the separator for many seconds, and the loop doesn't need it.
                    val st = if (avail < BLOCK && !looping) stems.poll(200, TimeUnit.MILLISECONDS) else stems.poll()
                    if (st != null) {
                        // every live block goes into the loop buffer, played or not
                        val lh = looper
                        if (lh != 0L) {
                            // Frame numbers are stem frames since the start: the looper
                            // loads after the first stems may have arrived. Silence (a
                            // gap) stands in for those, so both count alike.
                            val missing = polledFrames - Native.looperTotal(lh)
                            if (missing > 0) {
                                Log.i(TAG, "looper is %d frames behind the stems; padding with a gap".format(missing))
                                var left = missing
                                while (left > 0) {
                                    val n = minOf(left, RATE.toLong()).toInt()
                                    val z = FloatArray(2 * n)
                                    Native.looperPush(lh, z, z, FloatArray(0))
                                    left -= n
                                }
                                Native.looperMarkGap(lh, polledFrames - missing, polledFrames)
                                cut("looper started late")
                            } else if (missing < 0) {
                                Log.w(TAG, "looper is AHEAD of the stems by %d frames".format(-missing))
                            }
                            Native.looperPush(lh, st.bass, st.band, st.second)
                            if (st.plain) Native.looperMarkPlain(lh, st.bass.size / 2)
                            pushedFrames = Native.looperTotal(lh)
                        }
                        polledFrames += st.bass.size / 2
                        bass = bass.copyOfRange(2 * pos, bass.size) + st.bass
                        band = band.copyOfRange(2 * pos, band.size) + st.band
                        sec = sec.copyOfRange(2 * pos, sec.size) + st.second
                        pos = 0
                        avail = bass.size / 2
                    }
                }
                val h = looper
                val src = if (h != 0L) Native.looperRead(h, lb, lr, ls, lc, 2 * BLOCK, BLOCK) else -1L
                looping = src >= 0
                if (looping) {
                    // Looping (or playing on after a loop): from the buffer instead of the
                    // live block, at the same levels. Live blocks that have arrived are
                    // used up at the same pace (kept in the buffer, not played).
                    trackOffset = src - written
                    // what plays here hasn't been separated yet: play it as it is
                    val plain = Native.looperPlainNow(h)
                    if (plain) Native.mixerMix(mixer, lb, lr, ls, 0, BLOCK, BLOCK, 0f, 1f, 0f, out)
                    else Native.mixerMix(mixer, lb, lr, ls, 0, BLOCK, BLOCK, Control.bass, Control.band, Control.second, out)
                    for (i in 0 until BLOCK) {
                        val c = CLICK_LEVEL * lc[i]
                        out[2 * i] = (out[2 * i] + c).coerceIn(-1f, 1f)
                        out[2 * i + 1] = (out[2 * i + 1] + c).coerceIn(-1f, 1f)
                    }
                    if (avail >= BLOCK) {
                        pos += BLOCK
                        played += BLOCK
                    }
                } else {
                    if (avail < BLOCK) continue
                    val ahead = minOf(BLOCK, avail - BLOCK)
                    if (Control.stemsOnLoop) Native.mixerMix(mixer, bass, band, sec, pos, BLOCK, ahead, 0f, 1f, 0f, out)
                    else Native.mixerMix(mixer, bass, band, sec, pos, BLOCK, ahead, Control.bass, Control.band, Control.second, out)
                    trackOffset = played - written
                    pos += BLOCK
                    played += BLOCK
                }
                written += BLOCK
                track.write(out, 0, out.size, AudioTrack.WRITE_BLOCKING)
                writtenFrames = written
                // what the speaker plays now, for "loop the last N bars" (every 100 ms)
                val d = deck
                if (d != 0L && written % (5 * BLOCK) == 0L) Native.deckHeard(d, heardFrame(), unheardFrames())
            }
        } finally {
            CaptureService.track = null
            track.release()
        }
    }

    override fun onDestroy() {
        // Read the source before the route change below: the player pauses itself on
        // it, so afterwards it always looks paused. A loop counts as not playing the
        // source: we paused it (or are about to, once the next phrase is buffered).
        val ctl = NowPlaying.controller(this)
        val st = Loops.state()
        val inLoop = st?.optBoolean("playing") == true
        val loopAt = st?.optLong("loop_from", -1)?.takeIf { inLoop && it >= 0 }
        val wasPlaying = !inLoop && sourcePlaying(this)
        Log.i(TAG, "off: source %s".format(when {
            inLoop -> "paused for a loop${loopAt?.let { " at ${Loops.time(it)}" } ?: ""}"
            wasPlaying -> "playing: resume it"
            else -> "paused: leave it"
        }))
        pendingRecall = 0L
        running = false
        loading = false
        val error = status.contains("failed") || status.startsWith("No model") || status.contains("disconnected")
        if (!error) status = "Off"
        try { unregisterReceiver(commands) } catch (_: Exception) {}
        try { serverSocket?.close() } catch (_: Exception) {} // the helper exits on this
        record?.let { try { it.stop() } catch (_: Exception) {}; it.release() }
        projection?.stop()
        Shizu.stop()
        chunks.clear()
        stems.clear()
        val app = applicationContext
        if (inLoop) {
            // Leaving a loop: keep the song quiet, and put it where the loop is, so
            // the next play starts from what was heard (not the phrase it ran on to).
            mediaKey(app, KeyEvent.KEYCODE_MEDIA_PAUSE)
            if (loopAt != null) Handler(Looper.getMainLooper()).postDelayed({
                (NowPlaying.controller(app) ?: ctl)?.transportControls?.seekTo(loopAt)
            }, 1200)
        } else if (source == "helper" && wasPlaying) {
            // Dropping the helper's mix moves the source back to the speaker, and
            // Spotify pauses itself when its route changes: press play for it, but
            // only if it was playing (else switching off would start the music).
            Handler(Looper.getMainLooper()).postDelayed({ mediaKey(app, KeyEvent.KEYCODE_MEDIA_PLAY) }, 1200)
        }
        Control.save(this)
        tileChanged()
        super.onDestroy()
    }
}
