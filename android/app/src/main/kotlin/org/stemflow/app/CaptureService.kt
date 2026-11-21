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
 * Looping (src/looper.rs): the player keeps the last minutes of stems. "Loop the
 * last N bars" runs Beat This! once over the last minute on a low-priority thread,
 * then the player reads its blocks from the loop (the live stems keep arriving and
 * are kept, just not played) until "Live" goes back to the live stream.
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
        const val XFADE_S = 0.25f
        const val BLOCK = RATE / 50 // 20 ms, as the desktop's limiter assumes
        const val CLICK_LEVEL = 0.35f
        /** Seconds of input kept waiting for the separator while a loop plays. */
        const val MAX_BACKLOG_S = 60
        /** The looper (0 until its beat model is loaded); the screen draws and edits it. */
        @Volatile var looper = 0L; private set
        /** "" or what the last loop request came to (an error, for the screen). */
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
        /** Frames the looper keeps (src/looper.rs KEEP_S). */
        private const val KEEP_FRAMES = 150L * RATE

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

        /**
         * Buffer frames to song time (taken as each hop is separated: the newest input
         * is what the player is at now, and its stems land `delay` frames later in the
         * buffer), in unbroken stretches: see SongMap.
         */
        val songMap = SongMap()
        /** The newest anchor: what is playing, and where. */
        val anchor: Anchor? get() = songMap.latest
        private var anchorWhy: String? = "start"
        /** elapsedRealtime of our last seek: player reports from before it are stale. */
        @Volatile private var seekedAt = 0L

        /**
         * Every seek goes through here: it is logged (a negative request too: it is
         * clamped, and says where it came from), and it ends the buffer's stretch.
         */
        private fun seek(ctl: android.media.session.MediaController, ms: Long, why: String, play: Boolean = true) {
            if (ms < 0) Log.w(TAG, "seek: NEGATIVE %d ms asked for (%s); seeking to 0. %s".format(ms, why, songMap.describe()))
            val to = ms.coerceAtLeast(0)
            val from = NowPlaying.positionMs(ctl)
            Log.i(TAG, "seek: %s -> %d ms (%s)%s; buffer at frame %d, anchor %s".format(
                from?.let { "$it ms" } ?: "?", to, why, if (play) ", play" else "",
                looper.let { if (it != 0L) Native.looperTotal(it) else -1 }, anchor))
            songMap.cut("seek to $to ms ($why)", to)
            seekedAt = SystemClock.elapsedRealtime()
            ctl.transportControls.seekTo(to)
            if (play) ctl.transportControls.play()
        }

        /** The current loop's [start, end] in song ms, if one stretch of the song holds it. */
        fun loopSongRange(): LongArray? {
            val h = looper
            if (h == 0L) return null
            val t = Native.looperTimes(h)
            if (t.size < 6) return null
            val a = songMap.forFrames(t[0], t[1]) ?: return null
            return longArrayOf(a.msOf(t[0]), a.msOf(t[1]))
        }

        /** Save the current loop as a bookmark; an error message, or null when saved. */
        fun saveBookmark(c: Context, name: String): String? {
            val h = looper
            if (h == 0L) return "No loop"
            val t = Native.looperTimes(h)
            if (t.size < 6) return "No loop"
            if (!NowPlaying.granted(c)) return "Allow notification access first"
            // One stretch of the song must hold the whole loop (and its bar 1): an anchor
            // from across a seek, pause or song change would put it somewhere else.
            val from = minOf(t[0], t[2])
            val a = songMap.forFrames(from, t[1])
            Log.i(TAG, "bookmark %s: loop frames %.0f–%.0f bar1 %.0f, buffer %d; anchor %s; map: %s".format(
                name, t[0], t[1], t[2], Native.looperTotal(h), a, songMap.describe()))
            if (a == null) return if (anchor == null)
                "Not saved: stemflow didn't know the song when this loop was made. Press Live, then loop again."
                else "Not saved: the song was seeked, paused or changed during this loop, so its time in the song isn't known. Press Live, then loop again."
            val s = a.song
            val startMs = a.msOf(t[0])
            if (startMs < 0) {
                Log.w(TAG, "bookmark %s: NEGATIVE start %d ms; not saved".format(name, startMs))
                return "Not saved: this loop seems to start before the song does."
            }
            var bm = Bookmarks.Bookmark(
                name, s.pkg, s.title, s.artist, s.album, s.mediaId,
                startMs, a.msOf(t[1]), a.msOf(t[2]), t[3].toInt(), t[4].toInt(), t[5].toFloat(),
            )
            grid(h)?.let { bm = bm.withGrid(it) }
            Bookmarks.add(c, bm)
            Log.i(TAG, "bookmark %s: %s %d-%d ms bar1 %d, %d beats kept".format(
                name, s.title, bm.startMs, bm.endMs, bm.bar1Ms, bm.beats.size))
            return null
        }

        /**
         * The looper's beats around the loop, in song ms: only those in the loop's own
         * stretch of the song (the padding either side is dropped where a break cuts it).
         */
        private fun grid(h: Long): Bookmarks.Grid? {
            val g = Native.looperGrid(h)
            if (g.size < 7) return null
            val f = DoubleArray(g.size - 5) { g[5 + it] }
            val a0 = g[0].toInt()
            val b0 = g[1].toInt()
            if (b0 >= f.size) return null
            val a = songMap.forFrames(f[a0], f[b0]) ?: return null
            var lo = a0
            while (lo > 0 && songMap.forFrames(f[lo - 1], f[b0]) != null) lo--
            var hi = b0
            while (hi < f.size - 1 && songMap.forFrames(f[a0], f[hi + 1]) != null) hi++
            if (lo > 0 || hi < f.size - 1) Log.i(TAG, "grid: kept beats %d..%d of %d (a break cuts the rest)".format(lo, hi, f.size))
            val beats = DoubleArray(hi - lo + 1) { a.msOf(f[lo + it]).toDouble() }
            return Bookmarks.Grid(beats, a0 - lo, b0 - lo, g[2].toInt() - lo, g[4].toFloat())
        }

        /**
         * Play a bookmark: the player to that recording (by media id) and to a run-up
         * before the loop, then, once its stems are in the buffer, find the bars there
         * and put the loop where it was saved, and pause the player as `loopLast` does.
         */
        fun recall(c: Context, b: Bookmarks.Bookmark) {
            val h = looper
            if (h == 0L || !running) { loopStatus = "Switch isolation on first"; return }
            val id = ++recallId
            recalling = true
            background("stemflow-recall") {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                try {
                    recallNow(c, h, b, id)
                    loopStatus = ""
                } catch (e: RecallCancelled) {
                    Log.i(TAG, "recall of ${b.name} called off")
                    if (!recalling) loopStatus = ""
                } catch (e: Exception) {
                    Log.e(TAG, "recall failed: ${e.message}")
                    if (id == recallId) loopStatus = e.message ?: "Couldn't recall"
                } finally {
                    if (id == recallId) recalling = false
                }
            }
        }

        /** A bookmark is on its way back: the music plays live until its loop is laid. */
        @Volatile var recalling = false; private set
        @Volatile private var recallId = 0
        private class RecallCancelled : Exception()

        private fun cancelRecall() {
            recallId++
            if (recalling) loopStatus = ""
            recalling = false
        }

        private fun recallNow(c: Context, h: Long, b: Bookmarks.Bookmark, id: Int) {
            val check = { if (id != recallId) throw RecallCancelled() }
            val ctl = NowPlaying.controller(c) ?: throw Exception(
                if (NowPlaying.granted(c)) "No player found: start ${b.title} in the player" else "Allow notification access first")
            loopStatus = "Going to ${b.title} at ${Bookmarks.time(b.startMs)}…"
            Native.looperEdit(h, 8, 0, 0, 0f) // straight to live: we listen to the run-up
            val now = NowPlaying.song(ctl)
            if (now == null || !now.same(b.song)) {
                // Ask the player's session first: no app switch. Spotify ignores these
                // from apps it doesn't know, so after a few seconds open its link
                // instead (spotify:track:…), which comes to the front; come back after.
                val can = ctl.playbackState?.actions ?: 0L
                val tc = ctl.transportControls
                Log.i(TAG, "recall: open ${b.mediaId} (actions $can) from ${NowPlaying.song(ctl)?.mediaId}")
                when {
                    b.mediaId.contains(':') && can and PlaybackState.ACTION_PLAY_FROM_URI != 0L ->
                        tc.playFromUri(android.net.Uri.parse(b.mediaId), null)
                    b.mediaId.isNotEmpty() && can and PlaybackState.ACTION_PLAY_FROM_MEDIA_ID != 0L ->
                        tc.playFromMediaId(b.mediaId, null)
                    can and PlaybackState.ACTION_PLAY_FROM_SEARCH != 0L ->
                        tc.playFromSearch("${b.title} ${b.artist}".trim(), null)
                }
                loopStatus = "Opening ${b.title} in the player…"
                var linked = false
                val t0 = System.currentTimeMillis()
                var seen = ""
                while (NowPlaying.song(ctl)?.same(b.song) != true) {
                    check()
                    val cur = NowPlaying.song(ctl)?.let { "${it.title} / ${it.artist} state ${ctl.playbackState?.state}" } ?: "none"
                    if (cur != seen) { Log.i(TAG, "recall: player has $cur"); seen = cur }
                    val waited = System.currentTimeMillis() - t0
                    if (!linked && waited > 3000) {
                        linked = true
                        if (!openLink(c, b.pkg, b.mediaId)) throw Exception("Play ${b.title} in the player, then try again")
                    }
                    if (waited > 30_000) throw Exception("The player didn't open ${b.title}")
                    Thread.sleep(200)
                }
                if (linked) {
                    Thread.sleep(500)
                    comeBack(c)
                }
            }
            // With its own beats, only the stems are needed: a short run-up for the
            // separator to settle, then up to the last kept beat. Without, enough
            // before the loop for Beat This! to find the tempo.
            val own = b.hasGrid
            val first = if (own) b.beats.first().toLong() else b.startMs
            val last = if (own) b.beats.last().toLong() else b.endMs
            val runUp = if (own) 4_000L else 12_000L
            val seekTo = (first - runUp).coerceAtLeast(0)
            Log.i(TAG, "recall %s: %s start %d end %d bar1 %d ms, %s, beats %d–%d ms".format(
                b.name, b.title, b.startMs, b.endMs, b.bar1Ms, if (own) "own grid" else "no grid", first, last))
            if (b.startMs < 0 || first < 0) {
                // saved by an older version across a seek or song change: its times are wrong
                Log.w(TAG, "recall %s: NEGATIVE times in the bookmark".format(b.name))
                throw Exception("This bookmark's time in the song is broken (saved across a seek or song change). Delete it and save the loop again.")
            }
            phraseId++
            val since = System.nanoTime()
            streamSince = since
            streamFromMs = seekTo
            seek(ctl, seekTo, "recall ${b.name}")
            // one unbroken stretch, begun after the seek, from the loop's start (or its
            // first beat) to past its end
            val lo = minOf(first, b.startMs, b.bar1Ms)
            val hi = last + if (own) 500 else 1500
            val until = System.currentTimeMillis() + 60_000
            var a: Anchor
            var lastLog = 0L
            while (true) {
                check()
                if (!running) throw Exception("Stopped")
                if (System.currentTimeMillis() > until) {
                    Log.w(TAG, "recall %s: timed out; want %d–%d ms; map: %s".format(b.name, lo, hi, songMap.describe()))
                    throw Exception("Timed out waiting for the music")
                }
                Thread.sleep(250)
                val x = anchor ?: continue
                if (x.at >= since && x.song.same(b.song)) loopStatus = "Listening up to the loop… %s".format(Bookmarks.time(x.songMs))
                val now = System.currentTimeMillis()
                if (now - lastLog > 2000) {
                    lastLog = now
                    Log.i(TAG, "recall: waiting for %d–%d ms; newest %s; map: %s".format(lo, hi, x, songMap.describe()))
                }
                a = songMap.forSong(b.song, lo, hi, after = since) ?: continue
                break
            }
            // the stems up to there are in the buffer
            val need = a.frameOf(last + if (own) 300 else 1000)
            Log.i(TAG, "recall: mapped by %s; need frame %d, buffer at %d".format(a, need, Native.looperTotal(h)))
            while (Native.looperTotal(h) < need) {
                check()
                if (!running || System.currentTimeMillis() > until) throw Exception("Timed out")
                Thread.sleep(100)
            }
            check()
            if (own) {
                val frames = DoubleArray(b.beats.size) { a.frame + (b.beats[it] - a.songMs) * RATE / 1000 }
                Log.i(TAG, "recall: laying beats at frames %.0f..%.0f (loop %.0f–%.0f)".format(
                    frames.first(), frames.last(), frames[b.a.coerceIn(0, frames.size - 1)], frames[b.b.coerceIn(0, frames.size - 1)]))
                Native.looperLay(h, frames, b.a, b.b, b.one, b.meter, b.bpm, b.feel)
            } else {
                loopStatus = "Finding the bars…"
                Native.looperFind(h, a.frameOf(b.endMs), b.bars, b.meter, b.feel, if (b.bpm > 0f) b.bpm / b.feel else 0f)
                Native.looperSetTimes(h, a.frameOf(b.startMs).toDouble(), a.frameOf(b.endMs).toDouble(), a.frameOf(b.bar1Ms).toDouble())
                // keep the beats now, so next time there's nothing to find
                grid(h)?.let { Bookmarks.replace(c, b.withGrid(it)) }
            }
            if (id != recallId) { // Live was pressed while the bars were being found
                Native.looperEdit(h, 8, 0, 0, 0f)
                throw RecallCancelled()
            }
            loopChanged()
            pauseAfterPhrase(c, h)
            Log.i(TAG, "recalled %s: loop frames %s".format(b.name, Native.looperTimes(h).take(3).joinToString { "%.0f".format(it) }))
        }

        /** Open the bookmark's recording through its link (the player comes to the front). */
        private fun openLink(c: Context, pkg: String, mediaId: String): Boolean {
            if (!mediaId.contains(':')) return false
            val i = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(mediaId))
                .setPackage(pkg).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            return try {
                c.startActivity(i)
                Log.i(TAG, "recall: opened link $mediaId")
                true
            } catch (e: Exception) {
                Log.e(TAG, "recall: link failed: ${e.message}")
                false
            }
        }

        /**
         * Back to stemflow after the player came to the front. Android won't let an
         * app in the background start its own activity, but the Shizuku helper (the
         * shell) may; without it, say so and let the user switch back.
         */
        private fun comeBack(c: Context) {
            if (Shizu.front(c)) Log.i(TAG, "recall: back to stemflow")
            else loopStatus = "Switch back to stemflow: the loop is on its way"
        }

        /** A media key to whatever is playing (Spotify): we have no media session of our own. */
        private fun mediaKey(c: Context, code: Int) {
            val am = c.applicationContext.getSystemService(AudioManager::class.java)
            for (a in intArrayOf(KeyEvent.ACTION_DOWN, KeyEvent.ACTION_UP)) am.dispatchMediaKeyEvent(KeyEvent(a, code))
        }

        /**
         * The buffer holds the song unbroken since our last seek or play (`streamSince`,
         * nanoTime) from `streamFromMs`: frames there map to song time by the anchor.
         * A pause breaks it (silence goes into the buffer meanwhile), so phrases are
         * found by song time, never by buffer position.
         */
        @Volatile private var streamSince = 0L
        @Volatile private var streamFromMs = 0L
        @Volatile private var restemWanted = false
        @Volatile private var restemBusy = false

        /** A loop was made or changed: separate whatever in it is still the plain mix. */
        private fun loopChanged() {
            val h = looper
            if (h != 0L && Native.looperNeedsStems(h)) restemWanted = true
        }

        /**
         * The threads that use the looper (or the engine) besides the worker's own:
         * switching off waits for them before freeing either, else they crash on it.
         */
        private val helpers = java.util.concurrent.ConcurrentLinkedQueue<Thread>()
        private fun background(name: String, f: () -> Unit): Thread =
            Thread({ try { f() } finally { helpers.remove(Thread.currentThread()) } }, name)
                .also { helpers.add(it); it.start() }

        /** Bumped to call off whatever phrase work is running. */
        @Volatile private var phraseId = 0

        /** The anchor mapping song ms [from, to] into the buffer, if it's all there. */
        private fun covering(h: Long, song: NowPlaying.Song, from: Long, to: Long): Anchor? {
            if (from < streamFromMs) return null
            val x = songMap.forSong(song, minOf(from + 6_000, to), to, after = streamSince) ?: return null
            val total = Native.looperTotal(h)
            // inside the newest minute, which is what gets analysed
            if (total < x.frameOf(to) || x.frameOf(from) < total - 55L * RATE) return null
            // what's wanted (past the run-up) must be one unbroken stretch of the song
            if (!Native.looperContinuous(h, x.frameOf(minOf(from + 6_000, to)), x.frameOf(to))) return null
            return x
        }

        /**
         * Have the buffer hold song ms [from, to]: wait if the player is already playing
         * through it, else seek there and play. Null if called off (or the loop stopped).
         */
        private fun stream(c: Context, h: Long, id: Int, song: NowPlaying.Song, from: Long, to: Long, open: Boolean = false): Anchor? {
            covering(h, song, from, to)?.let { return it }
            val ctl = NowPlaying.controller(c) ?: throw Exception("No player found")
            // never seek in another song (the queue may have moved on past this one)
            if (NowPlaying.song(ctl)?.same(song) != true) {
                if (!open || !openLink(c, song.pkg, song.mediaId)) throw Exception("The player has moved on from ${song.title}")
                val until = System.currentTimeMillis() + 30_000
                while (NowPlaying.song(ctl)?.same(song) != true) {
                    if (id != phraseId) return null
                    if (System.currentTimeMillis() > until) throw Exception("The player didn't open ${song.title}")
                    Thread.sleep(200)
                }
                comeBack(c)
            }
            val x = anchor
            val onTheWay = NowPlaying.playing(ctl) && x != null && x.song.same(song) &&
                from >= streamFromMs && x.songMs <= to &&
                songMap.forSong(song, minOf(from + 6_000, x.songMs), x.songMs, after = streamSince) != null &&
                Native.looperContinuous(h, x.frameOf(minOf(from + 6_000, x.songMs)), x.frame)
            if (!onTheWay) {
                if (from < 0) Log.w(TAG, "stream: NEGATIVE start %d ms asked for %s (to %d)".format(from, song.title, to))
                val seekTo = from.coerceAtLeast(0)
                streamSince = System.nanoTime()
                streamFromMs = seekTo
                seek(ctl, seekTo, "stream %d–%d ms".format(from, to))
            }
            val until = System.currentTimeMillis() + 60_000 + (to - from)
            val t0 = System.currentTimeMillis()
            var lastLog = 0L
            Log.i(TAG, "stream: want %d-%d ms, %s".format(from, to, if (onTheWay) "playing through" else "seeked to $from"))
            while (true) {
                if (id != phraseId || !running || !Native.looperIsPlaying(h)) return null
                if (System.currentTimeMillis() > until) throw Exception("Timed out waiting for the music")
                covering(h, song, from, to)?.let {
                    Log.i(TAG, "stream: covered after %.1f s".format((System.currentTimeMillis() - t0) / 1e3))
                    return it
                }
                anchor?.let { if (it.at >= streamSince && !it.song.same(song)) throw Exception("The song ended before that part") }
                val now = System.currentTimeMillis()
                if (now - lastLog > 1000) {
                    lastLog = now
                    val y = anchor
                    if (y != null) Log.i(TAG, "stream: %.1f s: player %d ms, buffer %.1f s short of %d ms%s".format(
                        (now - t0) / 1e3, y.songMs, (y.frameOf(to) - Native.looperTotal(h)) / RATE.toDouble(), to,
                        if (y.at < streamSince) " (anchor older than the seek)" else ""))
                    if (now - t0 > 5_000 && (now - t0) % 5_000 < 1_100) Log.i(TAG, "stream: map: ${songMap.describe()}")
                }
                Thread.sleep(200)
            }
        }

        /** The next phrase in song ms: [the loop's end, + its length], or null. */
        private fun nextRange(h: Long): Pair<Anchor, LongArray>? {
            val t = Native.looperTimes(h)
            if (t.size < 6) return null
            val a0 = songMap.forFrames(t[0], t[1]) ?: return null.also {
                Log.i(TAG, "next phrase: the loop %.0f–%.0f isn't one stretch of the song; map: %s".format(t[0], t[1], songMap.describe()))
            }
            val end = a0.msOf(t[1])
            return a0 to longArrayOf(end, end + (end - a0.msOf(t[0])))
        }

        /**
         * With a loop set: let the source play on (unheard: the loop is what plays)
         * until the buffer has the next phrase, then pause it, so "Next phrase" finds
         * it there. The source is paused straight away without song positions.
         */
        private fun pauseAfterPhrase(c: Context, h: Long) {
            val id = ++phraseId
            val (a0, r) = nextRange(h) ?: run { mediaKey(c, KeyEvent.KEYCODE_MEDIA_PAUSE); return }
            background("stemflow-prefetch") {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                try {
                    val a = stream(c, h, id, a0.song, r[0] - 6_000, r[1] + 1_000)
                    if (a != null) {
                        Log.i(TAG, "next phrase buffered")
                        if (id == phraseId && Native.looperIsPlaying(h)) mediaKey(c, KeyEvent.KEYCODE_MEDIA_PAUSE)
                        // its beats too, so extending into it is instant; only when the
                        // buffered run reaches back into the loop (else it's a gap away)
                        val t = Native.looperTimes(h)
                        if (id == phraseId && t.size >= 6 && a.frameOf(r[0] - 6_000) <= t[1]) {
                            val t0 = System.nanoTime()
                            if (Native.looperRefresh(h)) {
                                Log.i(TAG, "beats ahead found in %.2f s".format((System.nanoTime() - t0) / 1e9))
                            }
                        }
                        return@background
                    }
                } catch (e: Exception) {
                    Log.i(TAG, "prefetch: ${e.message}")
                }
                if (id == phraseId && Native.looperIsPlaying(h)) mediaKey(c, KeyEvent.KEYCODE_MEDIA_PAUSE)
            }
        }

        /**
         * Move the loop's start by `da` beats and its end by `db` (extend, trim, or
         * move it: by a beat, a bar or a phrase). In the buffer, at once; an end past
         * what has been heard is played first (the old loop keeps playing meanwhile)
         * and placed by song time, keeping bar 1 where it was.
         */
        fun adjust(c: Context, da: Int, db: Int) {
            val h = looper
            if (h == 0L || !Native.looperIsPlaying(h) || (da == 0 && db == 0)) return
            if (Native.looperNudge(h, da, db)) {
                loopStatus = ""
                if (da < 0 || db > 0) loopChanged() // new audio in the loop
                if (db > 0) pauseAfterPhrase(c, h)
                return
            }
            // trimming past the other end: nothing to fetch
            if (da >= 0 && db <= 0) { loopStatus = "The loop can't get shorter than that"; return }
            val id = ++phraseId
            background("stemflow-adjust") {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                loopStatus = try { streamTo(c, h, id, da, db); "" } catch (e: Exception) {
                    Log.e(TAG, "adjust failed: ${e.message}"); e.message ?: "Couldn't get there"
                }
            }
        }

        /**
         * Take the loop's audio in again from the player: back to before the loop and
         * through it, then the loop is laid on the new audio (it keeps playing as it is
         * meanwhile), separated as usual.
         */
        fun repair(c: Context) {
            val h = looper
            if (h == 0L || !Native.looperIsPlaying(h)) return
            val id = ++phraseId
            background("stemflow-repair") {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                loopStatus = try { streamTo(c, h, id, 0, 0, fresh = true); "" } catch (e: Exception) {
                    Log.e(TAG, "repair failed: ${e.message}"); e.message ?: "Couldn't repair the loop"
                }
            }
        }

        private fun streamTo(c: Context, h: Long, id: Int, da: Int, db: Int, fresh: Boolean = false) {
            val t = Native.looperTimes(h)
            val v = Native.looperView(h, 1)
            if (t.size < 6 || v.size < 14) throw Exception("No loop")
            val a0 = songMap.forFrames(minOf(t[0], t[2]), t[1]) ?: run {
                Log.w(TAG, "streamTo: loop %.0f–%.0f (bar1 %.0f) not in one stretch; map: %s".format(t[0], t[1], t[2], songMap.describe()))
                throw Exception(if (anchor == null) "Needs notification access (to know where in the song this is)"
                    else "Can't tell where in the song this loop is (a seek, pause or song change runs through it)")
            }
            Log.i(TAG, "streamTo da=%d db=%d fresh=%b: loop %.0f–%.0f by %s".format(da, db, fresh, t[0], t[1], a0))
            val startMs = a0.msOf(t[0])
            val endMs = a0.msOf(t[1])
            val beatMs = (endMs - startMs).toDouble() / (v[11] - v[10]).coerceAtLeast(1f)
            val newStart = startMs + (da * beatMs).toLong()
            val newEnd = endMs + (db * beatMs).toLong()
            // bar 1: where it was, moved with the start when that moves by whole bars
            // (onto the new start itself if the loop started on bar 1)
            val m = t[3].toInt().coerceAtLeast(1)
            val bar1 = when {
                da % m != 0 -> a0.msOf(t[2])
                t[2] == t[0] -> newStart
                else -> a0.msOf(t[2]) + (da * beatMs).toLong()
            }
            loopStatus = "Listening to %s–%s…".format(Bookmarks.time(newStart), Bookmarks.time(newEnd))
            if (fresh) streamFromMs = Long.MAX_VALUE // nothing buffered counts: play it again
            val a = stream(c, h, id, a0.song, minOf(newStart, bar1) - 6_000, newEnd + 1_000, open = fresh) ?: return
            loopStatus = "Finding the bars…"
            Native.looperFind(h, a.frameOf(newEnd), t[4].toInt(), t[3].toInt(), t[5].toFloat(), v[2] / t[5].toFloat())
            Native.looperSetTimes(h, a.frameOf(newStart).toDouble(), a.frameOf(newEnd).toDouble(), a.frameOf(bar1).toDouble())
            // the length in beats is what was asked for: count it from the start, so a
            // slightly different tempo in the new analysis can't add or drop a beat
            val beats = (v[11] - v[10]).toInt() + db - da
            val w = Native.looperView(h, 1)
            if (w.size >= 14 && beats > 0) Native.looperEdit(h, 3, w[10].toInt(), w[10].toInt() + beats, 0f)
            loopChanged()
            Log.i(TAG, "loop now %s–%s".format(Bookmarks.time(newStart), Bookmarks.time(newEnd)))
            pauseAfterPhrase(c, h)
        }

        /** Back from a loop to the live stream: stop the loop and un-pause the source. */
        fun goLive(c: Context) {
            phraseId++
            cancelRecall()
            val h = looper
            if (h != 0L && Native.looperIsPlaying(h)) {
                Log.i(TAG, "back to live")
                Native.looperEdit(h, 0, 0, 0, 0f)
                mediaKey(c, KeyEvent.KEYCODE_MEDIA_PLAY)
            }
        }

        /** The last hop of input was silence: the source is paused (or between songs). */
        @Volatile var sourceSilent = false

        /** Is the source playing? Its media session says, else whether we hear it. */
        fun sourcePlaying(c: Context): Boolean =
            NowPlaying.controller(c)?.let { NowPlaying.playing(it) } ?: (running && !sourceSilent)

        /** The pause button: pause or play the source (live only; a loop has Live). */
        fun playPause(c: Context) {
            val playing = sourcePlaying(c)
            Log.i(TAG, "play/pause: source was ${if (playing) "playing" else "paused"}")
            if (playing) {
                sourceSilent = true // until the next hop says otherwise, so the button flips now
                mediaKey(c, KeyEvent.KEYCODE_MEDIA_PAUSE)
            } else {
                sourceSilent = false
                mediaKey(c, KeyEvent.KEYCODE_MEDIA_PLAY)
            }
        }

        /**
         * Loop the last `bars` bars of what is being heard (or, while looping, resize).
         * Once the loop plays, the source is paused: nobody hears the song meanwhile, and
         * the loop plays from its own copy. "Live" presses play again.
         */
        fun loopLast(c: Context, bars: Int) {
            val h = looper
            if (h == 0L) { loopStatus = "Looping isn't ready yet"; return }
            if (Native.looperIsPlaying(h)) {
                Native.looperEdit(h, 4, bars, 0, 0f)
                loopChanged()
                return
            }
            val press = heardFrame()
            if (press < 0) return
            cancelRecall()
            streamSince = 0L // live up to now: unbroken, as far as we know
            streamFromMs = 0L
            Log.i(TAG, "loop pressed: %d bars, heard frame %d (%s ms), buffer at %d, written %d, anchor %s; map: %s".format(
                bars, press, songMap.msOf(press.toDouble()), Native.looperTotal(h), writtenFrames, anchor, songMap.describe()))
            loopStatus = "Finding the bars…"
            background("stemflow-find") {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                val t = System.nanoTime()
                loopStatus = try {
                    Native.looperFind(h, press, bars, Control.meter, Control.feel)
                    Log.i(TAG, "loop found in %.2f s".format((System.nanoTime() - t) / 1e9))
                    loopChanged()
                    pauseAfterPhrase(c, h)
                    ""
                } catch (e: Throwable) {
                    Log.e(TAG, "loop failed: ${e.message}")
                    "No loop: ${e.message}"
                }
            }
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
                NowPlaying.controller(c)?.let { seek(it, ms, "adb") }
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
            if (i.hasExtra("meter")) Control.meter = i.getIntExtra("meter", 0).coerceIn(0, 12)
            if (i.hasExtra("feel")) Control.feel = i.getFloatExtra("feel", 1f)
            i.getStringExtra("beatmodel")?.let { Models.selectBeatModel(c, it) } // next start
            if (i.hasExtra("loop")) loopLast(c, i.getIntExtra("loop", 4)) // bars
            if (i.getBooleanExtra("live", false)) goLive(c)
            if (i.hasExtra("playpause")) playPause(c)
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

        Control.load(this)
        Control.loadBeats(this)
        // frame numbers start again with this run
        songMap.clear()
        anchorWhy = "start"
        seekedAt = 0L
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
            val h = looper
            looper = 0L
            // Called off (phraseId was bumped), they end soon; one that doesn't keeps
            // the looper and the engine: leaking them beats freeing them under it.
            val deadline = System.nanoTime() + 10_000_000_000L
            for (t in helpers) t.join(((deadline - System.nanoTime()) / 1_000_000).coerceAtLeast(1))
            val stuck = helpers.filter { it.isAlive }.map { it.name }
            if (stuck.isEmpty()) {
                if (h != 0L) Native.looperFree(h)
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
    private fun silence(quiet: Boolean) {
        sourceSilent = quiet
        if (quiet == notifiedSilent) return
        notifiedSilent = quiet
        getSystemService(NotificationManager::class.java).notify(1, notification())
    }
    private var notifiedSilent = false

    private var lastStateLog = 0L
    /** Debug builds only: the per-hop line and the state line every 10 s are for chasing bugs, not to ship. */
    private val debuggable by lazy { applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0 }

    /** Every few seconds: where the buffer, the player and the speaker are. */
    private fun logState(frame: Long) {
        val h = looper
        val ctl = NowPlaying.controller(this)
        Log.i(TAG, "state: anchor frame %d, buffer %d (pushed %d, polled %d), written %d, heard %d, unheard %d ms; queues: chunks %d, stems %d; player %s at %s ms (reported %s ms ago); looping %b; map: %s".format(
            frame, if (h != 0L) Native.looperTotal(h) else -1, pushedFrames, polledFrames, writtenFrames, heardFrame(),
            unheardFrames() * 1000 / RATE, chunks.size, stems.size,
            ctl?.playbackState?.state, ctl?.let { NowPlaying.positionMs(it) },
            ctl?.let { NowPlaying.reportedAt(it) }?.let { SystemClock.elapsedRealtime() - it },
            h != 0L && Native.looperIsPlaying(h), songMap.describe()))
    }

    /** Buffer frame `frame` is where the player is now (if it says). */
    private fun markAnchor(frame: Long) {
        val ctl = NowPlaying.controller(this)
        val song = ctl?.let { NowPlaying.song(it) }
        val ms = ctl?.let { NowPlaying.positionMs(it) }
        val why = when {
            ctl == null -> "no player session"
            song == null -> "${ctl.packageName}: no title"
            ms == null -> "${ctl.packageName}: no position"
            !NowPlaying.playing(ctl) -> "${ctl.packageName}: state ${ctl.playbackState?.state}"
            else -> null
        }
        // After our seek the player's report can lag: positions from before it would
        // map the new audio to the old place.
        // (For 3 s at most: a player that ignored the seek never reports again.)
        val stale = why == null && (NowPlaying.reportedAt(ctl!!) ?: Long.MAX_VALUE) < seekedAt &&
            SystemClock.elapsedRealtime() - seekedAt < 3_000
        if (why == null && !stale) {
            // The song time didn't follow the buffer (a seek, a pause, skipped input,
            // the next track): what lies between isn't continuous song audio.
            val a = Anchor(frame, ms!!, song!!)
            val seen = songMap.observe(a)
            val h = looper
            if (seen != null) {
                if (h != 0L) Native.looperMarkGap(h, seen.gapFrom, seen.gapTo)
                Log.i(TAG, "gap in the buffer: frames %d–%d: %s; buffer at %d".format(
                    seen.gapFrom, seen.gapTo, seen.why, if (h != 0L) Native.looperTotal(h) else -1))
            }
            if (h != 0L) songMap.prune(Native.looperTotal(h) - KEEP_FRAMES)
        }
        val w = if (stale) "${ctl!!.packageName}: report from before our seek" else why
        if (w != anchorWhy) { Log.i(TAG, "anchor: ${w ?: anchor}"); anchorWhy = w }
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
                songMap.cut("model switch to ${want.name}")
                Log.i(TAG, "model switch at input %d; buffer at %d".format(processed, looper.let { if (it != 0L) Native.looperTotal(it) else -1 }))
                if (!load(model)) return
                restemLoop(model)
                hop = Native.engineHop(engine)
                buf = FloatArray(2 * hop)
                fill = 0
            }
            // A loop to separate: while only capturing, on its own thread, so capture
            // (and listening ahead) carries on; the engine is free meanwhile. The
            // separator below waits for it before using the engine again.
            if (restemWanted && !restemBusy) {
                restemWanted = false
                restemBusy = true
                val m = model
                val work = {
                    try { restemLoop(m, full = false) } finally {
                        restemBusy = false
                        loopChanged() // a part edited in meanwhile, or the next plain stretch
                    }
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
                    songMap.cut("separating -> capture only")
                    Log.i(TAG, "capture only from input %d (delay %d dropped)".format(processed, delayFrames))
                }
                capturing = true
                silence(c.all { it > -1e-4f && it < 1e-4f })
                processed += c.size / 2
                markAnchor(processed + chunks.size * (RATE / 10))
                val z = FloatArray(c.size)
                stems.put(Stems(z, c, z, plain = true))
                status = "%s · capturing\nStems only when looping\ninput %.0f dBFS".format(model.label, levelDb)
                continue
            }
            if (restemBusy) { Thread.sleep(50); continue }
            if (capturing) {
                // back to separating: start the stream afresh
                songMap.cut("capture only -> separating")
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
                songMap.cut("skipped ${n * perHop} chunks behind")
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
                markAnchor(processed + chunks.size * (RATE / 10) + delayFrames)
                val bass = FloatArray(2 * hop)
                val band = FloatArray(2 * hop)
                val second = FloatArray(2 * hop)
                if (x.all { it > -1e-4f && it < 1e-4f }) {
                    // The source is paused: play silence and leave the GPU alone.
                    stems.put(Stems(bass, band, second))
                    status = "${model.label} · paused"
                    silence(true)
                    continue
                }
                silence(false)
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
                val h = Native.looperNew(f.path, 2)
                Native.looperEdit(h, 9, (Control.semis * 100).toInt(), if (Control.keepPitch) 1 else 0, Control.speed)
                looper = h
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
                                songMap.cut("looper started late")
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
        val h = looper
        val inLoop = h != 0L && Native.looperIsPlaying(h)
        val loopAt = if (inLoop) loopSongRange()?.get(0) else null
        val wasPlaying = !inLoop && sourcePlaying(this)
        phraseId++ // a pending "pause after the next phrase" must not fire once we're off
        cancelRecall()
        Log.i(TAG, "off: source %s".format(when {
            inLoop -> "paused for a loop${loopAt?.let { " at ${Bookmarks.time(it)}" } ?: ""}"
            wasPlaying -> "playing: resume it"
            else -> "paused: leave it"
        }))
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
                (NowPlaying.controller(app) ?: ctl)?.let { seek(it, loopAt, "off in a loop", play = false) }
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
