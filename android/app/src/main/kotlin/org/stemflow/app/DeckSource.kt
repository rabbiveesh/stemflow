package org.stemflow.app

import android.content.Context
import android.content.Intent
import android.media.session.MediaController
import android.media.session.PlaybackState
import android.os.SystemClock
import android.view.KeyEvent
import kotlin.concurrent.thread

/**
 * The player, as the loop deck (src/loopdeck, in Rust) reaches it: its media session.
 * The deck calls these from its own threads, holding its lock: they must never call
 * back into the deck.
 */
class DeckSource(context: Context) {
    private val c = context.applicationContext

    /** Our last seek: the player's reports from before it are stale for a moment. */
    @Volatile private var seekedAt = 0L
    @Volatile private var seekedTo = 0L

    private fun ctl(): MediaController? = NowPlaying.controller(c)

    /**
     * What plays, and where: id, title, artist, album, app, ms, playing (1/0) and
     * length ms, tab-separated; null when the player doesn't say.
     */
    fun now(): String? {
        val m = ctl() ?: return null
        val s = NowPlaying.song(m) ?: return null
        var ms = NowPlaying.positionMs(m) ?: return null
        val playing = NowPlaying.playing(m)
        // Right after our seek the player can still report the old place, which would
        // map the new audio to it. For a few seconds, take where we sent it instead.
        val since = SystemClock.elapsedRealtime() - seekedAt
        if (since < 3_000 && (NowPlaying.reportedAt(m) ?: Long.MAX_VALUE) < seekedAt) {
            ms = seekedTo + if (playing) since else 0
        }
        val len = m.metadata?.getLong(android.media.MediaMetadata.METADATA_KEY_DURATION) ?: 0L
        val clean = { x: String -> x.replace('\t', ' ').replace('\n', ' ') }
        return listOf(clean(s.mediaId), clean(s.title), clean(s.artist), clean(s.album), s.pkg,
            ms.toString(), if (playing) "1" else "0", len.coerceAtLeast(0).toString()).joinToString("\t")
    }

    fun pause(): Boolean = key(KeyEvent.KEYCODE_MEDIA_PAUSE)

    fun play(): Boolean = key(KeyEvent.KEYCODE_MEDIA_PLAY)

    fun seek(ms: Long): Boolean {
        val m = ctl() ?: return false
        val to = ms.coerceAtLeast(0)
        Log.i(CaptureService.TAG, "seek: %s -> %d ms".format(NowPlaying.positionMs(m)?.let { "$it ms" } ?: "?", to))
        seekedTo = to
        seekedAt = SystemClock.elapsedRealtime()
        m.transportControls.seekTo(to)
        return true
    }

    /**
     * Play a saved loop's recording. Ask the player's session first (no app switch);
     * Spotify ignores that from apps it doesn't know, so after a few seconds open its
     * link instead (the player comes to the front), and come back once it plays.
     */
    fun open(id: String, app: String, what: String): Boolean {
        val m = ctl() ?: return false
        val can = m.playbackState?.actions ?: 0L
        val tc = m.transportControls
        Log.i(CaptureService.TAG, "open $id in ${app.ifEmpty { m.packageName }} (actions $can)")
        when {
            id.contains(':') && can and PlaybackState.ACTION_PLAY_FROM_URI != 0L ->
                tc.playFromUri(android.net.Uri.parse(id), null)
            id.isNotEmpty() && can and PlaybackState.ACTION_PLAY_FROM_MEDIA_ID != 0L ->
                tc.playFromMediaId(id, null)
            can and PlaybackState.ACTION_PLAY_FROM_SEARCH != 0L ->
                tc.playFromSearch(what, null)
        }
        if (!id.contains(':')) return true
        val pkg = app.ifEmpty { m.packageName }
        thread(name = "stemflow-open") {
            Thread.sleep(3_000)
            if (playingIt(id)) return@thread
            if (!openLink(pkg, id)) return@thread
            val until = SystemClock.elapsedRealtime() + 30_000
            while (!playingIt(id) && SystemClock.elapsedRealtime() < until) Thread.sleep(200)
            Thread.sleep(500)
            // Android won't let an app in the background start its own activity, but
            // the Shizuku helper (the shell) may; else the user switches back.
            if (Shizu.front(c)) Log.i(CaptureService.TAG, "open: back to stemflow")
        }
        return true
    }

    private fun playingIt(id: String) = ctl()?.let { NowPlaying.song(it) }?.mediaId == id

    private fun openLink(pkg: String, id: String): Boolean {
        val i = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(id))
            .setPackage(pkg).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return try {
            c.startActivity(i)
            Log.i(CaptureService.TAG, "open: link $id")
            true
        } catch (e: Exception) {
            Log.e(CaptureService.TAG, "open: link failed: ${e.message}")
            false
        }
    }

    /** Through the session when we can read it, else as a media key. */
    private fun key(code: Int): Boolean {
        CaptureService.mediaKey(c, code)
        return true
    }
}
