package org.stemflow.app

import android.app.NotificationManager
import android.content.ComponentName
import android.content.Context
import android.media.MediaMetadata
import android.media.session.MediaController
import android.media.session.MediaSessionManager
import android.media.session.PlaybackState
import android.os.SystemClock
import android.service.notification.NotificationListenerService

/**
 * Exists only so the user can grant "Notification access": that is what lets an app
 * read other apps' media sessions (what is playing, where, and seek it). It never
 * looks at notifications.
 */
class MediaAccess : NotificationListenerService()

/** The player's media session: the song, the position in it, and seeking. */
object NowPlaying {
    class Song(val pkg: String, val title: String, val artist: String, val album: String, val mediaId: String) {
        /** Same recording: the media id when the player gives one, else the names. */
        fun same(o: Song) = if (mediaId.isNotEmpty() && o.mediaId.isNotEmpty()) mediaId == o.mediaId
            else title == o.title && artist == o.artist && album == o.album
    }

    fun granted(c: Context): Boolean =
        c.getSystemService(NotificationManager::class.java)
            .isNotificationListenerAccessGranted(ComponentName(c, MediaAccess::class.java))

    /** The session of the player we capture (the "From" pick), else the one playing. */
    fun controller(c: Context): MediaController? {
        if (!granted(c)) return null
        val all = try {
            c.getSystemService(MediaSessionManager::class.java)
                .getActiveSessions(ComponentName(c, MediaAccess::class.java))
        } catch (e: SecurityException) {
            return null
        }.filter { it.packageName != c.packageName && it.metadata != null }
        val pick = Sources.current(c)?.pkg
        return all.firstOrNull { it.packageName == pick }
            ?: all.firstOrNull { it.playbackState?.state == PlaybackState.STATE_PLAYING }
            ?: all.firstOrNull()
    }

    fun song(m: MediaController): Song? {
        val md = m.metadata ?: return null
        val s = { k: String -> md.getString(k) ?: "" }
        val title = s(MediaMetadata.METADATA_KEY_TITLE).ifEmpty { md.description?.title?.toString() ?: "" }
        if (title.isEmpty()) return null
        return Song(m.packageName, title, s(MediaMetadata.METADATA_KEY_ARTIST),
            s(MediaMetadata.METADATA_KEY_ALBUM), s(MediaMetadata.METADATA_KEY_MEDIA_ID))
    }

    fun playing(m: MediaController) = m.playbackState?.state == PlaybackState.STATE_PLAYING

    /** When the player last reported its position (elapsedRealtime ms), or null. */
    fun reportedAt(m: MediaController): Long? = m.playbackState?.lastPositionUpdateTime

    /** Where the player is now, ms (its last report moved on by the time since). */
    fun positionMs(m: MediaController): Long? {
        val st = m.playbackState ?: return null
        if (st.state != PlaybackState.STATE_PLAYING) return st.position
        return st.position + ((SystemClock.elapsedRealtime() - st.lastPositionUpdateTime) * st.playbackSpeed).toLong()
    }
}
