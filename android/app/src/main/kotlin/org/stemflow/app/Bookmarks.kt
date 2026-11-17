package org.stemflow.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Saved loops: the recording (title, artist, album, the player's media id) and the
 * loop in song time, with its bars, so recalling one seeks the player there and lays
 * the same loop again. Kept in `bookmarks.json` in the app's files.
 */
object Bookmarks {
    class Bookmark(
        val name: String,
        val pkg: String, val title: String, val artist: String, val album: String, val mediaId: String,
        val startMs: Long, val endMs: Long, val bar1Ms: Long,
        val meter: Int, val bars: Int, val feel: Float,
        val created: Long = System.currentTimeMillis(),
    ) {
        val song get() = NowPlaying.Song(pkg, title, artist, album, mediaId)

        fun json(): JSONObject = JSONObject()
            .put("name", name).put("pkg", pkg).put("title", title).put("artist", artist).put("album", album)
            .put("mediaId", mediaId).put("startMs", startMs).put("endMs", endMs).put("bar1Ms", bar1Ms)
            .put("meter", meter).put("bars", bars).put("feel", feel.toDouble()).put("created", created)

        companion object {
            fun of(o: JSONObject) = Bookmark(
                o.getString("name"), o.optString("pkg"), o.optString("title"), o.optString("artist"),
                o.optString("album"), o.optString("mediaId"), o.getLong("startMs"), o.getLong("endMs"),
                o.optLong("bar1Ms", o.getLong("startMs")), o.optInt("meter", 4), o.optInt("bars", 4),
                o.optDouble("feel", 1.0).toFloat(), o.optLong("created"),
            )
        }
    }

    private fun file(c: Context) = File(c.filesDir, "bookmarks.json")

    fun all(c: Context): List<Bookmark> = try {
        val a = JSONArray(file(c).readText())
        (0 until a.length()).map { Bookmark.of(a.getJSONObject(it)) }
    } catch (e: Exception) {
        emptyList()
    }

    private fun write(c: Context, list: List<Bookmark>) {
        val a = JSONArray()
        list.forEach { a.put(it.json()) }
        val tmp = File(c.filesDir, "bookmarks.json.tmp")
        tmp.writeText(a.toString(1))
        tmp.renameTo(file(c))
    }

    fun add(c: Context, b: Bookmark) = write(c, all(c) + b)

    fun rename(c: Context, b: Bookmark, name: String) = write(c, all(c).map {
        if (it.created != b.created) it else Bookmark(name, it.pkg, it.title, it.artist, it.album, it.mediaId,
            it.startMs, it.endMs, it.bar1Ms, it.meter, it.bars, it.feel, it.created)
    })

    fun remove(c: Context, b: Bookmark) = write(c, all(c).filter { it.created != b.created })

    fun time(ms: Long) = "%d:%02d".format(ms / 60000, ms / 1000 % 60)
}
