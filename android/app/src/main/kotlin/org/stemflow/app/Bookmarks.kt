package org.stemflow.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Saved loops: the recording (title, artist, album, the player's media id) and the
 * loop in song time, with its bars, so recalling one seeks the player there and lays
 * the same loop again. Kept in `bookmarks.json` in the app's files.
 *
 * The beats around the loop are kept too (song ms; the loop is beats[a]..beats[b],
 * with a bar line on beats[one]), so recalling lays them straight back without
 * analysing. Older bookmarks get theirs the first time they're recalled.
 *
 * Compatibility: each entry says its version (`v`; none = 1). Reading takes every
 * field with a fallback, so older entries load, and is otherwise version-blind: a
 * later version adds fields or new names, never changes what an existing one means.
 * Writing merges into the entry as read, so fields this version doesn't know (from a
 * newer app) survive a rename or upgrade; an entry that doesn't parse is kept as is;
 * and a file that doesn't parse is set aside before anything is written over it.
 *
 *   v1  the recording, the loop in song ms, meter, bars, feel
 *   v2  + beats (song ms), a, b, one, bpm: the grid, so recall needn't analyse
 */
object Bookmarks {
    class Bookmark(
        val name: String,
        val pkg: String, val title: String, val artist: String, val album: String, val mediaId: String,
        val startMs: Long, val endMs: Long, val bar1Ms: Long,
        val meter: Int, val bars: Int, val feel: Float,
        val created: Long = System.currentTimeMillis(),
        val beats: DoubleArray = DoubleArray(0), val a: Int = 0, val b: Int = 0, val one: Int = 0, val bpm: Float = 0f,
        /** The entry as read, fields unknown to this version included. */
        private val raw: JSONObject = JSONObject(),
    ) {
        val hasGrid get() = beats.size > 1 && a < b && b < beats.size

        fun withGrid(g: Grid) = Bookmark(name, pkg, title, artist, album, mediaId, startMs, endMs, bar1Ms,
            meter, bars, feel, created, g.beats, g.a, g.b, g.one, g.bpm, raw)

        fun named(n: String) = Bookmark(n, pkg, title, artist, album, mediaId, startMs, endMs, bar1Ms,
            meter, bars, feel, created, beats, a, b, one, bpm, raw)

        val song get() = NowPlaying.Song(pkg, title, artist, album, mediaId)

        fun json(): JSONObject = JSONObject(raw.toString())
            .put("v", if (hasGrid) maxOf(VERSION, raw.optInt("v", 1)) else raw.optInt("v", 1))
            .put("name", name).put("pkg", pkg).put("title", title).put("artist", artist).put("album", album)
            .put("mediaId", mediaId).put("startMs", startMs).put("endMs", endMs).put("bar1Ms", bar1Ms)
            .put("meter", meter).put("bars", bars).put("feel", feel.toDouble()).put("created", created)
            .apply {
                if (hasGrid) put("beats", JSONArray().apply { beats.forEach { put(Math.round(it * 10) / 10.0) } })
                    .put("a", a).put("b", b).put("one", one).put("bpm", bpm.toDouble())
            }

        companion object {
            fun of(o: JSONObject) = Bookmark(
                o.getString("name"), o.optString("pkg"), o.optString("title"), o.optString("artist"),
                o.optString("album"), o.optString("mediaId"), o.getLong("startMs"), o.getLong("endMs"),
                o.optLong("bar1Ms", o.getLong("startMs")), o.optInt("meter", 4), o.optInt("bars", 4),
                o.optDouble("feel", 1.0).toFloat(), o.optLong("created"),
                o.optJSONArray("beats")?.let { j -> DoubleArray(j.length()) { j.getDouble(it) } } ?: DoubleArray(0),
                o.optInt("a"), o.optInt("b"), o.optInt("one"), o.optDouble("bpm", 0.0).toFloat(), o,
            )
        }
    }

    /** Beats in song ms, the loop as indices into them, a bar line, the tempo. */
    class Grid(val beats: DoubleArray, val a: Int, val b: Int, val one: Int, val bpm: Float)

    /** The newest entry version this app writes. */
    const val VERSION = 2

    private fun file(c: Context) = File(c.filesDir, "bookmarks.json")

    /** Every entry as stored, with the bookmark it holds (null if it doesn't parse). */
    private fun load(c: Context): List<Pair<JSONObject, Bookmark?>> {
        val f = file(c)
        if (!f.exists()) return emptyList()
        val a = try {
            JSONArray(f.readText())
        } catch (e: Exception) {
            // Unreadable: keep it aside rather than save over it.
            f.renameTo(File(c.filesDir, "bookmarks.json.bad-${System.currentTimeMillis()}"))
            return emptyList()
        }
        return (0 until a.length()).mapNotNull { i ->
            val o = a.optJSONObject(i) ?: return@mapNotNull null
            o to try { Bookmark.of(o) } catch (e: Exception) { null }
        }
    }

    fun all(c: Context): List<Bookmark> = load(c).mapNotNull { it.second }

    /** Rewrite the file: each entry through `f` (null drops it), unparsed ones kept. */
    private fun edit(c: Context, extra: Bookmark? = null, f: (Bookmark) -> Bookmark? = { it }) {
        val a = JSONArray()
        for ((o, b) in load(c)) {
            if (b == null) a.put(o) else f(b)?.let { a.put(it.json()) }
        }
        extra?.let { a.put(it.json()) }
        val tmp = File(c.filesDir, "bookmarks.json.tmp")
        tmp.writeText(a.toString(1))
        if (!tmp.renameTo(file(c))) throw java.io.IOException("couldn't write bookmarks.json")
    }

    fun add(c: Context, b: Bookmark) = edit(c, extra = b)

    fun rename(c: Context, b: Bookmark, name: String) = replace(c, b.named(name))

    /** Write `b` over the bookmark saved at the same time. */
    fun replace(c: Context, b: Bookmark) = edit(c) { if (it.created != b.created) it else b }

    fun remove(c: Context, b: Bookmark) = edit(c) { if (it.created != b.created) it else null }

    fun time(ms: Long) = "%d:%02d".format(ms / 60000, ms / 1000 % 60)
}
