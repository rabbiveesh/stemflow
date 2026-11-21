package org.stemflow.app

/**
 * Buffer frame `frame` is song position `songMs` of `song`, as the player said when
 * that frame's input arrived (`at`: nanoTime then).
 */
class Anchor(val frame: Long, val songMs: Long, val song: NowPlaying.Song, val at: Long = System.nanoTime()) {
    fun frameOf(ms: Long) = frame + (ms - songMs) * SongMap.RATE / 1000
    fun msOf(f: Double) = songMs + ((f - frame) * 1000 / SongMap.RATE).toLong()
    override fun toString() = "%s @ %d ms = frame %d".format(song.title, songMs, frame)
}

/**
 * Where the song is, along the loop buffer: a run of segments, each a stretch of
 * frames that is one unbroken play of one song. A seek, a pause, a skip, the next
 * track, frames the buffer never got: each ends a segment. An anchor maps frames
 * only inside its own segment; mapping across a break is how loops and bookmarks
 * came out at the wrong time (or a negative one).
 *
 * A segment runs from its first anchor to its last (the newest one runs on). What
 * lies between two segments isn't known: the break happened somewhere in there.
 *
 * Pure (no Android): unit-tested on the laptop.
 */
class SongMap {
    companion object {
        const val RATE = 44_100
        /** How far the song time may drift from the frames before it is a break. */
        const val TOLERANCE_MS = 600L
    }

    /**
     * `floorMs`: when known, the song time this stretch began at (we seeked there, or
     * the next track began at 0), so it reaches back before its first anchor to there,
     * but never past `floorFrame` (the previous stretch's last anchor).
     */
    private class Segment(val anchors: ArrayList<Anchor>, val floorMs: Long? = null, val floorFrame: Long = Long.MIN_VALUE) {
        val first get() = anchors.first()
        val last get() = anchors.last()
        /** The first frame and song ms this stretch maps. */
        val startFrame: Long get() = floorMs?.let { maxOf(first.frameOf(it), floorFrame + 1) }?.coerceAtMost(first.frame) ?: first.frame
        val startMs: Long get() = first.msOf(startFrame.toDouble())
    }

    /** What `observe` decided, for the log and for marking the buffer. */
    class Seen(val gapFrom: Long, val gapTo: Long, val why: String)

    private val segments = ArrayList<Segment>()
    /** Set by `cut`: the next anchor starts a segment whatever it says. */
    private var cutWhy: String? = null
    private var cutFloorMs: Long? = null

    @get:Synchronized val latest: Anchor? get() = segments.lastOrNull()?.last

    /**
     * A break we know of (we seeked, frames were skipped, the source changed);
     * `toMs` where the song carries on from, when known (a seek's target).
     */
    @Synchronized fun cut(why: String, toMs: Long? = null) {
        if (segments.isNotEmpty()) {
            cutWhy = why
            cutFloorMs = toMs
        }
    }

    /**
     * The player says buffer frame `a.frame` is at `a.songMs`. Null if that carries
     * on from the last anchor; else the frames [gapFrom, gapTo) to mark as a gap.
     */
    @Synchronized fun observe(a: Anchor): Seen? {
        val seg = segments.lastOrNull()
        if (seg == null) {
            segments.add(Segment(arrayListOf(a)))
            return null
        }
        val p = seg.last
        val why = when {
            cutWhy != null -> cutWhy
            !p.song.same(a.song) -> "next song (${a.song.title})"
            a.frame < p.frame -> "frames went back %d".format(p.frame - a.frame)
            else -> {
                val expected = p.songMs + (a.frame - p.frame) * 1000 / RATE
                val off = a.songMs - expected
                if (kotlin.math.abs(off) > TOLERANCE_MS) "song at %d ms, expected %d (%+d ms)".format(a.songMs, expected, off)
                else null
            }
        }
        val floor = when {
            cutWhy != null -> cutFloorMs?.takeIf { it <= a.songMs && it >= 0 }
            // the next track, caught near its start: it began at 0
            !p.song.same(a.song) && a.songMs in 0..15_000 -> 0L
            else -> null
        }
        cutWhy = null
        cutFloorMs = null
        if (why == null) {
            seg.anchors.add(a)
            // keep it small: the newest anchors and one per few seconds before them
            if (seg.anchors.size > 400) thin(seg)
            return null
        }
        val seg2 = Segment(arrayListOf(a), floor, p.frame)
        segments.add(seg2)
        // the gap: from the last anchor before to where this stretch is known to begin
        return Seen(minOf(p.frame, a.frame), maxOf(p.frame, seg2.startFrame), why)
    }

    private fun thin(seg: Segment) {
        val keep = ArrayList<Anchor>()
        var lastKept: Long? = null
        for ((i, x) in seg.anchors.withIndex()) {
            if (lastKept == null || i >= seg.anchors.size - 50 || x.frame - lastKept >= 3L * RATE) {
                keep.add(x)
                lastKept = x.frame
            }
        }
        seg.anchors.clear()
        seg.anchors.addAll(keep)
    }

    /** Forget segments that ended before buffer frame `oldest`. */
    @Synchronized fun prune(oldest: Long) {
        while (segments.size > 1 && segments[0].last.frame < oldest) segments.removeAt(0)
    }

    @Synchronized fun clear() {
        segments.clear()
        cutWhy = null
    }

    private fun contains(i: Int, f: Double): Boolean {
        val s = segments[i]
        return f >= s.startFrame && (i == segments.size - 1 || f <= s.last.frame)
    }

    /** The anchor nearest to frame `f` in its segment (the newest segment wins). */
    private fun nearest(i: Int, f: Double): Anchor =
        segments[i].anchors.minBy { kotlin.math.abs(it.frame - f) }

    /** An anchor that maps all of frames [from, to], or null if a break lies between. */
    @Synchronized fun forFrames(from: Double, to: Double): Anchor? {
        for (i in segments.indices.reversed()) {
            if (contains(i, from) && contains(i, to)) return nearest(i, (from + to) / 2)
        }
        return null
    }

    /** Song ms of frame `f`, or null outside every segment. */
    @Synchronized fun msOf(f: Double): Long? = forFrames(f, f)?.msOf(f)

    /**
     * An anchor that maps song ms [fromMs, toMs] of `song` to frames, all inside one
     * segment (the newest that has them); null if no stretch of the buffer has both.
     * The newest segment counts as reaching up to its last anchor only: what plays
     * after that isn't in the buffer yet. With `after` (nanoTime), only segments that
     * began then or later count (what was buffered since a seek, say).
     */
    @Synchronized fun forSong(song: NowPlaying.Song, fromMs: Long, toMs: Long, after: Long = Long.MIN_VALUE): Anchor? {
        for (s in segments.reversed()) {
            if (s.first.at < after) break // older segments began even earlier
            if (!s.first.song.same(song)) continue
            if (fromMs >= s.startMs && toMs <= s.last.songMs) {
                val mid = (fromMs + toMs) / 2
                return s.anchors.minBy { kotlin.math.abs(it.songMs - mid) }
            }
        }
        return null
    }

    /** The newest segment's anchors began after nanoTime `t` (a seek, say). */
    @Synchronized fun startedAfter(t: Long): Boolean = (segments.lastOrNull()?.first?.at ?: Long.MIN_VALUE) >= t

    /** For the log: each segment's song, frames and song times. */
    @Synchronized fun describe(): String = segments.joinToString(" | ") {
        "%s: frames %d–%d = %d–%d ms (%d anchors%s)".format(
            it.first.song.title.take(24), it.startFrame, it.last.frame, it.startMs, it.last.songMs, it.anchors.size,
            it.floorMs?.let { f -> ", from $f ms" } ?: "")
    }
}
