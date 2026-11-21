package org.stemflow.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class SongMapTest {
    private val R = SongMap.RATE.toLong()
    private val songA = NowPlaying.Song("spotify", "A", "x", "y", "spotify:track:a")
    private val songB = NowPlaying.Song("spotify", "B", "x", "y", "spotify:track:b")

    /** Anchors every `hop` seconds: frame f is song ms `startMs + (f - frame0) / RATE`. */
    private fun play(m: SongMap, song: NowPlaying.Song, frame0: Long, startMs: Long, seconds: Int, hop: Int = 1, at0: Long = 0L): Long {
        var f = frame0
        for (i in 0..seconds / hop) {
            f = frame0 + i * hop * R
            val seen = m.observe(Anchor(f, startMs + i * hop * 1000L, song, at0 + i))
            if (i > 0) assertNull("steady play is no break", seen) // the first may begin a stretch
        }
        return f
    }

    @Test fun steadyPlayIsOneStretch() {
        val m = SongMap()
        play(m, songA, 10 * R, 30_000, 20)
        assertEquals(35_000L, m.msOf(15.0 * R))
        assertNotNull(m.forFrames(10.0 * R, 30.0 * R))
        // the newest stretch runs on past its last anchor
        assertEquals(45_000L, m.msOf(25.0 * R + 0))
    }

    @Test fun nextSongIsABreakAndNoAnchorMapsAcrossIt() {
        val m = SongMap()
        val end = play(m, songA, 0, 200_000, 30)
        // the next track starts: song B at 1.5 s, 2 s of frames later
        val seen = m.observe(Anchor(end + 2 * R, 1_500, songB, 100))
        assertNotNull(seen)
        assertEquals(end, seen!!.gapFrom)
        // B began at 0, 1.5 s before its first anchor: the gap ends there
        assertEquals(end + R / 2, seen.gapTo)
        assertEquals(500L, m.msOf((end + R).toDouble()))
        play(m, songB, end + 2 * R, 1_500, 10, at0 = 101)
        // A loop pressed just after the change, reaching back into song A: no anchor
        // for it. (The old code mapped it with B's anchor: B at -27 s.)
        assertNull(m.forFrames((end - 25 * R).toDouble(), (end + 5 * R).toDouble()))
        // each side on its own maps by its own song
        assertEquals(songA.title, m.forFrames(10.0 * R, 20.0 * R)!!.song.title)
        assertEquals(210_000L, m.msOf(10.0 * R))
        assertEquals(songB.title, m.forFrames((end + 4 * R).toDouble(), (end + 8 * R).toDouble())!!.song.title)
        assertEquals(5_500L, m.msOf((end + 6 * R).toDouble()))
    }

    @Test fun seekWithinTheSongIsABreak() {
        val m = SongMap()
        val end = play(m, songA, 0, 60_000, 20)
        // seeked back to 10 s
        val seen = m.observe(Anchor(end + R, 10_000, songA))
        assertNotNull(seen)
        play(m, songA, end + R, 10_000, 20)
        assertNull(m.forFrames(5.0 * R, (end + 5 * R).toDouble()))
        // song time 15 s is only in the new stretch
        val a = m.forSong(songA, 12_000, 18_000)!!
        assertEquals(end + 6 * R, a.frameOf(15_000))
        // 70 s is only in the old one
        assertEquals(10 * R, m.forSong(songA, 65_000, 75_000)!!.frameOf(70_000))
        // a range across the seek is in neither
        assertNull(m.forSong(songA, 5_000, 25_000))
    }

    @Test fun aSeekReachesBackToItsTarget() {
        val m = SongMap()
        val end = play(m, songA, 0, 100_000, 20)
        // we seek to 40 s; the first anchor after it comes 3 s of song later (a hop)
        m.cut("seek", 40_000)
        val f = end + 4 * R
        val seen = m.observe(Anchor(f, 43_000, songA))!!
        assertEquals(end, seen.gapFrom)
        assertEquals(f - 3 * R, seen.gapTo)
        // the run-up from 40 s is mapped, though no anchor came before 43 s
        assertEquals(f - 2 * R, m.forSong(songA, 41_000, 43_000)!!.frameOf(41_000))
        assertNull(m.forSong(songA, 39_000, 43_000))
    }

    @Test fun aSeekTargetNeverReachesIntoTheStretchBefore() {
        val m = SongMap()
        val end = play(m, songA, 0, 100_000, 20)
        m.cut("seek", 10_000)
        // anchor says 30 s: 20 s back would be before the last anchor of the old stretch
        m.observe(Anchor(end + 5 * R, 30_000, songA))
        assertEquals(100_000L + 20_000L, m.msOf(end.toDouble()))
        assertEquals(26_000L, m.msOf((end + R).toDouble()))
    }

    @Test fun smallJitterIsNoBreakButAPauseIs() {
        val m = SongMap()
        m.observe(Anchor(0, 1_000, songA))
        assertNull(m.observe(Anchor(R, 2_400, songA))) // +400 ms off: within tolerance
        // paused 5 s: 5 s of silence went into the buffer, the song didn't move
        assertNotNull(m.observe(Anchor(7 * R, 3_400, songA)))
    }

    @Test fun framesGoingBackGiveAnOrderedGap() {
        val m = SongMap()
        play(m, songA, 0, 0, 10)
        // separating -> capture only: the anchor frame drops by the separator's delay
        val seen = m.observe(Anchor(9 * R, 10_500, songA))!!
        assertEquals(9 * R, seen.gapFrom)
        assertEquals(10 * R, seen.gapTo)
    }

    @Test fun aCutBreaksEvenWhenTheTimesAgree() {
        val m = SongMap()
        play(m, songA, 0, 0, 10)
        m.cut("seek")
        assertNotNull(m.observe(Anchor(11 * R, 11_000, songA)))
        assertNull(m.observe(Anchor(12 * R, 12_000, songA)))
    }

    @Test fun forSongOnlyCountsStretchesBegunAfter() {
        val m = SongMap()
        play(m, songA, 0, 0, 30, at0 = 0)
        assertNotNull(m.forSong(songA, 10_000, 20_000))
        assertNull(m.forSong(songA, 10_000, 20_000, after = 50))
        m.cut("seek")
        play(m, songA, 40 * R, 5_000, 20, at0 = 100)
        // the newest stretch has 10-20 s too, and it's the one that began after
        val a = m.forSong(songA, 10_000, 20_000, after = 100)!!
        assertEquals(45 * R, a.frameOf(10_000))
    }

    @Test fun forSongWontReachPastWhatIsBuffered() {
        val m = SongMap()
        play(m, songA, 0, 0, 10)
        assertNull(m.forSong(songA, 5_000, 12_000))
    }

    @Test fun pruneForgetsOldStretches() {
        val m = SongMap()
        play(m, songA, 0, 0, 10)
        m.cut("x")
        play(m, songB, 20 * R, 0, 10)
        m.prune(15 * R)
        assertNull(m.msOf(5.0 * R))
        assertNotNull(m.msOf(25.0 * R))
    }

    @Test fun thinningKeepsTheMapping() {
        val m = SongMap()
        // 0.1 s anchors (capture only) for 100 s
        for (i in 0..1000) m.observe(Anchor(i * R / 10, 50_000 + i * 100L, songA))
        assertEquals(60_000L, m.msOf(10.0 * R))
        assertEquals(149_000L, m.msOf(99.0 * R))
    }

    @Test fun anchorRoundTrips() {
        val a = Anchor(1_000_000, 90_000, songA)
        assertEquals(90_000L, a.msOf(a.frameOf(90_000).toDouble()))
        assertEquals(a.frame - 5 * R, a.frameOf(85_000))
    }
}
