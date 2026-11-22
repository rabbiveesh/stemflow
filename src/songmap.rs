//! Where the song is, along the loop buffer (the Android app's SongMap.kt): a run of
//! segments, each a stretch of buffer frames that is one unbroken play of one song.
//! A seek, a pause, a skip, the next track: each ends a segment. An anchor maps
//! frames only inside its own segment; mapping across a break is how loops and
//! bookmarks came out at the wrong time.
//!
//! A segment runs from its first anchor to its last (the newest one runs on). What
//! lies between two segments isn't known: the break happened somewhere in there.

use std::time::Instant;

pub const RATE: i64 = 44_100;
/// How far the song time may drift from the frames before it is a break.
const TOLERANCE_MS: i64 = 600;

/// A recording, as the player names it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Song {
    /// The player's id for it (`spotify:track:…`), or "".
    pub id: String,
    pub title: String,
    pub artist: String,
    pub album: String,
}

impl Song {
    /// Same recording: the id when both have one, else the names.
    pub fn same(&self, o: &Song) -> bool {
        if !self.id.is_empty() && !o.id.is_empty() {
            self.id == o.id
        } else {
            self.title == o.title && self.artist == o.artist && self.album == o.album
        }
    }
}

/// Buffer frame `frame` is song position `ms` of `song`, as the player said when that
/// frame's input arrived (`at`).
#[derive(Clone, Debug)]
pub struct Anchor {
    pub frame: i64,
    pub ms: i64,
    pub song: Song,
    pub at: Instant,
}

impl Anchor {
    pub fn new(frame: i64, ms: i64, song: Song) -> Self {
        Anchor {
            frame,
            ms,
            song,
            at: Instant::now(),
        }
    }
    pub fn frame_of(&self, ms: i64) -> i64 {
        self.frame + (ms - self.ms) * RATE / 1000
    }
    pub fn ms_of(&self, f: f64) -> i64 {
        self.ms + ((f - self.frame as f64) * 1000.0 / RATE as f64) as i64
    }
}

/// `floor_ms`: when known, the song time this stretch began at (we seeked there, or
/// the next track began at 0), so it reaches back before its first anchor to there,
/// but never past `floor_frame` (the previous stretch's last anchor).
struct Segment {
    anchors: Vec<Anchor>,
    floor_ms: Option<i64>,
    floor_frame: i64,
}

impl Segment {
    fn first(&self) -> &Anchor {
        &self.anchors[0]
    }
    fn last(&self) -> &Anchor {
        self.anchors.last().unwrap()
    }
    fn start_frame(&self) -> i64 {
        match self.floor_ms {
            Some(ms) => self
                .first()
                .frame_of(ms)
                .max(self.floor_frame + 1)
                .min(self.first().frame),
            None => self.first().frame,
        }
    }
    fn start_ms(&self) -> i64 {
        self.first().ms_of(self.start_frame() as f64)
    }
}

/// What `observe` decided: frames [gap_from, gap_to) to mark as a gap, and why.
#[derive(Debug)]
pub struct Seen {
    pub gap_from: i64,
    pub gap_to: i64,
    pub why: String,
}

#[derive(Default)]
pub struct SongMap {
    segments: Vec<Segment>,
    /// Set by `cut`: the next anchor starts a segment whatever it says.
    cut_why: Option<String>,
    cut_floor_ms: Option<i64>,
}

impl SongMap {
    /// The newest anchor: what is playing, and where.
    pub fn latest(&self) -> Option<&Anchor> {
        self.segments.last().map(|s| s.last())
    }

    /// A break we know of (we seeked, frames were skipped, the source paused);
    /// `to_ms` where the song carries on from, when known (a seek's target).
    pub fn cut(&mut self, why: &str, to_ms: Option<i64>) {
        if !self.segments.is_empty() {
            self.cut_why = Some(why.to_string());
            self.cut_floor_ms = to_ms;
        }
    }

    /// The player says buffer frame `a.frame` is at `a.ms`. None if that carries on
    /// from the last anchor; else the frames to mark as a gap.
    pub fn observe(&mut self, a: Anchor) -> Option<Seen> {
        let Some(seg) = self.segments.last_mut() else {
            self.segments.push(Segment {
                anchors: vec![a],
                floor_ms: None,
                floor_frame: i64::MIN,
            });
            return None;
        };
        let p = seg.last().clone();
        let why = if let Some(w) = &self.cut_why {
            Some(w.clone())
        } else if !p.song.same(&a.song) {
            Some(format!("next song ({})", a.song.title))
        } else if a.frame < p.frame {
            Some(format!("frames went back {}", p.frame - a.frame))
        } else {
            let expected = p.ms + (a.frame - p.frame) * 1000 / RATE;
            let off = a.ms - expected;
            (off.abs() > TOLERANCE_MS)
                .then(|| format!("song at {} ms, expected {expected} ({off:+} ms)", a.ms))
        };
        let floor = if self.cut_why.is_some() {
            // only if the player has caught up with the seek
            self.cut_floor_ms
                .filter(|&f| f >= 0 && (0..=10_000).contains(&(a.ms - f)))
        } else if !p.song.same(&a.song) && (0..=15_000).contains(&a.ms) {
            Some(0) // the next track, caught near its start: it began at 0
        } else {
            None
        };
        self.cut_why = None;
        self.cut_floor_ms = None;
        let Some(why) = why else {
            seg.anchors.push(a);
            if seg.anchors.len() > 400 {
                thin(seg);
            }
            return None;
        };
        let seg2 = Segment {
            anchors: vec![a.clone()],
            floor_ms: floor,
            floor_frame: p.frame,
        };
        let seen = Seen {
            gap_from: p.frame.min(a.frame),
            gap_to: p.frame.max(seg2.start_frame()),
            why,
        };
        self.segments.push(seg2);
        // a song or two's worth of stretches is plenty
        if self.segments.len() > 200 {
            self.segments.remove(0);
        }
        Some(seen)
    }

    fn contains(&self, i: usize, f: f64) -> bool {
        let s = &self.segments[i];
        f >= s.start_frame() as f64 && (i == self.segments.len() - 1 || f <= s.last().frame as f64)
    }

    /// An anchor that maps all of frames [from, to], or None if a break lies between.
    pub fn for_frames(&self, from: f64, to: f64) -> Option<Anchor> {
        let mid = (from + to) / 2.0;
        (0..self.segments.len())
            .rev()
            .find(|&i| self.contains(i, from) && self.contains(i, to))
            .map(|i| {
                self.segments[i]
                    .anchors
                    .iter()
                    .min_by(|x, y| {
                        (x.frame as f64 - mid)
                            .abs()
                            .total_cmp(&(y.frame as f64 - mid).abs())
                    })
                    .unwrap()
                    .clone()
            })
    }

    /// Song ms of frame `f`, or None outside every segment.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn ms_of(&self, f: f64) -> Option<i64> {
        self.for_frames(f, f).map(|a| a.ms_of(f))
    }

    /// An anchor that maps song ms [from, to] of `song` to frames, all inside one
    /// segment (the newest that has them). The newest segment reaches only up to its
    /// last anchor: what plays after that isn't in the buffer yet. With `after`,
    /// only segments that began then or later count.
    pub fn for_song(
        &self,
        song: &Song,
        from: i64,
        to: i64,
        after: Option<Instant>,
    ) -> Option<Anchor> {
        for s in self.segments.iter().rev() {
            if after.is_some_and(|t| s.first().at < t) {
                break; // older segments began even earlier
            }
            if !s.first().song.same(song) {
                continue;
            }
            if from >= s.start_ms() && to <= s.last().ms {
                let mid = (from + to) / 2;
                return s.anchors.iter().min_by_key(|a| (a.ms - mid).abs()).cloned();
            }
        }
        None
    }

    /// For the log: each segment's song, frames and song times.
    pub fn describe(&self) -> String {
        self.segments
            .iter()
            .map(|s| {
                format!(
                    "{}: frames {}–{} = {}–{} ms ({} anchors)",
                    s.first().song.title.chars().take(24).collect::<String>(),
                    s.start_frame(),
                    s.last().frame,
                    s.start_ms(),
                    s.last().ms,
                    s.anchors.len()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// Keep it small: the newest anchors and one per few seconds before them.
fn thin(seg: &mut Segment) {
    let n = seg.anchors.len();
    let mut keep = Vec::new();
    let mut last: Option<i64> = None;
    for (i, x) in seg.anchors.drain(..).enumerate() {
        if last.is_none() || i + 50 >= n || x.frame - last.unwrap() >= 3 * RATE {
            last = Some(x.frame);
            keep.push(x);
        }
    }
    seg.anchors = keep;
}

/// `m:ss`
pub fn time(ms: i64) -> String {
    format!("{}:{:02}", ms / 60_000, ms / 1000 % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn song(t: &str) -> Song {
        Song {
            id: format!("id:{t}"),
            title: t.into(),
            ..Default::default()
        }
    }

    #[test]
    fn stretches() {
        let mut m = SongMap::default();
        let a = song("a");
        // a second of song per second of frames
        for k in 0..10 {
            assert!(m
                .observe(Anchor::new(k * RATE, 30_000 + k * 1000, a.clone()))
                .is_none());
        }
        let x = m.for_frames(2.0 * RATE as f64, 8.0 * RATE as f64).unwrap();
        assert_eq!(x.ms_of(5.0 * RATE as f64), 35_000);
        // a seek to 1:00: a new stretch, beginning at the seek
        m.cut("seek", Some(60_000));
        let seen = m
            .observe(Anchor::new(11 * RATE, 60_500, a.clone()))
            .unwrap();
        assert_eq!(seen.why, "seek");
        for k in 12..20 {
            assert!(m
                .observe(Anchor::new(k * RATE, 60_500 + (k - 11) * 1000, a.clone()))
                .is_none());
        }
        // nothing maps across the seek
        assert!(m
            .for_frames(8.0 * RATE as f64, 12.0 * RATE as f64)
            .is_none());
        // the new stretch reaches back to the seek target
        assert_eq!(m.ms_of(10.6 * RATE as f64).unwrap() / 100, 601);
        // by song time: in the new stretch only up to its last anchor
        assert!(m.for_song(&a, 61_000, 68_000, None).is_some());
        assert!(m.for_song(&a, 61_000, 70_000, None).is_none());
        // a jump in song time without a cut is a break too
        let seen = m
            .observe(Anchor::new(21 * RATE, 10_000, a.clone()))
            .unwrap();
        assert!(seen.why.starts_with("song at"), "{}", seen.why);
        // the next song, near its start, begins at 0
        m.observe(Anchor::new(30 * RATE, 2_000, song("b")));
        assert_eq!(m.ms_of(28.5 * RATE as f64).unwrap() / 100, 5);
        assert!(m.latest().unwrap().song.same(&song("b")));
    }
}
