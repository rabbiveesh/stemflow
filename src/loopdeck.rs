//! Looping while live: what a player around the looper (src/looper.rs) has to do,
//! kept apart from any platform so the desktop and the Android app can share it.
//!
//! The player pushes every block of stems into the looper as it arrives, tells the
//! deck which buffer frame is being heard (`set_heard`), and plays `Looper::read`
//! instead of the live stream whenever that returns a block. The deck turns requests
//! ("loop the last 4 bars", "live", "move the end a beat on") into looper calls, runs
//! the slow ones (finding the bars) on a thread of their own, and pauses the source
//! app while a loop plays, so that going live again carries on from the loop
//! instead of from wherever the song got to meanwhile.

//!
//! With the player's song position (`Source::now`) it also does what needs the song
//! rather than the buffer, as the Android app does (CaptureService.kt): a map from
//! buffer frames to song time (src/songmap.rs); letting the source play on, unheard,
//! until the next phrase is buffered, then pausing it; extending the loop past what
//! was buffered by seeking the source there and playing it through; repairing a
//! loop by playing it again; and bookmarks (src/bookmarks.rs) recalled by seeking.

use crate::beats::Settings;
use crate::bookmarks::{self, Bookmark};
use crate::looper::Looper;
use crate::songmap::{time, Anchor, Song, SongMap, RATE as SRATE};
use anyhow::{bail, Context, Result};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the player says it is playing.
#[derive(Clone, Debug)]
pub struct Now {
    pub song: Song,
    /// Position in it.
    pub ms: i64,
    pub playing: bool,
}

/// The app whose audio is looped, as far as controlling it goes.
pub trait Source: Send + Sync {
    /// Pause it; false if that isn't possible (no such player, no permission).
    fn pause(&self) -> bool;
    /// Play it again.
    fn play(&self) -> bool;
    /// The song and the position in it, if the player says.
    fn now(&self) -> Option<Now> {
        None
    }
    /// Go to `ms` in the song playing.
    fn seek(&self, _ms: i64) -> bool {
        false
    }
    /// Play the recording with this id (a bookmark's).
    fn open(&self, _id: &str) -> bool {
        false
    }
}

/// A source that can't be paused: the loop plays over the song carrying on unheard,
/// and "live" goes straight back to the live stream.
#[allow(dead_code)] // for players without a remote control (and the tests)
pub struct Unpausable;

impl Source for Unpausable {
    fn pause(&self) -> bool {
        false
    }
    fn play(&self) -> bool {
        false
    }
}

pub struct Deck {
    pub looper: Looper,
    source: Box<dyn Source>,
    /// Buffer frame being heard now, -1 until something is.
    heard: AtomicI64,
    /// We paused the source for the loop playing now.
    paused: AtomicBool,
    /// "" or why the last request didn't work.
    status: Mutex<String>,
    /// Bumped by each request, so a slow find that's been overtaken is dropped.
    job: AtomicU64,
    click: AtomicBool,
    /// speed, transposition (semitones), keep pitch when slowed
    speed: Mutex<(f32, f32, bool)>,
    /// Called after anything a UI shows changed.
    notify: Box<dyn Fn() + Send + Sync>,
    /// Frames between the looper's read position and the speakers, so a drawn
    /// playhead follows the ears.
    lag: AtomicI64,
    /// Buffer frame of the newest input (what the player is at now), -1 unknown.
    input: AtomicI64,
    /// Buffer frames to song time.
    songs: Mutex<SongMap>,
    /// What the player said last.
    now: Mutex<Option<Now>>,
    /// The buffer holds the song unbroken since our last seek (`stream_since`; None:
    /// since ever) from `stream_from_ms`.
    stream_since: Mutex<Option<Instant>>,
    stream_from_ms: AtomicI64,
    /// Bumped to call off whatever phrase work (prefetch, extend, repair) is running.
    phrase: AtomicU64,
    /// Bumped to call off a recall; `recalling` while one is on its way.
    recall_id: AtomicU64,
    recalling: AtomicBool,
    /// Play the plain mix while live, and separate only what is looped.
    stems_only: AtomicBool,
}

/// One bookmark-or-loop-sized error, for the status line.
fn msg(e: anyhow::Error) -> String {
    format!("{e:#}")
}

/// What a UI shows of the loop.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct State {
    /// A loop is playing.
    pub playing: bool,
    /// The bars are being found.
    pub finding: bool,
    /// After a loop: the buffer plays on from it until it reaches the live stream.
    pub tail: bool,
    pub bpm: f32,
    pub meter: usize,
    pub bars: usize,
    /// The loop's ends, as beat indices, and its length in beats (`b - a`).
    pub a: usize,
    pub b: usize,
    pub beats: usize,
    /// Loop length, seconds (at speed 1).
    pub length: f32,
    /// Playhead, 0..1 of the loop.
    pub pos: f32,
    pub speed: f32,
    pub semis: f32,
    pub keep_pitch: bool,
    pub click: bool,
    pub status: String,
    /// What the player is playing, and where (ms), if it says.
    pub title: String,
    pub artist: String,
    pub song_ms: i64,
    pub song_playing: bool,
    /// The loop's place in the song (ms), when the song map holds it.
    pub loop_from: i64,
    pub loop_to: i64,
    /// A bookmark is on its way back.
    pub recalling: bool,
    pub stems_only: bool,
}

const RATE: f64 = 44_100.0;

impl Deck {
    pub fn new(
        looper: Looper,
        source: Box<dyn Source>,
        notify: Box<dyn Fn() + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(Deck {
            looper,
            source,
            heard: AtomicI64::new(-1),
            paused: AtomicBool::new(false),
            status: Mutex::new(String::new()),
            job: AtomicU64::new(0),
            click: AtomicBool::new(false),
            speed: Mutex::new((1.0, 0.0, true)),
            notify,
            lag: AtomicI64::new(0),
            input: AtomicI64::new(-1),
            songs: Mutex::new(SongMap::default()),
            now: Mutex::new(None),
            stream_since: Mutex::new(None),
            stream_from_ms: AtomicI64::new(0),
            phrase: AtomicU64::new(0),
            recall_id: AtomicU64::new(0),
            recalling: AtomicBool::new(false),
            stems_only: AtomicBool::new(load_stems_only()),
        })
    }

    /// The worker: buffer frame `frame` is the newest input, what plays now.
    pub fn set_input(&self, frame: i64) {
        self.input.store(frame, Ordering::Relaxed);
    }

    /// The buffer jumps here (frames skipped or dropped): the song map breaks.
    pub fn cut(&self, why: &str) {
        self.songs.lock().unwrap().cut(why, None);
    }

    pub fn stems_only(&self) -> bool {
        self.stems_only.load(Ordering::Relaxed)
    }

    /// Ask the player where it is, every `every`, and map the buffer to song time.
    pub fn watch(self: &Arc<Self>, every: Duration) {
        let me = self.clone();
        std::thread::spawn(move || {
            let mut shown: Option<(String, bool)> = None;
            let mut was_playing = false;
            loop {
                std::thread::sleep(every);
                let n = me.source.now();
                let f = me.input.load(Ordering::Relaxed);
                match &n {
                    Some(n) if n.playing && f >= 0 => {
                        let a = Anchor::new(f, n.ms, n.song.clone());
                        let seen = me.songs.lock().unwrap().observe(a);
                        if let Some(s) = seen {
                            eprintln!("songmap: break at frame {}: {}", s.gap_to, s.why);
                            if s.gap_to > s.gap_from {
                                me.looper.mark_gap(s.gap_from, s.gap_to);
                            }
                        }
                    }
                    Some(n) if !n.playing && was_playing => {
                        me.songs.lock().unwrap().cut("paused", None);
                    }
                    _ => {}
                }
                was_playing = n.as_ref().is_some_and(|n| n.playing);
                let key = n.as_ref().map(|n| (n.song.title.clone(), n.playing));
                *me.now.lock().unwrap() = n;
                if key != shown {
                    shown = key;
                    (me.notify)();
                }
            }
        });
    }

    /// Seek the source to `ms` and play: the buffer holds the song unbroken from
    /// there on.
    fn seek(&self, ms: i64, why: &str) {
        let to = ms.max(0);
        eprintln!("loop: seek to {} ({why})", time(to));
        self.songs
            .lock()
            .unwrap()
            .cut(&format!("seek to {to} ms ({why})"), Some(to));
        *self.stream_since.lock().unwrap() = Some(Instant::now());
        self.stream_from_ms.store(to, Ordering::Relaxed);
        self.source.seek(to);
        self.source.play();
    }

    fn pause_source(&self) {
        if self.source.pause() {
            self.paused.store(true, Ordering::Relaxed);
        }
    }

    /// A loop has just started: the click and speed it should have.
    fn started(&self) {
        self.looper.set_click(self.click.load(Ordering::Relaxed));
        let (sp, semis, keep) = *self.speed.lock().unwrap();
        self.looper.set_speed(sp, semis, keep);
    }

    /// The anchor mapping song ms [from, to] into the buffer, if it's all there.
    fn covering(&self, song: &Song, from: i64, to: i64) -> Option<Anchor> {
        if from < self.stream_from_ms.load(Ordering::Relaxed) {
            return None;
        }
        let since = *self.stream_since.lock().unwrap();
        let lo = (from + 6_000).min(to);
        let x = self.songs.lock().unwrap().for_song(song, lo, to, since)?;
        let total = self.looper.total() as i64;
        // inside the newest minute of music, which is what gets analysed
        let music = self.looper.sound_end() as i64;
        if total < x.frame_of(to) || x.frame_of(from) < music.max(x.frame_of(to)) - 55 * SRATE {
            return None;
        }
        // what's wanted (past the run-up) must be one unbroken stretch of the song
        self.looper
            .continuous(x.frame_of(lo), x.frame_of(to))
            .then_some(x)
    }

    /// Have the buffer hold song ms [from, to]: wait if the player is already playing
    /// through it, else seek there and play. None if called off (`phrase` moved on,
    /// or the loop stopped).
    fn stream(
        &self,
        id: u64,
        song: &Song,
        from: i64,
        to: i64,
        open: bool,
    ) -> Result<Option<Anchor>> {
        let from = from.max(0);
        if let Some(x) = self.covering(song, from, to) {
            return Ok(Some(x));
        }
        let off = || self.phrase.load(Ordering::Relaxed) != id || !self.looper.playing();
        let mut cur = self.source.now();
        // never seek in another song (the queue may have moved on past this one)
        if !cur.as_ref().is_some_and(|n| n.song.same(song)) {
            if !open || !self.source.open(&song.id) {
                bail!("The player has moved on from {}", song.title);
            }
            let until = Instant::now() + Duration::from_secs(30);
            loop {
                if off() {
                    return Ok(None);
                }
                cur = self.source.now();
                if cur.as_ref().is_some_and(|n| n.song.same(song)) {
                    break;
                }
                if Instant::now() > until {
                    bail!("The player didn't open {}", song.title);
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        let on_the_way = {
            let since = *self.stream_since.lock().unwrap();
            let m = self.songs.lock().unwrap();
            cur.as_ref().is_some_and(|n| n.playing)
                && m.latest().is_some_and(|x| {
                    let lo = (from + 6_000).min(x.ms);
                    x.song.same(song)
                        && from >= self.stream_from_ms.load(Ordering::Relaxed)
                        && x.ms <= to
                        && m.for_song(song, lo, x.ms, since).is_some()
                        && self.looper.continuous(x.frame_of(lo), x.frame)
                })
        };
        if !on_the_way {
            self.seek(from, &format!("to hear {}–{}", time(from), time(to)));
        }
        let until = Instant::now() + Duration::from_millis(60_000 + (to - from).max(0) as u64);
        loop {
            if off() {
                return Ok(None);
            }
            if Instant::now() > until {
                eprintln!(
                    "loop: timed out; map: {}",
                    self.songs.lock().unwrap().describe()
                );
                bail!("Timed out waiting for the music");
            }
            if let Some(x) = self.covering(song, from, to) {
                return Ok(Some(x));
            }
            let since = *self.stream_since.lock().unwrap();
            if let Some(x) = self.songs.lock().unwrap().latest() {
                if since.is_some_and(|t| x.at >= t) && !x.song.same(song) {
                    bail!("The song ended before that part");
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// The loop in song ms (start, end) and the anchor that maps it, if one stretch
    /// of the song holds it.
    fn loop_in_song(&self) -> Option<(Anchor, i64, i64)> {
        let t = self.looper.times()?;
        let a = self.songs.lock().unwrap().for_frames(t.0.min(t.2), t.1)?;
        let (s, e) = (a.ms_of(t.0), a.ms_of(t.1));
        Some((a, s, e))
    }

    /// With a loop set: let the source play on (unheard: the loop is what plays)
    /// until the buffer has the next phrase, then pause it, so moving into it is
    /// instant. Without song positions, pause it straight away.
    fn pause_after_phrase(self: &Arc<Self>) {
        let id = self.phrase.fetch_add(1, Ordering::Relaxed) + 1;
        let Some((a0, s, e)) = self.loop_in_song() else {
            return self.pause_source();
        };
        let (r0, r1) = (e, e + (e - s));
        let me = self.clone();
        std::thread::spawn(move || {
            match me.stream(id, &a0.song, r0 - 6_000, r1 + 1_000, false) {
                Ok(Some(a)) => {
                    eprintln!("loop: next phrase buffered");
                    if me.phrase.load(Ordering::Relaxed) == id && me.looper.playing() {
                        me.pause_source();
                        // its beats too, so moving into it needs no analysis; only
                        // when the buffered run reaches back into the loop
                        let t = me.looper.times();
                        if t.is_some_and(|t| a.frame_of(r0 - 6_000) as f64 <= t.1) {
                            if let Err(e) = me.looper.refresh() {
                                eprintln!("loop: beats ahead: {e:#}");
                            }
                        }
                    }
                    return;
                }
                Ok(None) => {}
                Err(e) => eprintln!("loop: next phrase: {e:#}"),
            }
            if me.phrase.load(Ordering::Relaxed) == id && me.looper.playing() {
                me.pause_source();
            }
        });
    }

    /// Move the loop's start by `da` beats and its end by `db` (extend, trim, or move
    /// it). In the buffer at once; a part not buffered yet is played first (the old
    /// loop keeps playing meanwhile) and placed by song time, keeping bar 1.
    fn adjust(self: &Arc<Self>, da: i64, db: i64) {
        if !self.looper.playing() || (da == 0 && db == 0) {
            return;
        }
        if self.looper.nudge(da, db) {
            self.say("");
            if db > 0 {
                self.pause_after_phrase();
            }
            return;
        }
        if da >= 0 && db <= 0 {
            return self.say("The loop can't get shorter than that");
        }
        // What was buffered after (or before) the loop may hold it: find the beats
        // there (keeping the loop as it is) and try again. Only if that isn't enough
        // is the song played again.
        let id = self.phrase.fetch_add(1, Ordering::Relaxed) + 1;
        let me = self.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            match me.looper.refresh() {
                Ok(true) => eprintln!(
                    "loop: beats around found in {:.2}s",
                    t.elapsed().as_secs_f32()
                ),
                Ok(false) => {}
                Err(e) => eprintln!("loop: beats around: {e:#}"),
            }
            if me.phrase.load(Ordering::Relaxed) != id {
                return;
            }
            if me.looper.nudge(da, db) {
                me.say("");
                if db > 0 {
                    me.pause_after_phrase();
                }
            } else {
                me.stream_to(da, db, false);
            }
        });
    }

    /// Play the loop's audio again from the player (back to before it and through
    /// it), and lay the loop on that: for a loop with a glitch in it.
    fn repair(self: &Arc<Self>) {
        if self.looper.playing() {
            self.stream_to(0, 0, true);
        }
    }

    fn stream_to(self: &Arc<Self>, da: i64, db: i64, fresh: bool) {
        let id = self.phrase.fetch_add(1, Ordering::Relaxed) + 1;
        let me = self.clone();
        std::thread::spawn(move || {
            let r = me.stream_to_now(id, da, db, fresh);
            if me.phrase.load(Ordering::Relaxed) == id {
                me.say(r.err().map(msg).unwrap_or_default());
            }
        });
    }

    fn stream_to_now(self: &Arc<Self>, id: u64, da: i64, db: i64, fresh: bool) -> Result<()> {
        let (t0, t1, bar1f, meter, bars, feel) = self.looper.times().context("No loop")?;
        let v = self.looper.view(1).context("No loop")?;
        let a0 = self
            .songs
            .lock()
            .unwrap()
            .for_frames(t0.min(bar1f), t1)
            .with_context(|| {
                if self.source.now().is_none() {
                    "The player doesn't say where in the song it is, so the loop can't go past what was heard"
                } else {
                    "Can't tell where in the song this loop is (a seek, pause or song change runs through it)"
                }
            })?;
        let (start, end) = (a0.ms_of(t0), a0.ms_of(t1));
        let beat_ms = (end - start) as f64 / (v.b - v.a).max(1) as f64;
        let new_start = start + (da as f64 * beat_ms) as i64;
        let new_end = end + (db as f64 * beat_ms) as i64;
        // (the first beat may be a moment before 0: the song's own timing)
        if new_start < -(beat_ms as i64) / 2 {
            bail!("The song starts there: the loop can't start any earlier");
        }
        // bar 1: where it was, moved with the start when that moves by whole bars
        let m = meter.max(1) as i64;
        let bar1 = if da % m != 0 {
            a0.ms_of(bar1f)
        } else if bar1f == t0 {
            new_start
        } else {
            a0.ms_of(bar1f) + (da as f64 * beat_ms) as i64
        };
        self.say(format!(
            "Listening to {}–{}…",
            time(new_start),
            time(new_end)
        ));
        if fresh {
            self.stream_from_ms.store(i64::MAX, Ordering::Relaxed); // nothing buffered counts
        }
        let Some(a) = self.stream(
            id,
            &a0.song,
            new_start.min(bar1) - 6_000,
            new_end + 1_000,
            fresh,
        )?
        else {
            return Ok(());
        };
        self.say("Finding the bars…");
        let hint = v.bpm / feel;
        let s = Settings {
            meter,
            feel,
            min_bpm: hint * 0.85,
            max_bpm: hint * 1.18,
        };
        self.looper
            .find(a.frame_of(new_end).max(0) as u64, bars, s)?;
        self.looper.set_times(
            a.frame_of(new_start) as f64,
            a.frame_of(new_end) as f64,
            a.frame_of(bar1) as f64,
        );
        // the length in beats is what was asked for, counted from the start, so a
        // slightly different tempo can't add or drop a beat
        let beats = (v.b - v.a) as i64 + db - da;
        if let Some(w) = self.looper.view(1) {
            if beats > 0 && (w.b - w.a) as i64 != beats {
                self.looper.set_edges(w.a, w.a + beats as usize);
            }
        }
        self.started();
        eprintln!("loop: now {}–{}", time(new_start), time(new_end));
        self.pause_after_phrase();
        Ok(())
    }

    /// Save the loop as a bookmark called `name`.
    fn save(&self, name: &str) -> Result<()> {
        let t = self.looper.times().context("No loop to save")?;
        let a = self
            .songs
            .lock()
            .unwrap()
            .for_frames(t.0.min(t.2), t.1)
            .with_context(|| {
                if self.source.now().is_none() {
                    "Not saved: the player doesn't say what's playing or where"
                } else {
                    "Not saved: the song was seeked, paused or changed during this loop, so its time in the song isn't known. Press Live, then loop again."
                }
            })?;
        let start = a.ms_of(t.0);
        if start < 0 {
            bail!("Not saved: this loop seems to start before the song does");
        }
        let mut b = Bookmark::new(
            name,
            a.song.clone(),
            start,
            a.ms_of(t.1),
            a.ms_of(t.2),
            t.3,
            t.4,
            t.5,
        );
        if let Some(g) = self.grid() {
            b = b.with_grid(g);
        }
        bookmarks::add(&b)?;
        eprintln!(
            "loop: saved {name}: {} {}–{}",
            b.song.title,
            time(b.start_ms),
            time(b.end_ms)
        );
        Ok(())
    }

    /// The looper's beats around the loop, in song ms: only those in the loop's own
    /// stretch of the song.
    fn grid(&self) -> Option<bookmarks::Grid> {
        let (f, a0, b0, one, _, bpm) = self.looper.grid_around(2)?;
        let m = self.songs.lock().unwrap();
        let a = m.for_frames(f[a0], f[b0])?;
        let mut lo = a0;
        while lo > 0 && m.for_frames(f[lo - 1], f[b0]).is_some() {
            lo -= 1;
        }
        let mut hi = b0;
        while hi + 1 < f.len() && m.for_frames(f[a0], f[hi + 1]).is_some() {
            hi += 1;
        }
        let beats = f[lo..=hi].iter().map(|&x| a.ms_of(x) as f64).collect();
        Some(bookmarks::Grid {
            beats,
            a: a0 - lo,
            b: b0 - lo,
            one: one.saturating_sub(lo),
            bpm,
        })
    }

    /// Play a bookmark: the player to that recording and to a run-up before the
    /// loop, then, once its stems are in the buffer, lay the loop where it was saved.
    fn recall(self: &Arc<Self>, created: i64) -> Result<()> {
        let b = bookmarks::all()
            .into_iter()
            .find(|b| b.created == created)
            .context("No such bookmark")?;
        let id = self.recall_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.recalling.store(true, Ordering::Relaxed);
        let me = self.clone();
        std::thread::spawn(move || {
            let r = me.recall_now(&b, id);
            if me.recall_id.load(Ordering::Relaxed) == id {
                me.recalling.store(false, Ordering::Relaxed);
                me.say(match r {
                    Ok(()) => String::new(),
                    Err(e) if e.downcast_ref::<Cancelled>().is_some() => String::new(),
                    Err(e) => msg(e),
                });
            }
        });
        Ok(())
    }

    fn recall_now(self: &Arc<Self>, b: &Bookmark, id: u64) -> Result<()> {
        let check = || -> Result<()> {
            if self.recall_id.load(Ordering::Relaxed) != id {
                return Err(Cancelled.into());
            }
            Ok(())
        };
        self.say(format!(
            "Going to {} at {}…",
            b.song.title,
            time(b.start_ms)
        ));
        self.phrase.fetch_add(1, Ordering::Relaxed);
        self.looper.stop(); // straight to live: we listen to the run-up
        self.paused.store(false, Ordering::Relaxed);
        if !self.source.now().is_some_and(|n| n.song.same(&b.song)) {
            if b.song.id.is_empty() || !self.source.open(&b.song.id) {
                bail!("Play {} in the player, then try again", b.song.title);
            }
            self.say(format!("Opening {}…", b.song.title));
            let until = Instant::now() + Duration::from_secs(30);
            while !self.source.now().is_some_and(|n| n.song.same(&b.song)) {
                check()?;
                if Instant::now() > until {
                    bail!("The player didn't open {}", b.song.title);
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        // With its own beats, only the stems are needed: a short run-up for the
        // separator to settle, then up to the last kept beat. Without, enough before
        // the loop for the beat model to find the tempo.
        let own = b.has_grid();
        let (first, last) = if own {
            (b.beats[0] as i64, *b.beats.last().unwrap() as i64)
        } else {
            (b.start_ms, b.end_ms)
        };
        let seek_to = (first - if own { 4_000 } else { 12_000 }).max(0);
        self.seek(seek_to, &format!("recall {}", b.name));
        let since = *self.stream_since.lock().unwrap();
        let lo = first.min(b.start_ms).min(b.bar1_ms);
        let hi = last + if own { 500 } else { 1500 };
        let until = Instant::now() + Duration::from_secs(60);
        let a = loop {
            check()?;
            if Instant::now() > until {
                bail!("Timed out waiting for the music");
            }
            std::thread::sleep(Duration::from_millis(250));
            if let Some(x) = self.songs.lock().unwrap().latest() {
                if since.is_some_and(|t| x.at >= t) && x.song.same(&b.song) {
                    *self.status.lock().unwrap() =
                        format!("Listening up to the loop… {}", time(x.ms));
                }
            }
            (self.notify)();
            if let Some(a) = self.songs.lock().unwrap().for_song(&b.song, lo, hi, since) {
                break a;
            }
        };
        // the stems up to there are in the buffer
        let need = a.frame_of(last + if own { 300 } else { 1000 });
        while (self.looper.total() as i64) < need {
            check()?;
            if Instant::now() > until {
                bail!("Timed out waiting for the music");
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        check()?;
        if own {
            let frames = b
                .beats
                .iter()
                .map(|&ms| a.frame as f64 + (ms - a.ms as f64) * SRATE as f64 / 1000.0)
                .collect();
            self.looper
                .lay(frames, b.a, b.b, b.one, b.meter, b.bpm, b.feel)?;
        } else {
            self.say("Finding the bars…");
            let d = Settings::default();
            let s = Settings {
                meter: b.meter,
                feel: b.feel,
                min_bpm: if b.bpm > 0.0 {
                    b.bpm / b.feel * 0.85
                } else {
                    d.min_bpm
                },
                max_bpm: if b.bpm > 0.0 {
                    b.bpm / b.feel * 1.18
                } else {
                    d.max_bpm
                },
            };
            self.looper
                .find(a.frame_of(b.end_ms).max(0) as u64, b.bars, s)?;
            self.looper.set_times(
                a.frame_of(b.start_ms) as f64,
                a.frame_of(b.end_ms) as f64,
                a.frame_of(b.bar1_ms) as f64,
            );
            // keep the beats now, so next time there's nothing to find
            if let Some(g) = self.grid() {
                let _ = bookmarks::replace(&b.clone().with_grid(g));
            }
        }
        if self.recall_id.load(Ordering::Relaxed) != id {
            self.looper.stop(); // Live was pressed meanwhile
            return Err(Cancelled.into());
        }
        self.started();
        self.pause_after_phrase();
        eprintln!("loop: recalled {}", b.name);
        Ok(())
    }

    /// The player: buffer frame `frame` is what the speakers play now.
    pub fn set_heard(&self, frame: i64) {
        self.heard.store(frame, Ordering::Relaxed);
    }

    /// The player: how many frames the speakers trail what it reads.
    pub fn set_lag(&self, frames: i64) {
        self.lag.store(frames, Ordering::Relaxed);
    }

    fn say(&self, s: impl Into<String>) {
        *self.status.lock().unwrap() = s.into();
        (self.notify)();
    }

    /// Loop the last `bars` bars of what is being heard, or resize the loop playing.
    pub fn loop_last(self: &Arc<Self>, bars: usize, s: Settings) {
        if self.looper.playing() {
            self.looper.set_bars(bars);
            self.say("");
            return;
        }
        self.recall_id.fetch_add(1, Ordering::Relaxed);
        self.recalling.store(false, Ordering::Relaxed);
        // live up to now: unbroken, as far as we know
        *self.stream_since.lock().unwrap() = None;
        self.stream_from_ms.store(0, Ordering::Relaxed);
        let press = self.heard.load(Ordering::Relaxed);
        if press < 0 {
            return self.say("Nothing heard yet");
        }
        let id = self.job.fetch_add(1, Ordering::Relaxed) + 1;
        self.say("Finding the bars…");
        let me = self.clone();
        std::thread::spawn(move || {
            let t = std::time::Instant::now();
            let r = me.looper.find(press as u64, bars, s);
            if me.job.load(Ordering::Relaxed) != id {
                // "live" was asked for meanwhile: don't start the loop after all
                if r.is_ok() {
                    me.looper.stop();
                }
                return (me.notify)();
            }
            match r {
                Ok(()) => {
                    me.started();
                    eprintln!(
                        "loop: {bars} bars found in {:.2}s",
                        t.elapsed().as_secs_f32()
                    );
                    // Nobody hears the song now: pause it (once the next phrase is
                    // buffered), so live picks up from here.
                    me.pause_after_phrase();
                    // fewer bars when the song (or a seek, or a pause) began since
                    let got = me.looper.view(1).map(|v| (v.b - v.a, v.meter.max(1)));
                    me.say(match got {
                        Some((n, m)) if n < bars * m => format!(
                            "Only {} of {bars} bars: the song started (or was seeked or paused) just before",
                            if n % m == 0 { format!("{}", n / m) } else { format!("{:.1}", n as f32 / m as f32) }
                        ),
                        _ => String::new(),
                    });
                }
                Err(e) => {
                    eprintln!("loop: {e:#}");
                    me.say(format!("No loop: {e:#}"));
                }
            }
        });
    }

    /// Back to the live stream. If we paused the source, the music carries on from the
    /// loop (through what was buffered after it) and the source plays again; if not,
    /// it went on unheard, so jump straight to it.
    pub fn live(&self) {
        self.job.fetch_add(1, Ordering::Relaxed);
        self.phrase.fetch_add(1, Ordering::Relaxed);
        self.recall_id.fetch_add(1, Ordering::Relaxed);
        self.recalling.store(false, Ordering::Relaxed);
        // We paused the source, or it is playing on to buffer the next phrase (and
        // would have been paused): carry on from the loop.
        let ours = self.paused.swap(false, Ordering::Relaxed)
            || self.source.now().is_some_and(|n| n.playing);
        if self.looper.playing() && ours {
            self.looper.go_live();
            self.source.play();
        } else {
            self.looper.stop();
        }
        self.say("");
    }

    /// Run a text command (the control socket's and D-Bus's `loop ...`):
    /// `<bars>`, `live`, `again` (or `top`: from the top), `move <da> <db>` (beats),
    /// `shift <beats>`, `edges <a> <b>` (beat indices), `bar1 <beat index>`,
    /// `click on|off`, `meter`/`feel` (re-decode with the settings now set),
    /// `speed <x>`, `pitch <semitones>`, `keep_pitch on|off`, `repair`,
    /// `save <name>`, `recall <created>`, `rename <created> <name>`,
    /// `delete <created>`, `playpause` (the source), `stems_only on|off`.
    pub fn command(self: &Arc<Self>, cmd: &str, s: Settings) -> Result<()> {
        let mut w = cmd.split_whitespace();
        let verb = w.next().context("loop what?")?;
        let mut int = || -> Result<i64> {
            w.next()
                .context("missing value")?
                .parse()
                .context("not a number")
        };
        let on = |v: Option<&str>| matches!(v, Some("1" | "true" | "on"));
        match verb {
            "live" => self.live(),
            "again" | "top" => {
                self.looper.play();
                if self.looper.playing()
                    && !self.paused.load(Ordering::Relaxed)
                    && self.source.pause()
                {
                    self.paused.store(true, Ordering::Relaxed);
                }
                (self.notify)();
            }
            "move" => {
                let (da, db) = (int()?, int()?);
                self.adjust(da, db);
            }
            "repair" => self.repair(),
            "save" => {
                let name = cmd.split_once(' ').map_or("", |x| x.1).trim();
                self.save(if name.is_empty() { "Loop" } else { name })?;
                (self.notify)();
            }
            "recall" => self.recall(int()?)?,
            "rename" => {
                let created = int()?;
                let name = cmd.splitn(3, ' ').nth(2).unwrap_or("").trim();
                if name.is_empty() {
                    bail!("rename to what?");
                }
                bookmarks::rename(created, name)?;
                (self.notify)();
            }
            "delete" => {
                bookmarks::remove(int()?)?;
                (self.notify)();
            }
            "playpause" => {
                if self.source.now().is_some_and(|n| n.playing) {
                    self.source.pause();
                } else {
                    self.source.play();
                }
                (self.notify)();
            }
            "stems_only" => {
                let v = on(cmd.split_whitespace().nth(1));
                self.stems_only.store(v, Ordering::Relaxed);
                save_stems_only(v);
                (self.notify)();
            }
            "shift" => {
                self.looper.shift(int()? as i32);
                (self.notify)();
            }
            "edges" => {
                let (a, b) = (int()?, int()?);
                if a < 0 || b <= a {
                    bail!("the start has to come before the end");
                }
                let before = self.looper.view(1).map(|v| (v.a, v.b));
                self.looper.set_edges(a as usize, b as usize);
                if self.looper.view(1).map(|v| (v.a, v.b)) == before {
                    self.say("The loop can't go there: not heard yet, or across a pause");
                } else {
                    self.say("");
                }
            }
            "bar1" => {
                self.looper.bar_one(int()?.max(0) as usize);
                (self.notify)();
            }
            "click" => {
                let v = on(cmd.split_whitespace().nth(1));
                self.click.store(v, Ordering::Relaxed);
                self.looper.set_click(v);
                (self.notify)();
            }
            "meter" | "feel" => {
                // re-decode the bars with the beat options now set (this can run the
                // beat network, so not on the caller's thread)
                let me = self.clone();
                std::thread::spawn(move || {
                    me.looper.redecode(s.meter, s.feel);
                    (me.notify)();
                });
            }
            "speed" | "pitch" | "keep_pitch" => {
                let v = cmd.split_whitespace().nth(1).context("missing value")?;
                let mut sp = self.speed.lock().unwrap();
                match verb {
                    "speed" => sp.0 = v.parse::<f32>().context("not a number")?.clamp(0.25, 2.0),
                    "pitch" => sp.1 = v.parse::<f32>().context("not a number")?.clamp(-12.0, 12.0),
                    _ => sp.2 = on(Some(v)),
                }
                self.looper.set_speed(sp.0, sp.1, sp.2);
                drop(sp);
                (self.notify)();
            }
            n => match n.parse::<usize>() {
                Ok(bars) if (1..=64).contains(&bars) => self.loop_last(bars, s),
                _ => bail!("unknown loop command `{n}`"),
            },
        }
        Ok(())
    }

    pub fn state(&self) -> State {
        let (speed, semis, keep_pitch) = *self.speed.lock().unwrap();
        let mut s = State {
            playing: self.looper.playing(),
            finding: self.looper.finding(),
            tail: self.looper.tail_playing(),
            speed,
            semis,
            keep_pitch,
            click: self.click.load(Ordering::Relaxed),
            status: self.status.lock().unwrap().clone(),
            recalling: self.recalling.load(Ordering::Relaxed),
            stems_only: self.stems_only(),
            loop_from: -1,
            loop_to: -1,
            ..Default::default()
        };
        if let Some(n) = self.now.lock().unwrap().as_ref() {
            s.title = n.song.title.clone();
            s.artist = n.song.artist.clone();
            s.song_ms = n.ms;
            s.song_playing = n.playing;
        }
        if s.playing {
            if let Some((_, from, to)) = self.loop_in_song() {
                (s.loop_from, s.loop_to) = (from, to);
            }
        }
        if let Some(v) = self.looper.view(1) {
            s.bpm = v.bpm;
            s.meter = v.meter;
            s.bars = v.bars;
            (s.a, s.b, s.beats) = (v.a, v.b, v.b - v.a);
            let len = v.end - v.start;
            s.length = (len / RATE) as f32;
            if len > 0.0 {
                s.pos = ((v.playhead - v.start) / len).clamp(0.0, 1.0) as f32;
            }
        }
        s
    }

    /// The loop for drawing, as JSON (`null` without one): its ends and the
    /// playhead as heard, a bar either side, the beats in view as
    /// `[frame, bar line 0|1, beat index]` and `bins` peak levels across the view.
    /// Frames count from the view's left edge, at 44.1 kHz.
    pub fn view_json(&self, bins: usize) -> String {
        let Some(v) = self.looper.view(bins.clamp(1, 2000)) else {
            return "null".into();
        };
        let rel = |f: f64| (f - v.from).round() as i64;
        let len = v.end - v.start;
        let speed = self.speed.lock().unwrap().0 as f64;
        // what's heard: the lag back from where the looper reads, wrapped into the loop
        let back = self.lag.load(Ordering::Relaxed) as f64 * speed;
        let heard = if len > 0.0 {
            v.start + (v.playhead - v.start - back).rem_euclid(len)
        } else {
            v.playhead
        };
        let beats: Vec<String> = v
            .beats
            .iter()
            .map(|(f, bar, i)| format!("[{},{},{}]", rel(*f), *bar as u8, i))
            .collect();
        let peaks: Vec<String> = v.peaks.iter().map(|p| format!("{:.3}", p)).collect();
        format!(
            "{{\"playing\":{},\"bpm\":{:.1},\"meter\":{},\"bars\":{},\"a\":{},\"b\":{},\
             \"to\":{},\"start\":{},\"end\":{},\"playhead\":{},\"beats\":[{}],\"peaks\":[{}]}}",
            v.playing,
            v.bpm,
            v.meter,
            v.bars,
            v.a,
            v.b,
            rel(v.to),
            rel(v.start),
            rel(v.end),
            rel(heard),
            beats.join(","),
            peaks.join(","),
        )
    }
}

impl State {
    /// One JSON object, for the control socket and D-Bus.
    pub fn json(&self) -> String {
        serde_json::json!({
            "playing": self.playing, "finding": self.finding, "tail": self.tail,
            "bpm": (self.bpm * 10.0).round() / 10.0, "meter": self.meter, "bars": self.bars,
            "a": self.a, "b": self.b, "beats": self.beats,
            "length": (self.length * 1000.0).round() / 1000.0, "pos": (self.pos * 1000.0).round() / 1000.0,
            "speed": self.speed, "semis": self.semis, "keep_pitch": self.keep_pitch,
            "click": self.click, "status": self.status,
            "title": self.title, "artist": self.artist, "song_ms": self.song_ms,
            "song_playing": self.song_playing, "loop_from": self.loop_from, "loop_to": self.loop_to,
            "recalling": self.recalling, "stems_only": self.stems_only,
        })
        .to_string()
    }
}

/// `recall_now` was called off (Live, or another recall).
#[derive(Debug)]
struct Cancelled;
impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("called off")
    }
}
impl std::error::Error for Cancelled {}

fn stems_only_file() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/state"))
        })?;
    Some(base.join("stemflow/stems-only"))
}

fn load_stems_only() -> bool {
    stems_only_file()
        .and_then(|f| std::fs::read_to_string(f).ok())
        .is_some_and(|s| s.trim() == "1")
}

fn save_stems_only(on: bool) {
    if let Some(f) = stems_only_file() {
        if let Some(d) = f.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(f, if on { "1" } else { "0" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beats::BeatModel;

    struct NoNet;
    impl BeatModel for NoNet {
        fn run(&mut self, _: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
            anyhow::bail!("no network")
        }
    }

    #[test]
    fn commands_and_state() {
        let d = Deck::new(
            Looper::new(Box::new(NoNet)),
            Box::new(Unpausable),
            Box::new(|| {}),
        );
        let s = Settings::default();
        d.command("speed 0.5", s).unwrap();
        d.command("pitch -2", s).unwrap();
        d.command("click on", s).unwrap();
        assert!(d.command("sideways", s).is_err());
        assert!(d.command("move 1", s).is_err());
        // nothing heard yet: no loop, and it says so
        d.command("4", s).unwrap();
        let st = d.state();
        assert_eq!((st.speed, st.semis, st.click), (0.5, -2.0, true));
        assert_eq!(st.status, "Nothing heard yet");
        assert!(!st.playing);
        assert!(st.json().contains("\"playing\":false,"), "{}", st.json());
        assert_eq!(d.view_json(10), "null");
        assert!(d.command("edges 3 1", s).is_err());
        d.live();
        assert_eq!(d.state().status, "");
    }
}
