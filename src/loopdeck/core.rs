//! The deck's one owner of state: a state machine fed one event at a time (the clock
//! ticked, the player said where it is, a UI asked for something, a slow job
//! finished). Nothing in here sleeps, spawns or reads a clock, so the simulator
//! (sim.rs) runs the same code as the app, in fake time.
//!
//! What the deck is doing is `Mode`, and what a loop is waiting for is `Work`: each
//! event is matched against them, so a mode that ignores an event says so. The rules
//! that kept breaking when they were spread over threads live in one place each:
//! - what the player does that we didn't ask for (`external`): another song, a seek,
//!   pause or play from the player's own controls;
//! - what plays while a loop waits for the player (`Work`): replacing it calls the old
//!   wait off, and a wait that ends one way or another leaves the player paused.

use super::command::{Command, Unit};
use super::{Io, Now, Source, State};
use crate::beats::Settings;
use crate::bookmarks::{self, Bookmark};
use crate::looper::Looper;
use crate::songmap::{time, Anchor, Song, SongMap, RATE as SRATE};
use anyhow::{anyhow, bail, Context, Result};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Frames per second of the buffer.
const RATE: f64 = 44_100.0;
/// How long the player gets to show what we asked of it before what it does is
/// taken as someone else's doing (ms).
pub(super) const OURS_MS: i64 = 1_500;
/// ...after asking it to open another recording.
const OPEN_MS: i64 = 6_000;
/// How long the player must stay on a song we opened before we seek in it.
const SETTLE_MS: i64 = 1_500;
/// The most song time one fetch can ask for at once (ms): what's wanted must lie
/// within the newest minute of music (see `covering`), with room to spare.
const REACH_MS: i64 = 50_000;

pub enum Event {
    /// The clock moved on (waits and time-outs).
    Tick,
    /// What the player says now.
    Player(Option<Now>),
    Cmd(Command, Settings),
    /// A job (`Job`) finished. `starts`: it sets a loop playing when it works.
    Done {
        id: u64,
        starts: bool,
        r: Result<bool>,
    },
    /// The buffer broke (frames dropped, the model switched).
    Cut(String),
    /// The beat options changed: read the loop's bars again.
    Redecode(Settings),
}

/// Slow work on the looper (the beat network), run off the deck's thread.
pub enum Job {
    /// Loop the last `bars` bars before frame `press`.
    Find {
        press: u64,
        bars: usize,
        s: Settings,
    },
    /// Find the bars up to `end`, then lay the loop on frames `start`..`stop` with
    /// bar 1 at `bar1`, `beats` long if given.
    Place {
        end: u64,
        bars: usize,
        s: Settings,
        start: f64,
        stop: f64,
        bar1: f64,
        beats: Option<usize>,
    },
    /// Lay a saved loop on its saved beats.
    Lay {
        frames: Vec<f64>,
        a: usize,
        b: usize,
        one: usize,
        meter: usize,
        bpm: f32,
        feel: f32,
    },
    /// Find the beats around the loop again, keeping it where it is.
    Refresh,
    /// Read the bars again with this meter and feel.
    Redecode { meter: usize, feel: f32 },
}

impl Job {
    pub fn run(self, l: &Looper) -> Result<bool> {
        match self {
            Job::Find { press, bars, s } => l.find(press, bars, s).map(|_| true),
            Job::Place {
                end,
                bars,
                s,
                start,
                stop,
                bar1,
                beats,
            } => {
                l.find(end, bars, s)?;
                l.set_times(start, stop, bar1);
                // the length in beats is what was asked for, counted from the start,
                // so a slightly different tempo can't add or drop a beat
                if let (Some(n), Some(w)) = (beats, l.view(1)) {
                    if n > 0 && w.b - w.a != n {
                        l.set_edges(w.a, w.a + n);
                    }
                }
                Ok(true)
            }
            Job::Lay {
                frames,
                a,
                b,
                one,
                meter,
                bpm,
                feel,
            } => l.lay(frames, a, b, one, meter, bpm, feel).map(|_| true),
            Job::Refresh => l.refresh(),
            Job::Redecode { meter, feel } => {
                l.redecode(meter, feel);
                Ok(true)
            }
        }
    }

    pub fn starts(&self) -> bool {
        matches!(self, Job::Find { .. } | Job::Place { .. } | Job::Lay { .. })
    }
}

/// What the deck is doing.
enum Mode {
    /// The live stream (or, after a loop, the buffer playing on from it).
    Live,
    /// Finding the bars for a new loop.
    Finding { job: u64, bars: usize },
    /// A loop plays.
    Looping { work: Work },
    /// A bookmark is on its way: the player goes there, then the loop is laid.
    Recalling { b: Box<Bookmark>, work: Work },
}

/// What a loop (or a recall) is waiting for.
enum Work {
    Idle,
    /// The buffer to hold part of the song.
    Fetch(Fetch, Then),
    /// A job.
    Job(u64, Then),
}

/// What to do when the wait is over.
enum Then {
    /// The next phrase is buffered: pause the player.
    Prefetched { r0: i64 },
    /// Beats found ahead of the loop; nothing more.
    Beats,
    /// The beats around were found again for an edit: try it again, else play the
    /// song there.
    Refreshed { da: i64, db: i64 },
    /// The song is buffered for an edit: lay the loop on it.
    Heard(Plan),
    /// An edit's loop is laid.
    Placed { from: i64, to: i64 },
    /// A bookmark's part of the song is buffered: lay it.
    RecallHeard,
    /// A bookmark's loop is laid.
    Recalled,
}

/// An edit the buffer didn't have: the loop's new place in the song.
struct Plan {
    start: i64,
    end: i64,
    bar1: i64,
    beats: usize,
    bars: usize,
    s: Settings,
}

/// Have the buffer hold song ms [from, to] of `song`: wait if the player is already
/// playing through it, else seek there and play (open the recording first, if
/// `open`). `fresh`: what's already buffered doesn't count.
struct Fetch {
    song: Song,
    from: i64,
    to: i64,
    open: bool,
    fresh: bool,
    stage: Stage,
    /// Opened the song again after the player drifted to another.
    reopened: bool,
}

enum Stage {
    Start,
    /// `on_since`: since when the player has said it's on the song.
    Opening {
        until: i64,
        on_since: Option<i64>,
    },
    Waiting {
        until: i64,
    },
}

impl Fetch {
    fn new(song: Song, from: i64, to: i64, open: bool, fresh: bool) -> Self {
        Fetch {
            song,
            from: from.max(0),
            to,
            open,
            fresh,
            stage: Stage::Start,
            reopened: false,
        }
    }
}

/// What the player did that we didn't ask for.
#[derive(Debug, Clone, Copy, PartialEq)]
enum External {
    /// Another song: `natural` when the last one played to its end.
    NextSong {
        natural: bool,
    },
    /// It jumped within the song.
    Seeked,
    Paused,
    /// Played while we held it paused for a loop.
    Played,
}

#[derive(Default)]
struct Expect {
    jump: i64,
    song: i64,
    state: i64,
}

#[derive(Clone, Copy)]
enum Expected {
    Jump,
    Song,
    State,
}

pub struct Core {
    pub looper: Arc<Looper>,
    source: Arc<dyn Source>,
    io: Arc<Io>,
    /// ms on the deck's clock, as of the event being handled
    t: i64,
    mode: Mode,
    /// We paused the player for the loop playing now.
    paused: bool,
    /// It wouldn't pause (no remote control): don't keep asking.
    cant_pause: bool,
    songs: SongMap,
    /// What the player said last.
    now: Option<Now>,
    /// The buffer holds the song unbroken since our last seek (at `since`; None:
    /// since ever) from song ms `from_ms`.
    since: Option<i64>,
    from_ms: i64,
    /// What we asked of the player, and until when it may still be doing it: a
    /// jump in the song (a seek), another song (an open), playing or pausing.
    /// Only that kind of change is taken as ours meanwhile.
    expect: Expect,
    click: bool,
    unit: Unit,
    /// speed, transposition (semitones), keep pitch when slowed
    speed: (f32, f32, bool),
    status: String,
    next_id: u64,
    jobs: Vec<(u64, Job)>,
    changed: bool,
}

fn msg(e: anyhow::Error) -> String {
    format!("{e:#}")
}

impl Core {
    pub fn new(looper: Arc<Looper>, source: Arc<dyn Source>, io: Arc<Io>) -> Self {
        Core {
            looper,
            source,
            io,
            t: 0,
            mode: Mode::Live,
            paused: false,
            cant_pause: false,
            songs: SongMap::default(),
            now: None,
            since: None,
            from_ms: 0,
            expect: Expect::default(),
            click: false,
            unit: Unit::Beat,
            speed: (1.0, 0.0, true),
            status: String::new(),
            next_id: 1,
            jobs: Vec::new(),
            changed: false,
        }
    }

    /// Handle one event at `t` (ms); the jobs to run, each to come back as `Done`.
    pub fn on(&mut self, t: i64, ev: Event) -> Vec<(u64, Job)> {
        self.t = self.t.max(t);
        match ev {
            Event::Tick => {}
            Event::Player(n) => self.player(n),
            Event::Cmd(c, s) => {
                self.command(c, s);
                self.changed = true;
            }
            Event::Done { id, starts, r } => {
                self.done(id, starts, r);
                self.changed = true;
            }
            Event::Cut(why) => self.songs.cut(&why, None),
            Event::Redecode(s) => self.redecode(s.meter, s.feel),
        }
        self.advance();
        std::mem::take(&mut self.jobs)
    }

    /// Whether anything a UI shows changed since last asked.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    fn job(&mut self, j: Job) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.jobs.push((id, j));
        id
    }

    fn say(&mut self, s: impl Into<String>) {
        self.status = s.into();
        self.changed = true;
    }

    fn expect(&mut self, what: Expected, ms: i64) {
        let until = self.t + ms;
        let e = &mut self.expect;
        let slot = match what {
            Expected::Jump => &mut e.jump,
            Expected::Song => &mut e.song,
            Expected::State => &mut e.state,
        };
        *slot = (*slot).max(until);
    }

    fn expected(&self, what: Expected) -> bool {
        self.t
            <= match what {
                Expected::Jump => self.expect.jump,
                Expected::Song => self.expect.song,
                Expected::State => self.expect.state,
            }
    }

    fn on_song(&self, song: &Song) -> bool {
        self.now.as_ref().is_some_and(|n| n.song.same(song))
    }

    fn player_playing(&self) -> bool {
        self.now.as_ref().is_some_and(|n| n.playing)
    }

    // ------------------------------------------------------------ the player

    /// Seek the player to `ms` and play: the buffer holds the song unbroken from
    /// there on.
    fn seek(&mut self, ms: i64, why: &str) {
        let to = ms.max(0);
        eprintln!("loop: seek to {} ({why})", time(to));
        self.songs
            .cut(&format!("seek to {to} ms ({why})"), Some(to));
        self.since = Some(self.t);
        self.from_ms = to;
        self.paused = false;
        self.expect(Expected::Jump, OURS_MS);
        self.expect(Expected::State, OURS_MS);
        self.source.seek(to);
        self.source.play();
    }

    fn pause_source(&mut self) {
        if self.source.pause() {
            self.paused = true;
            self.expect(Expected::State, OURS_MS);
        } else {
            self.cant_pause = true;
        }
    }

    fn play_source(&mut self) {
        self.paused = false;
        self.expect(Expected::State, OURS_MS);
        self.source.play();
    }

    /// The player said `n`: map the buffer to the song, and catch what it did that
    /// we didn't ask for.
    fn player(&mut self, n: Option<Now>) {
        let f = self.io.input.load(Ordering::Relaxed);
        let was = self.player_playing();
        let mut ext = None;
        if let Some(n) = n.as_ref().filter(|n| n.playing && f >= 0) {
            let prev = self.songs.latest().cloned();
            let seen = self
                .songs
                .observe(Anchor::new(f, n.ms, n.song.clone(), self.t));
            if let Some(s) = seen {
                eprintln!("songmap: break at frame {}: {}", s.gap_to, s.why);
                if s.gap_to > s.gap_from {
                    self.looper.mark_gap(s.gap_from, s.gap_to);
                }
                if !self.expected(Expected::Song) && s.why.starts_with("next song") {
                    // played to its end, or skipped?
                    let len = self
                        .now
                        .as_ref()
                        .filter(|o| prev.as_ref().is_some_and(|p| o.song.same(&p.song)))
                        .map_or(0, |o| o.len_ms);
                    let natural = len > 0
                        && prev
                            .as_ref()
                            .is_some_and(|p| p.ms + (self.t - p.at) >= len - 2_500);
                    ext = Some(External::NextSong { natural });
                } else if !self.expected(Expected::Jump) && s.why.starts_with("song at") {
                    ext = Some(External::Seeked);
                }
            }
        }
        let playing = n.as_ref().is_some_and(|n| n.playing);
        // another song while paused: no audio to map, but the person skipped all the same
        if ext.is_none() && !playing && !self.expected(Expected::Song) {
            if let (Some(o), Some(n)) = (self.now.as_ref(), n.as_ref()) {
                if !o.song.same(&n.song) {
                    ext = Some(External::NextSong { natural: false });
                }
            }
        }
        if !playing && was {
            // it played up to where its sound stopped: one stretch to there
            self.songs.extend_to(self.looper.sound_end() as i64);
            self.songs.cut("paused", None);
        }
        if ext.is_none() && !self.expected(Expected::State) {
            if playing && self.paused {
                ext = Some(External::Played);
            } else if !playing && !self.paused && n.is_some() {
                // (a state, not the moment it changed: that moment may have
                // passed while we were waiting for the player to do our bidding)
                ext = Some(External::Paused);
            }
        }
        let key = |n: &Option<Now>| n.as_ref().map(|n| (n.song.title.clone(), n.playing));
        if key(&n) != key(&self.now) {
            self.changed = true;
        }
        self.now = n;
        if let Some(e) = ext {
            self.external(e);
        }
    }

    /// The one place for what the player does that we didn't ask for.
    fn external(&mut self, e: External) {
        if e != External::Paused || !matches!(self.mode, Mode::Live) || self.looper.tail_playing() {
            eprintln!("loop: the player: {e:?}");
        }
        match &self.mode {
            // After a loop the buffer plays on from it, behind the player: a skip, a
            // seek or a pause there means that buffered music isn't wanted any more.
            // A song that played to its end is: the buffer carries on into the next.
            Mode::Live => {
                let drop = match e {
                    External::NextSong { natural } => !natural,
                    External::Seeked | External::Paused => true,
                    External::Played => false,
                };
                if drop && self.looper.tail_playing() {
                    eprintln!("loop: straight to live");
                    self.looper.stop();
                    self.changed = true;
                }
            }
            // The loop is what's heard: the player only plays unheard, for us. Played
            // from its own controls, or skipped to another song, it's wanted again.
            // (A song that ends while we buffer the next phrase isn't a skip.)
            Mode::Looping { .. } => {
                let back = match e {
                    External::Played => Some("played"),
                    External::NextSong { natural: false } => Some("skipped"),
                    _ => None,
                };
                if let Some(why) = back {
                    eprintln!("loop: the player was {why}; back to live");
                    self.paused = false;
                    self.looper.stop();
                    self.mode = Mode::Live;
                    self.say("");
                }
            }
            Mode::Recalling { .. } => {
                if matches!(
                    e,
                    External::NextSong { .. } | External::Seeked | External::Paused
                ) {
                    self.mode = Mode::Live;
                    self.say("Recall called off: the player moved on");
                }
            }
            Mode::Finding { .. } => {}
        }
    }

    // ------------------------------------------------------------ the buffer

    /// The anchor mapping song ms [from, to] into the buffer, if it's all there.
    fn covering(&self, song: &Song, from: i64, to: i64) -> Option<Anchor> {
        if from < self.from_ms {
            return None;
        }
        let lo = (from + 6_000).min(to);
        let x = self.songs.for_song(song, lo, to, self.since)?;
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

    /// The player is playing through [from, ..] already, unbroken since then.
    fn on_the_way(&self, f: &Fetch) -> bool {
        self.player_playing()
            && self.songs.latest().is_some_and(|x| {
                let lo = (f.from + 6_000).min(x.ms);
                x.song.same(&f.song)
                    && f.from >= self.from_ms
                    && x.ms <= f.to
                    && self.songs.for_song(&f.song, lo, x.ms, self.since).is_some()
                    && self.looper.continuous(x.frame_of(lo), x.frame)
            })
    }

    fn fetch_go(&mut self, f: &mut Fetch) {
        if f.fresh || !self.on_the_way(f) {
            self.seek(f.from, &format!("to hear {}–{}", time(f.from), time(f.to)));
        }
        f.stage = Stage::Waiting {
            until: self.t + 60_000 + (f.to - f.from).max(0),
        };
    }

    /// Move a fetch on; its result once it has one.
    fn fetch_step(&mut self, f: &mut Fetch) -> Option<Result<Anchor>> {
        match f.stage {
            Stage::Start => {
                if !f.fresh {
                    if let Some(x) = self.covering(&f.song, f.from, f.to) {
                        return Some(Ok(x));
                    }
                }
                // never seek in another song (the queue may have moved on past it)
                if !self.on_song(&f.song) {
                    if !f.open || f.song.id.is_empty() || !self.source.open(&f.song) {
                        return Some(Err(anyhow!(
                            "The player has moved on from {}",
                            f.song.title
                        )));
                    }
                    self.expect(Expected::Song, OPEN_MS);
                    self.expect(Expected::State, OPEN_MS);
                    f.stage = Stage::Opening {
                        until: self.t + 30_000,
                        on_since: None,
                    };
                    return None;
                }
                self.fetch_go(f);
                None
            }
            Stage::Opening { until, on_since } => {
                // Spotify can show the song a moment, then switch once more (Mac,
                // 2026-11-23: our seek landed in the next track): seek once it has
                // stayed on the song a while
                let on = self.on_song(&f.song);
                let on_since = if on { on_since.or(Some(self.t)) } else { None };
                f.stage = Stage::Opening { until, on_since };
                if on_since.is_some_and(|t0| self.t - t0 >= SETTLE_MS) {
                    self.fetch_go(f);
                } else if self.t > until {
                    return Some(Err(anyhow!("The player didn't open {}", f.song.title)));
                } else {
                    self.expect(Expected::Song, OURS_MS);
                    self.expect(Expected::State, OURS_MS);
                }
                None
            }
            Stage::Waiting { until } => {
                if let Some(x) = self.covering(&f.song, f.from, f.to) {
                    return Some(Ok(x));
                }
                if self.t > until {
                    eprintln!("loop: timed out; map: {}", self.songs.describe());
                    return Some(Err(anyhow!("Timed out waiting for the music")));
                }
                if let Some(x) = self.songs.latest() {
                    if self.since.is_some_and(|t| x.at >= t) && !x.song.same(&f.song) {
                        // went to another song after we opened this one: once more
                        if f.open && !f.reopened {
                            eprintln!(
                                "loop: the player went to {} instead; opening {} again",
                                x.song.title, f.song.title
                            );
                            f.reopened = true;
                            f.stage = Stage::Start;
                            return None;
                        }
                        return Some(Err(anyhow!("The song ended before that part")));
                    }
                }
                None
            }
        }
    }

    /// The loop in song ms (start, end) and the anchor that maps it, if one stretch
    /// of the song holds it.
    fn loop_in_song(&self) -> Option<(Anchor, i64, i64)> {
        let t = self.looper.times()?;
        let a = self.songs.for_frames(t.0.min(t.2), t.1)?;
        let (s, e) = (a.ms_of(t.0), a.ms_of(t.1));
        Some((a, s, e))
    }

    // ------------------------------------------------------------ the loop

    /// A loop has just started: the click and speed it should have.
    fn started(&self) {
        self.looper.set_click(self.click);
        let (sp, semis, keep) = self.speed;
        self.looper.set_speed(sp, semis, keep);
    }

    /// With a loop set: let the player play on (unheard: the loop is what plays)
    /// until the buffer has the next phrase, then pause it, so moving into it is
    /// instant. Without song positions, pause it straight away.
    fn prefetch(&mut self) -> Work {
        let Some((a0, s, e)) = self.loop_in_song() else {
            self.pause_source();
            return Work::Idle;
        };
        // Only what fits in the minute the beats are found in, with the loop and its
        // run-up: a long loop gets less of the next phrase (a fetch reaching further
        // could never be in the buffer at once, and left the player playing on
        // unheard until it timed out: Mac, 2026-11-23).
        let from = s - 6_000;
        let reach = from + REACH_MS;
        let r0 = e;
        let r1 = (e + (e - s) + 1_000).min(reach);
        if r1 < e + 1_000 {
            self.pause_source(); // the loop alone fills the minute
            return Work::Idle;
        }
        // Half the next phrase still buffered, in one stretch with the loop (after an
        // extend into it, say): enough. Otherwise the player plays it (unheard) from
        // a run-up before the loop's start, so the loop and what follows are one
        // stretch, and moving into it needs nothing played again.
        if self
            .covering(&a0.song, from, (e + (e - s) / 2).min(reach))
            .is_some()
        {
            self.pause_source();
            return Work::Idle;
        }
        Work::Fetch(
            Fetch::new(a0.song, from, r1, false, false),
            Then::Prefetched { r0 },
        )
    }

    /// Change what the loop waits for (calling off what it waited for before).
    fn set_work(&mut self, f: impl FnOnce(&mut Self, Work) -> Work) {
        if let Mode::Looping { work } = std::mem::replace(&mut self.mode, Mode::Live) {
            let w = f(self, work);
            self.mode = Mode::Looping { work: w };
        }
    }

    /// Move the loop's start by `da` beats and its end by `db`. In the buffer at
    /// once; a part not buffered yet is played first (the old loop keeps playing
    /// meanwhile) and placed by song time, keeping bar 1.
    fn adjust(&mut self, w: Work, da: i64, db: i64) -> Work {
        if da == 0 && db == 0 {
            return w;
        }
        if self.looper.nudge(da, db) {
            self.say("");
            return if db > 0 { self.prefetch() } else { w };
        }
        if da >= 0 && db <= 0 {
            self.say("The loop can't get shorter than that");
            return w;
        }
        // Only the end moving right can gain from the beats of what was buffered
        // since (a prefetched phrase): the newest minute only ever moves later, so
        // it never holds more before the loop than the loop's own copy. The start
        // moving left goes straight to the song (which takes it from the buffer
        // when it's there, without playing it again).
        if db <= 0 {
            return self.stream_to(da, db, false);
        }
        // Find the beats there (keeping the loop as it is) and try again. Only if
        // that isn't enough is the song played again.
        let id = self.job(Job::Refresh);
        Work::Job(id, Then::Refreshed { da, db })
    }

    /// Play the song from before the loop's new place through it, then lay the loop
    /// there. `fresh`: play it again even if it's buffered (a repair).
    fn stream_to(&mut self, da: i64, db: i64, fresh: bool) -> Work {
        match self.plan(da, db) {
            Ok((song, p)) => {
                self.say(format!("Listening to {}–{}…", time(p.start), time(p.end)));
                Work::Fetch(
                    Fetch::new(
                        song,
                        p.start.min(p.bar1) - 6_000,
                        p.end + 1_000,
                        fresh,
                        fresh,
                    ),
                    Then::Heard(p),
                )
            }
            Err(e) => {
                self.say(msg(e));
                Work::Idle
            }
        }
    }

    fn plan(&self, da: i64, db: i64) -> Result<(Song, Plan)> {
        let (t0, t1, bar1f, meter, bars, feel) = self.looper.times().context("No loop")?;
        let v = self.looper.view(1).context("No loop")?;
        let a0 = self.songs.for_frames(t0.min(bar1f), t1).with_context(|| {
            if self.now.is_none() {
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
        let hint = v.bpm / feel;
        let beats = ((v.b - v.a) as i64 + db - da).max(0) as usize;
        Ok((
            a0.song,
            Plan {
                start: new_start,
                end: new_end,
                bar1,
                beats,
                bars,
                s: Settings {
                    meter,
                    feel,
                    min_bpm: hint * 0.85,
                    max_bpm: hint * 1.18,
                },
            },
        ))
    }

    /// A fetch for the loop ended.
    fn fetched(&mut self, r: Result<Anchor>, then: Then) -> Work {
        match (then, r) {
            (Then::Prefetched { r0 }, Ok(a)) => {
                eprintln!("loop: next phrase buffered");
                self.pause_source();
                // its beats too, so moving into it needs no analysis; only when the
                // buffered run reaches back into the loop
                let t = self.looper.times();
                if t.is_some_and(|t| a.frame_of(r0 - 6_000) as f64 <= t.1) {
                    let id = self.job(Job::Refresh);
                    return Work::Job(id, Then::Beats);
                }
                Work::Idle
            }
            (Then::Prefetched { .. }, Err(e)) => {
                eprintln!("loop: next phrase: {e:#}");
                self.pause_source();
                Work::Idle
            }
            (Then::Heard(p), Ok(a)) => {
                self.say("Finding the bars…");
                let id = self.job(Job::Place {
                    end: a.frame_of(p.end).max(0) as u64,
                    bars: p.bars,
                    s: p.s,
                    start: a.frame_of(p.start) as f64,
                    stop: a.frame_of(p.end) as f64,
                    bar1: a.frame_of(p.bar1) as f64,
                    beats: Some(p.beats),
                });
                Work::Job(
                    id,
                    Then::Placed {
                        from: p.start,
                        to: p.end,
                    },
                )
            }
            (Then::Heard(_), Err(e)) => {
                self.say(msg(e));
                self.pause_source();
                Work::Idle
            }
            (_, _) => Work::Idle,
        }
    }

    /// A job for the loop finished.
    fn finished(&mut self, r: Result<bool>, then: Then) -> Work {
        match then {
            Then::Beats => {
                if let Err(e) = r {
                    eprintln!("loop: beats ahead: {e:#}");
                }
                Work::Idle
            }
            Then::Refreshed { da, db } => {
                match r {
                    Ok(true) => eprintln!("loop: beats around found"),
                    Ok(false) => {}
                    Err(e) => eprintln!("loop: beats around: {e:#}"),
                }
                if self.looper.nudge(da, db) {
                    self.say("");
                    if db > 0 {
                        self.prefetch()
                    } else {
                        Work::Idle
                    }
                } else {
                    self.stream_to(da, db, false)
                }
            }
            Then::Placed { from, to } => match r {
                Ok(_) => {
                    self.started();
                    eprintln!("loop: now {}–{}", time(from), time(to));
                    self.say("");
                    self.prefetch()
                }
                Err(e) => {
                    self.say(msg(e));
                    Work::Idle
                }
            },
            _ => Work::Idle,
        }
    }

    // ------------------------------------------------------------ events

    /// Move whatever waits on; then, whatever led here, a loop waiting for nothing
    /// leaves the player paused (it would only play unheard).
    fn advance(&mut self) {
        self.step();
        if matches!(self.mode, Mode::Looping { work: Work::Idle })
            && self.player_playing()
            && !self.paused
            && !self.cant_pause
        {
            self.pause_source();
        }
    }

    fn step(&mut self) {
        let mode = std::mem::replace(&mut self.mode, Mode::Live);
        self.mode = match mode {
            // the loop went (Live was pressed, or a stale job stopped it)
            Mode::Looping { .. } if !self.looper.playing() => Mode::Live,
            Mode::Looping {
                work: Work::Fetch(mut f, then),
            } => match self.fetch_step(&mut f) {
                None => Mode::Looping {
                    work: Work::Fetch(f, then),
                },
                Some(r) => {
                    let w = self.fetched(r, then);
                    self.mode = Mode::Looping { work: w };
                    // what it led to may already be ready
                    return self.step();
                }
            },
            Mode::Recalling {
                b,
                work: Work::Fetch(mut f, then),
            } => match self.fetch_step(&mut f) {
                None => {
                    if matches!(f.stage, Stage::Waiting { .. }) {
                        if let Some(x) = self.songs.latest() {
                            if self.since.is_some_and(|t| x.at >= t) && x.song.same(&b.song) {
                                let s = format!("Listening up to the loop… {}", time(x.ms));
                                if s != self.status {
                                    self.say(s);
                                }
                            }
                        }
                    }
                    Mode::Recalling {
                        b,
                        work: Work::Fetch(f, then),
                    }
                }
                Some(Ok(a)) => self.lay_recall(b, a),
                Some(Err(e)) => {
                    self.say(msg(e));
                    Mode::Live
                }
            },
            m => m,
        };
    }

    fn done(&mut self, id: u64, starts: bool, r: Result<bool>) {
        let mode = std::mem::replace(&mut self.mode, Mode::Live);
        self.mode = match mode {
            Mode::Finding { job, bars } if job == id => self.found(r, bars),
            Mode::Looping {
                work: Work::Job(j, then),
            } if j == id => {
                let w = self.finished(r, then);
                Mode::Looping { work: w }
            }
            Mode::Recalling {
                b,
                work: Work::Job(j, Then::Recalled),
            } if j == id => self.recalled(b, r),
            m => {
                // overtaken: a loop it set going isn't wanted
                if starts && r.is_ok() && !matches!(m, Mode::Looping { .. }) {
                    self.looper.stop();
                }
                m
            }
        };
    }

    /// The bars for a new loop were found (or not).
    fn found(&mut self, r: Result<bool>, want: usize) -> Mode {
        let got = self.looper.view(1).map(|v| (v.b - v.a, v.meter.max(1)));
        match r {
            Ok(_) => {
                self.started();
                eprintln!("loop: bars found");
                // fewer bars when the song (or a seek, or a pause) began since
                let note = match got {
                    Some((n, m)) if n < want * m => format!(
                        "Only {} of {want} bars: the song started (or was seeked or paused) just before",
                        if n % m == 0 {
                            format!("{}", n / m)
                        } else {
                            format!("{:.1}", n as f32 / m as f32)
                        }
                    ),
                    _ => String::new(),
                };
                self.say(note);
                self.mode = Mode::Looping { work: Work::Idle };
                // Nobody hears the song now: pause it (once the next phrase is
                // buffered), so live picks up from here.
                let w = self.prefetch();
                Mode::Looping { work: w }
            }
            Err(e) => {
                eprintln!("loop: {e:#}");
                self.say(format!("No loop: {e:#}"));
                Mode::Live
            }
        }
    }

    fn lay_recall(&mut self, b: Box<Bookmark>, a: Anchor) -> Mode {
        let id = if b.has_grid() {
            self.job(Job::Lay {
                frames: b
                    .beats
                    .iter()
                    .map(|&ms| a.frame as f64 + (ms - a.ms as f64) * RATE / 1000.0)
                    .collect(),
                a: b.a,
                b: b.b,
                one: b.one,
                meter: b.meter,
                bpm: b.bpm,
                feel: b.feel,
            })
        } else {
            self.say("Finding the bars…");
            let d = Settings::default();
            let hint = b.bpm / b.feel;
            let s = Settings {
                meter: b.meter,
                feel: b.feel,
                min_bpm: if b.bpm > 0.0 { hint * 0.85 } else { d.min_bpm },
                max_bpm: if b.bpm > 0.0 { hint * 1.18 } else { d.max_bpm },
            };
            self.job(Job::Place {
                end: a.frame_of(b.end_ms).max(0) as u64,
                bars: b.bars,
                s,
                start: a.frame_of(b.start_ms) as f64,
                stop: a.frame_of(b.end_ms) as f64,
                bar1: a.frame_of(b.bar1_ms) as f64,
                beats: None,
            })
        };
        Mode::Recalling {
            b,
            work: Work::Job(id, Then::Recalled),
        }
    }

    fn recalled(&mut self, b: Box<Bookmark>, r: Result<bool>) -> Mode {
        if let Err(e) = r {
            self.say(msg(e));
            self.looper.stop();
            return Mode::Live;
        }
        // keep the beats now, so next time there's nothing to find
        if !b.has_grid() {
            if let Some(g) = self.grid() {
                let _ = bookmarks::replace(&b.as_ref().clone().with_grid(g));
            }
        }
        self.started();
        eprintln!("loop: recalled {}", b.name);
        self.say("");
        self.mode = Mode::Looping { work: Work::Idle };
        let w = self.prefetch();
        Mode::Looping { work: w }
    }

    fn command(&mut self, c: Command, s: Settings) {
        match c {
            Command::Bars(n) => {
                if self.looper.playing() {
                    self.looper.set_bars(n);
                    self.say("");
                } else {
                    self.find(n, s);
                }
            }
            Command::Live => self.live(),
            Command::Top => {
                self.looper.play();
                if self.looper.playing() {
                    if !self.paused {
                        self.pause_source();
                    }
                    if !matches!(self.mode, Mode::Looping { .. }) {
                        self.mode = Mode::Looping { work: Work::Idle };
                    }
                }
            }
            Command::Move { start, end } => {
                if self.looper.playing() {
                    self.set_work(|me, w| me.adjust(w, start, end));
                }
            }
            Command::Repair => {
                if self.looper.playing() {
                    self.set_work(|me, _| me.stream_to(0, 0, true));
                }
            }
            Command::Unit(u) => self.unit = u,
            Command::Edges { a, b } => {
                let before = self.looper.view(1).map(|v| (v.a, v.b));
                self.looper.set_edges(a, b);
                if self.looper.view(1).map(|v| (v.a, v.b)) == before {
                    self.say("The loop can't go there: not heard yet, or across a pause");
                } else {
                    self.say("");
                }
            }
            Command::Bar1(i) => self.looper.bar_one(i),
            Command::Click(v) => {
                self.click = v;
                self.looper.set_click(v);
            }
            // (the app sets these as its beat options, which then read the bars
            // again: see Control::loop_command)
            Command::Meter(m) => self.redecode(m as usize, s.feel),
            Command::Feel(f) => self.redecode(s.meter, f),
            Command::Speed(x) => self.set_speed((x, self.speed.1, self.speed.2)),
            Command::Pitch(x) => self.set_speed((self.speed.0, x, self.speed.2)),
            Command::KeepPitch(v) => self.set_speed((self.speed.0, self.speed.1, v)),
            Command::Save(name) => {
                if let Err(e) = self.save(&name) {
                    self.say(msg(e));
                }
            }
            Command::Recall(created) => self.recall(created),
            Command::Rename { created, name } => {
                if let Err(e) = bookmarks::rename(created, &name) {
                    self.say(msg(e));
                }
            }
            Command::Delete(created) => {
                if let Err(e) = bookmarks::remove(created) {
                    self.say(msg(e));
                }
            }
            Command::PlayPause => self.play_pause(),
            Command::StemsOnly(v) => {
                self.io.stems_only.store(v, Ordering::Relaxed);
                super::save_stems_only(v);
            }
        }
    }

    fn redecode(&mut self, meter: usize, feel: f32) {
        if self.looper.playing() {
            // (the beat network may run: off this thread)
            self.job(Job::Redecode { meter, feel });
        }
    }

    fn set_speed(&mut self, sp: (f32, f32, bool)) {
        self.speed = sp;
        self.io.speed.store(sp.0.to_bits(), Ordering::Relaxed);
        self.looper.set_speed(sp.0, sp.1, sp.2);
    }

    /// Loop the last `bars` bars of what is being heard.
    fn find(&mut self, bars: usize, s: Settings) {
        // live up to now: unbroken, as far as we know
        self.since = None;
        self.from_ms = 0;
        let press = self.io.heard.load(Ordering::Relaxed);
        if press < 0 {
            self.mode = Mode::Live;
            return self.say("Nothing heard yet");
        }
        self.say("Finding the bars…");
        let job = self.job(Job::Find {
            press: press as u64,
            bars,
            s,
        });
        self.mode = Mode::Finding { job, bars };
    }

    /// Back to the live stream. If we paused the player, the music carries on from
    /// the loop (through what was buffered after it) and the player plays again; if
    /// not, it went on unheard, so jump straight to it.
    fn live(&mut self) {
        self.cant_pause = false;
        let ours = self.paused || self.player_playing();
        let looping = matches!(self.mode, Mode::Looping { .. }) && self.looper.playing();
        if looping && ours {
            self.looper.go_live();
            self.play_source();
        } else {
            self.paused = false;
            self.looper.stop();
        }
        self.mode = Mode::Live;
        self.say("");
    }

    fn play_pause(&mut self) {
        if self.player_playing() {
            // this pause is all we're waiting on now: what we asked before (opening
            // a song, a seek) is called off, so the person playing it again counts
            self.expect = Expect {
                jump: 0,
                song: 0,
                state: self.t + OURS_MS,
            };
            self.source.pause();
            match &self.mode {
                // the buffered music after a loop stops with it
                Mode::Live => self.looper.stop(),
                Mode::Looping { .. } => {
                    self.paused = true;
                    self.set_work(|_, _| Work::Idle);
                    self.say("");
                }
                Mode::Recalling { .. } => {
                    self.mode = Mode::Live;
                    self.say("");
                }
                Mode::Finding { .. } => {}
            }
        } else if matches!(self.mode, Mode::Looping { .. }) && self.paused {
            // the song from where the loop is, as Live does
            self.live();
        } else {
            self.play_source();
        }
    }

    /// Play a bookmark: the player to that recording and to a run-up before the
    /// loop, then, once its stems are in the buffer, lay the loop where it was saved.
    fn recall(&mut self, created: i64) {
        let Some(b) = bookmarks::all().into_iter().find(|b| b.created == created) else {
            return self.say("No such bookmark");
        };
        self.say(format!(
            "Going to {} at {}…",
            b.song.title,
            time(b.start_ms)
        ));
        self.looper.stop(); // straight to live: we listen to the run-up
        self.paused = false;
        // With its own beats, only the stems are needed: a short run-up for the
        // separator to settle, then up to the last kept beat. Without, enough before
        // the loop for the beat model to find the tempo.
        let own = b.has_grid();
        let (first, last) = if own {
            (b.beats[0] as i64, *b.beats.last().unwrap() as i64)
        } else {
            (b.start_ms, b.end_ms)
        };
        let seek_to =
            (first.min(b.start_ms).min(b.bar1_ms) - if own { 4_000 } else { 12_000 }).max(0);
        let hi = last + if own { 500 } else { 1500 };
        if !self.on_song(&b.song) && b.song.id.is_empty() {
            return self.say(format!(
                "Play {} in the player, then try again",
                b.song.title
            ));
        }
        let f = Fetch::new(b.song.clone(), seek_to, hi, true, true);
        self.mode = Mode::Recalling {
            b: Box::new(b),
            work: Work::Fetch(f, Then::RecallHeard),
        };
    }

    /// Save the loop as a bookmark called `name`.
    fn save(&self, name: &str) -> Result<()> {
        let t = self.looper.times().context("No loop to save")?;
        let a = self.songs.for_frames(t.0.min(t.2), t.1).with_context(|| {
            if self.now.is_none() {
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
        let m = &self.songs;
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

    pub fn state(&self) -> State {
        let (speed, semis, keep_pitch) = self.speed;
        let mut s = State {
            playing: self.looper.playing(),
            finding: self.looper.finding() || matches!(self.mode, Mode::Finding { .. }),
            tail: self.looper.tail_playing(),
            speed,
            semis,
            keep_pitch,
            click: self.click,
            unit: self.unit,
            status: self.status.clone(),
            recalling: matches!(self.mode, Mode::Recalling { .. }),
            stems_only: self.io.stems_only.load(Ordering::Relaxed),
            loop_from: -1,
            loop_to: -1,
            ..Default::default()
        };
        if let Some(n) = self.now.as_ref() {
            s.title = n.song.title.clone();
            s.artist = n.song.artist.clone();
            s.song_ms = n.ms;
            s.song_playing = n.playing;
        }
        if s.playing {
            if let Some((a, from, to)) = self.loop_in_song() {
                (s.loop_from, s.loop_to) = (from, to);
                // the loop's song, not whatever the player has moved on to
                s.title = a.song.title;
                s.artist = a.song.artist;
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

    /// For the simulator's checks.
    #[cfg(test)]
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            Mode::Live => "live",
            Mode::Finding { .. } => "finding",
            Mode::Looping { .. } => "looping",
            Mode::Recalling { .. } => "recalling",
        }
    }

    #[cfg(test)]
    pub fn paused_by_us(&self) -> bool {
        self.paused
    }

    /// A loop (or recall) waits for the player or a job.
    #[cfg(test)]
    pub fn waiting(&self) -> bool {
        match &self.mode {
            Mode::Looping { work } | Mode::Recalling { work, .. } => !matches!(work, Work::Idle),
            Mode::Finding { .. } => true,
            Mode::Live => false,
        }
    }
}
