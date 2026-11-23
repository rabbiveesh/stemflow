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

mod command;
mod core;
pub mod panel;
#[cfg(test)]
mod sim;

pub use command::Command;
#[allow(unused_imports)]
pub use command::Verb;
use core::{Core, Event, Job};

use crate::beats::Settings;
use crate::looper::Looper;
use crate::songmap::Song;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// What the player says it is playing.
#[derive(Clone, Debug)]
pub struct Now {
    pub song: Song,
    /// Position in it.
    pub ms: i64,
    pub playing: bool,
    /// Its length, 0 if not known.
    pub len_ms: i64,
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
    /// Play this recording (a bookmark's): by its id, in its app.
    fn open(&self, _song: &Song) -> bool {
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

/// What the audio threads tell the deck, and read back, without waiting on it.
pub struct Io {
    /// Buffer frame being heard now, -1 until something is.
    pub heard: AtomicI64,
    /// Buffer frame of the newest input (what the player is at now), -1 unknown.
    pub input: AtomicI64,
    /// Frames between the looper's read position and the speakers, so a drawn
    /// playhead follows the ears.
    pub lag: AtomicI64,
    /// The loop's speed (f32 bits).
    pub speed: AtomicU32,
    /// Play the plain mix while live, and separate only what is looped.
    pub stems_only: AtomicBool,
}

impl Io {
    pub fn new() -> Self {
        Io {
            heard: AtomicI64::new(-1),
            input: AtomicI64::new(-1),
            lag: AtomicI64::new(0),
            speed: AtomicU32::new(1f32.to_bits()),
            stems_only: AtomicBool::new(load_stems_only()),
        }
    }
}

/// The deck: the state machine (core.rs) behind a lock, fed by the player's watch,
/// a clock tick, the UIs' commands and a worker that runs its slow jobs.
pub struct Deck {
    pub looper: Arc<Looper>,
    source: Arc<dyn Source>,
    core: Mutex<Core>,
    io: Arc<Io>,
    /// Called after anything a UI shows changed.
    notify: Box<dyn Fn() + Send + Sync>,
    epoch: Instant,
    jobs: Mutex<mpsc::Sender<(u64, Job)>>,
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
    /// What the panel's ◀ ▶ step by.
    pub unit: command::Unit,
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

impl Deck {
    pub fn new(
        looper: Looper,
        source: Box<dyn Source>,
        notify: Box<dyn Fn() + Send + Sync>,
    ) -> Arc<Self> {
        let looper = Arc::new(looper);
        let source: Arc<dyn Source> = Arc::from(source);
        let io = Arc::new(Io::new());
        let (tx, rx) = mpsc::channel::<(u64, Job)>();
        let d = Arc::new(Deck {
            looper: looper.clone(),
            source: source.clone(),
            core: Mutex::new(Core::new(looper.clone(), source, io.clone())),
            io,
            notify,
            epoch: Instant::now(),
            jobs: Mutex::new(tx),
        });
        // the slow jobs, one at a time
        let w = Arc::downgrade(&d);
        std::thread::Builder::new()
            .name("loop-jobs".into())
            .spawn(move || {
                // The phone's separator must never wait for the bar finder: this
                // thread (and the beat network's, made at load) run in the background.
                #[cfg(target_os = "android")]
                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, 0, 10);
                }
                for (id, job) in rx {
                    let starts = job.starts();
                    let r = job.run(&looper);
                    let Some(d) = w.upgrade() else { return };
                    d.event(Event::Done { id, starts, r });
                }
            })
            .expect("thread");
        // the clock, for waits and time-outs
        let w = Arc::downgrade(&d);
        std::thread::Builder::new()
            .name("loop-clock".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_millis(100));
                let Some(d) = w.upgrade() else { return };
                d.event(Event::Tick);
            })
            .expect("thread");
        d
    }

    fn event(&self, ev: Event) {
        let t = self.epoch.elapsed().as_millis() as i64;
        let (jobs, changed) = {
            let mut c = self.core.lock().unwrap();
            let j = c.on(t, ev);
            (j, c.take_changed())
        };
        let tx = self.jobs.lock().unwrap();
        for j in jobs {
            let _ = tx.send(j);
        }
        drop(tx);
        if changed {
            (self.notify)();
        }
    }

    /// The worker: buffer frame `frame` is the newest input, what plays now.
    pub fn set_input(&self, frame: i64) {
        self.io.input.store(frame, Ordering::Relaxed);
    }

    /// The player: buffer frame `frame` is what the speakers play now.
    pub fn set_heard(&self, frame: i64) {
        self.io.heard.store(frame, Ordering::Relaxed);
    }

    /// The player: how many frames the speakers trail what it reads.
    pub fn set_lag(&self, frames: i64) {
        self.io.lag.store(frames, Ordering::Relaxed);
    }

    /// The buffer jumps here (frames skipped or dropped): the song map breaks.
    pub fn cut(&self, why: &str) {
        self.event(Event::Cut(why.into()));
    }

    pub fn stems_only(&self) -> bool {
        self.io.stems_only.load(Ordering::Relaxed)
    }

    /// Ask the player where it is, every `every` (not holding the deck meanwhile:
    /// asking can be slow).
    pub fn watch(self: &Arc<Self>, every: Duration) {
        let w = Arc::downgrade(self);
        let source = self.source.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(every);
            let n = source.now();
            let Some(d) = w.upgrade() else { return };
            d.event(Event::Player(n));
        });
    }

    /// Run a text command (the control socket's and D-Bus's `loop ...`; see
    /// `Command`). A command that doesn't parse is an error; one that can't be done
    /// says why in the state's status.
    pub fn command(&self, cmd: &str, s: Settings) -> Result<()> {
        let c: Command = cmd.parse()?;
        self.event(Event::Cmd(c, s));
        Ok(())
    }

    pub fn state(&self) -> State {
        self.core.lock().unwrap().state()
    }

    /// The beat options changed: read the loop's bars again.
    pub fn redecode(&self, s: Settings) {
        self.event(Event::Redecode(s));
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
        let speed = f32::from_bits(self.io.speed.load(Ordering::Relaxed)) as f64;
        // what's heard: the lag back from where the looper reads, wrapped into the loop
        let back = self.io.lag.load(Ordering::Relaxed) as f64 * speed;
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
            "click": self.click, "unit": self.unit.to_string(), "status": self.status,
            "title": self.title, "artist": self.artist, "song_ms": self.song_ms,
            "song_playing": self.song_playing, "loop_from": self.loop_from, "loop_to": self.loop_to,
            "recalling": self.recalling, "stems_only": self.stems_only,
        })
        .to_string()
    }
}

fn stems_only_file() -> Option<std::path::PathBuf> {
    Some(crate::bookmarks::state_dir()?.join("stems-only"))
}

fn load_stems_only() -> bool {
    stems_only_file()
        .and_then(|f| std::fs::read_to_string(f).ok())
        .is_some_and(|s| s.trim() == "1")
}

pub(crate) fn save_stems_only(on: bool) {
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
        d.command("live", s).unwrap();
        assert_eq!(d.state().status, "");
    }
}
