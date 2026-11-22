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

use crate::beats::Settings;
use crate::looper::Looper;
use anyhow::{bail, Context, Result};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The app whose audio is looped, as far as pausing it goes.
pub trait Source: Send + Sync {
    /// Pause it; false if that isn't possible (no such player, no permission).
    fn pause(&self) -> bool;
    /// Play it again.
    fn play(&self) -> bool;
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
        })
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
                    me.looper.set_click(me.click.load(Ordering::Relaxed));
                    let (sp, semis, keep) = *me.speed.lock().unwrap();
                    me.looper.set_speed(sp, semis, keep);
                    eprintln!(
                        "loop: {bars} bars found in {:.2}s",
                        t.elapsed().as_secs_f32()
                    );
                    // Nobody hears the song now: pause it, so live picks up from here.
                    if me.source.pause() {
                        me.paused.store(true, Ordering::Relaxed);
                    }
                    me.say("");
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
        if self.looper.playing() && self.paused.swap(false, Ordering::Relaxed) {
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
    /// `speed <x>`, `pitch <semitones>`, `keep_pitch on|off`.
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
                if !self.looper.nudge(da, db) {
                    self.say("The loop can't go there: not heard yet, or across a pause");
                } else {
                    self.say("");
                }
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
            ..Default::default()
        };
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
        let esc: String = self
            .status
            .chars()
            .flat_map(|c| match c {
                '"' => vec!['\\', '"'],
                '\\' => vec!['\\', '\\'],
                c if (c as u32) < 0x20 => vec![' '],
                c => vec![c],
            })
            .collect();
        format!(
            "{{\"playing\":{},\"finding\":{},\"tail\":{},\"bpm\":{:.1},\"meter\":{},\"bars\":{},\
             \"a\":{},\"b\":{},\"beats\":{},\"length\":{:.3},\"pos\":{:.3},\"speed\":{},\"semis\":{},\
             \"keep_pitch\":{},\"click\":{},\"status\":\"{esc}\"}}",
            self.playing,
            self.finding,
            self.tail,
            self.bpm,
            self.meter,
            self.bars,
            self.a,
            self.b,
            self.beats,
            self.length,
            self.pos,
            self.speed,
            self.semis,
            self.keep_pitch,
            self.click,
        )
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
        assert!(
            st.json().starts_with("{\"playing\":false,"),
            "{}",
            st.json()
        );
        assert_eq!(d.view_json(10), "null");
        assert!(d.command("edges 3 1", s).is_err());
        d.live();
        assert_eq!(d.state().status, "");
    }
}
