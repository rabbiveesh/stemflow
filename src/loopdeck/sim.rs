//! The deck in fake time: a fake player (songs, seeks, pauses, skips), fake music
//! (a click on every beat, a thump on every bar line, and the song and position
//! written into the band stem so what's heard can be read back) and a fake beat
//! network that finds those clicks. The same core as the app runs on it, one 50 ms
//! block at a time, so whole sessions (loop, extend, Live, skip...) run as tests in
//! a few seconds, with no audio device, player or model.
//!
//! The scripted tests are bugs that were heard live; `random_sessions` throws random
//! sequences of what a person might do at it and checks the rules after each step.

use super::command::Command;
use super::core::{Core, Event};
use super::{Io, Now, Source};
use crate::beats::{BeatModel, Settings, MELS};
use crate::looper::Looper;
use crate::songmap::Song;
use anyhow::Result;
use std::sync::{Arc, Mutex};

const SR: usize = 44_100;
/// One step of the simulation: 50 ms.
const BLOCK: usize = SR / 20;
const STEP_MS: i64 = 50;

#[derive(Clone)]
pub struct Tune {
    pub song: Song,
    pub len_ms: i64,
    pub bpm: f64,
    /// Where the first beat is (ms).
    pub offset: f64,
}

pub fn tune(n: usize, len_s: i64, bpm: f64) -> Tune {
    Tune {
        song: Song {
            id: format!("sim:track:{n}"),
            title: format!("Song {n}"),
            artist: "Sim".into(),
            album: String::new(),
            ..Default::default()
        },
        len_ms: len_s * 1000,
        bpm,
        offset: 0.0,
    }
}

pub struct Player {
    pub tunes: Vec<Tune>,
    pub cur: usize,
    pub ms: f64,
    pub playing: bool,
    /// What the deck asked of it.
    pub seeks: Vec<i64>,
    pub opens: usize,
    pub pauses: usize,
    pub plays: usize,
}

impl Player {
    /// The person skips to the next song.
    pub fn skip(&mut self) {
        self.cur = (self.cur + 1) % self.tunes.len();
        self.ms = 0.0;
    }
}

struct FakePlayer(Arc<Mutex<Player>>);

impl Source for FakePlayer {
    fn pause(&self) -> bool {
        let mut p = self.0.lock().unwrap();
        p.playing = false;
        p.pauses += 1;
        true
    }
    fn play(&self) -> bool {
        let mut p = self.0.lock().unwrap();
        p.playing = true;
        p.plays += 1;
        true
    }
    fn now(&self) -> Option<Now> {
        let p = self.0.lock().unwrap();
        let t = &p.tunes[p.cur];
        Some(Now {
            song: t.song.clone(),
            ms: p.ms as i64,
            playing: p.playing,
            len_ms: t.len_ms,
        })
    }
    fn seek(&self, ms: i64) -> bool {
        let mut p = self.0.lock().unwrap();
        p.ms = ms as f64;
        p.seeks.push(ms);
        true
    }
    fn open(&self, song: &Song) -> bool {
        let mut p = self.0.lock().unwrap();
        let Some(i) = p.tunes.iter().position(|t| t.song.id == song.id) else {
            return false;
        };
        p.cur = i;
        p.ms = 0.0;
        p.playing = true;
        p.opens += 1;
        true
    }
}

/// Song `i` at `ms`, written into a band sample pair exactly (i16 steps): the left
/// carries the song and second, the right the ms in the second. Zero is silence.
fn code(i: usize, ms: i64) -> [f32; 2] {
    let hi = (i as i64 + 1) * 1000 + ms / 1000;
    let lo = ms % 1000;
    [(hi as f32 + 0.5) / 32767.0, (lo as f32 + 0.5) / 32767.0]
}

fn decode(l: f32, r: f32) -> Option<(usize, i64)> {
    let hi = (l * 32767.0).round() as i64;
    let lo = (r * 32767.0).round() as i64;
    if hi < 1000 {
        return None;
    }
    Some(((hi / 1000 - 1) as usize, (hi % 1000) * 1000 + lo))
}

/// Finds the sim's clicks: a beat where the highs jump, a downbeat where the lows do.
struct ClickNet;

impl BeatModel for ClickNet {
    fn run(&mut self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = spect.len() / MELS;
        let band = |t: usize, a: usize, b: usize| {
            spect[t * MELS + a..t * MELS + b].iter().sum::<f32>() / (b - a) as f32
        };
        let pick = |a: usize, b: usize, frac: f32| {
            let e: Vec<f32> = (0..n).map(|t| band(t, a, b)).collect();
            let on: Vec<f32> = (0..n)
                .map(|t| {
                    if t == 0 {
                        0.0
                    } else {
                        (e[t] - e[t - 1]).max(0.0)
                    }
                })
                .collect();
            let max = on.iter().cloned().fold(0.0, f32::max);
            (0..n)
                .map(|t| {
                    let peak = on[t] > frac * max
                        && (t == 0 || on[t] >= on[t - 1])
                        && (t + 1 == n || on[t] > on[t + 1]);
                    if max > 0.0 && peak {
                        8.0
                    } else {
                        -8.0
                    }
                })
                .collect::<Vec<f32>>()
        };
        Ok((pick(40, MELS, 0.4), pick(0, 6, 0.6)))
    }
}

pub struct Sim {
    pub core: Core,
    pub looper: Arc<Looper>,
    pub io: Arc<Io>,
    pub p: Arc<Mutex<Player>>,
    pub t: i64,
    pub poll_ms: i64,
    next_poll: i64,
    /// What was heard, each step: (t, song and ms, or silence).
    pub heard: Vec<(i64, Option<(usize, i64)>)>,
    pub s: Settings,
    noise: u32,
}

impl Sim {
    pub fn new(tunes: Vec<Tune>) -> Self {
        let looper = Arc::new(Looper::new(Box::new(ClickNet)));
        let io = Arc::new(Io::new());
        io.stems_only
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let p = Arc::new(Mutex::new(Player {
            tunes,
            cur: 0,
            ms: 0.0,
            playing: false,
            seeks: Vec::new(),
            opens: 0,
            pauses: 0,
            plays: 0,
        }));
        let core = Core::new(looper.clone(), Arc::new(FakePlayer(p.clone())), io.clone());
        Sim {
            core,
            looper,
            io,
            p,
            t: 0,
            poll_ms: 1000,
            next_poll: 1000,
            heard: Vec::new(),
            s: Settings {
                meter: 4,
                ..Settings::default()
            },
            noise: 12345,
        }
    }

    fn rand(&mut self) -> f32 {
        self.noise = self
            .noise
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.noise >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
    }

    /// One block of what the player plays now (silence while paused).
    fn music(&mut self) -> (Vec<f32>, Vec<f32>, Option<(usize, i64)>) {
        let mut bass = vec![0f32; 2 * BLOCK];
        let mut band = vec![0f32; 2 * BLOCK];
        let (playing, mut cur, mut ms) = {
            let p = self.p.lock().unwrap();
            (p.playing, p.cur, p.ms)
        };
        if !playing {
            return (bass, band, None);
        }
        let mid = (cur, ms as i64 + 25);
        for i in 0..BLOCK {
            let t = self.p.lock().unwrap().tunes[cur].clone();
            if ms >= t.len_ms as f64 {
                // the next song in the queue
                let mut p = self.p.lock().unwrap();
                p.cur = (cur + 1) % p.tunes.len();
                cur = p.cur;
                ms = 0.0;
            }
            let beat = 60_000.0 / t.bpm;
            let x = ms - t.offset;
            let (k, dt) = if x < 0.0 {
                (-1, 1e9)
            } else {
                ((x / beat).floor() as i64, x - (x / beat).floor() * beat)
            };
            let mut v = 0.0;
            if dt < 30.0 {
                v += 0.5 * (-dt / 6.0).exp() as f32 * self.rand();
                if k % 4 == 0 {
                    let ph = 2.0 * std::f64::consts::PI * 80.0 * dt / 1000.0;
                    v += 0.6 * (ph.sin() * (-dt / 15.0).exp()) as f32;
                }
            }
            bass[2 * i] = v;
            bass[2 * i + 1] = v;
            let c = code(cur, ms as i64);
            band[2 * i] = c[0];
            band[2 * i + 1] = c[1];
            ms += 1000.0 / SR as f64;
        }
        let mut p = self.p.lock().unwrap();
        p.cur = cur;
        p.ms = ms;
        (bass, band, Some(mid))
    }

    /// One 50 ms step: the player plays a block into the buffer, the speakers play
    /// the loop's block or the live one, the clock ticks, the player is asked.
    pub fn step(&mut self) {
        let (bass, band, live) = self.music();
        self.looper.push(&bass, &band, &[]);
        let total = self.looper.total() as i64;
        self.io
            .input
            .store(total, std::sync::atomic::Ordering::Relaxed);
        let (mut lb, mut lr, mut lc) = (
            vec![0f32; 2 * BLOCK],
            vec![0f32; 2 * BLOCK],
            vec![0f32; BLOCK],
        );
        let heard = match self
            .looper
            .read(&mut lb, &mut lr, &mut [], &mut lc, BLOCK, BLOCK)
        {
            Some(at) => {
                self.io
                    .heard
                    .store(at, std::sync::atomic::Ordering::Relaxed);
                let i = BLOCK / 2;
                decode(lr[2 * i], lr[2 * i + 1])
            }
            None => {
                self.io
                    .heard
                    .store(total - BLOCK as i64, std::sync::atomic::Ordering::Relaxed);
                live
            }
        };
        self.heard.push((self.t, heard));
        self.t += STEP_MS;
        self.event(Event::Tick);
        if self.t >= self.next_poll {
            self.next_poll += self.poll_ms;
            let n = FakePlayer(self.p.clone()).now();
            self.event(Event::Player(n));
        }
    }

    pub fn run(&mut self, ms: i64) {
        for _ in 0..(ms / STEP_MS) {
            self.step();
        }
    }

    fn event(&mut self, ev: Event) {
        let jobs = self.core.on(self.t, ev);
        // the slow jobs, done at once (they take a second or so in the app)
        for (id, j) in jobs {
            let starts = j.starts();
            let r = j.run(&self.looper);
            self.event(Event::Done { id, starts, r });
        }
    }

    pub fn cmd(&mut self, c: &str) {
        let c: Command = c.parse().unwrap();
        self.event(Event::Cmd(c, self.s));
    }

    pub fn play(&mut self) {
        self.p.lock().unwrap().playing = true;
    }

    /// What's heard now.
    pub fn now_heard(&self) -> Option<(usize, i64)> {
        self.heard.last().and_then(|h| h.1)
    }

    /// The player's song and position.
    pub fn player(&self) -> (usize, i64, bool) {
        let p = self.p.lock().unwrap();
        (p.cur, p.ms as i64, p.playing)
    }

    pub fn status(&self) -> String {
        self.core.state().status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_songs() -> Sim {
        Sim::new(vec![tune(0, 120, 120.0), tune(1, 120, 100.0)])
    }

    /// A 4-bar loop at 120 BPM is 8 s: what's heard keeps coming back to its start.
    #[test]
    fn a_loop_repeats() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        assert_eq!(s.core.mode_name(), "looping", "{}", s.status());
        let st = s.core.state();
        assert_eq!(st.beats, 16, "{st:?}");
        assert!((st.length - 8.0).abs() < 0.1, "{st:?}");
        s.run(20_000);
        // the player is paused once the next phrase is buffered
        assert!(!s.player().2);
        assert!(s.core.paused_by_us());
        // heard: inside 8 s of song time, again and again
        let late: Vec<i64> = s.heard[s.heard.len() - 200..]
            .iter()
            .filter_map(|h| h.1.map(|x| x.1))
            .collect();
        let (lo, hi) = (late.iter().min().unwrap(), late.iter().max().unwrap());
        assert!(hi - lo <= 8_100 && hi - lo >= 7_500, "{lo}..{hi}");
        assert!(*hi <= 40_500, "the loop ends where it was pressed: {hi}");
    }

    /// Heard at `ms` after now... the song heard within the last `ms`.
    fn heard_since(s: &Sim, ms: i64) -> Vec<Option<(usize, i64)>> {
        s.heard
            .iter()
            .filter(|h| h.0 >= s.t - ms)
            .map(|h| h.1)
            .collect()
    }

    /// A loop, Live, then 16 s of buffer behind the player: switching songs in the
    /// player must be heard at once, not after the old song's buffered music.
    #[test]
    fn a_skip_after_live_is_heard_at_once() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(20_000);
        s.cmd("live");
        s.run(3_000);
        assert!(s.core.state().tail, "the buffer plays on from the loop");
        s.p.lock().unwrap().skip();
        s.run(2_500);
        let h = heard_since(&s, 300);
        assert!(h.iter().all(|x| x.is_some_and(|x| x.0 == 1)), "{h:?}");
    }

    /// Skipping in the player while a loop plays (live on the phone, 2026-11-23):
    /// the person has moved on, so the loop stops and the new song is heard.
    #[test]
    fn a_skip_during_a_loop_goes_live() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(2_000);
        assert_eq!(s.core.mode_name(), "looping");
        s.p.lock().unwrap().skip();
        s.p.lock().unwrap().playing = true;
        s.run(2_500);
        assert_eq!(s.core.mode_name(), "live");
        let h = heard_since(&s, 300);
        assert!(h.iter().all(|x| x.is_some_and(|x| x.0 == 1)), "{h:?}");
    }

    /// ...but a song that plays to its end carries on from the buffer, every bit of
    /// it heard.
    #[test]
    fn a_song_that_ends_plays_to_its_end() {
        let mut s = Sim::new(vec![tune(0, 50, 120.0), tune(1, 120, 100.0)]);
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(20_000);
        s.cmd("live");
        let from = s.heard.len();
        s.run(25_000);
        let after: Vec<(usize, i64)> = s.heard[from..].iter().filter_map(|h| h.1).collect();
        let first_b = after.iter().position(|x| x.0 == 1).expect("the next song");
        let last_a = after[..first_b].iter().map(|x| x.1).max().unwrap();
        assert!(last_a >= 49_800, "song 0 cut off at {last_a}");
        // and no going back: one run up the song, then the next
        assert!(after[..first_b].windows(2).all(|w| w[1].1 >= w[0].1 - 30));
    }

    /// Extending a loop by a bar into the phrase already buffered after it needs no
    /// seek (it once played the whole loop again).
    #[test]
    fn extending_into_the_next_phrase_plays_nothing_again() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("8");
        s.run(30_000);
        assert!(!s.player().2, "paused after the next phrase");
        let seeks = s.p.lock().unwrap().seeks.len();
        s.cmd("move 0 4");
        s.run(1_000);
        let st = s.core.state();
        assert_eq!(st.beats, 36, "{st:?}");
        assert_eq!(st.status, "");
        assert_eq!(
            s.p.lock().unwrap().seeks.len(),
            seeks,
            "a seek for what's buffered"
        );
        // and again: still buffered
        s.cmd("move 0 4");
        s.run(1_000);
        assert_eq!(s.core.state().beats, 40);
        assert_eq!(s.p.lock().unwrap().seeks.len(), seeks);
    }

    /// Looping 8 bars just after a song began: only what's of this song, and it
    /// says so (it once took a beat of the song before).
    #[test]
    fn a_loop_at_a_songs_start_stays_in_the_song() {
        let mut s = two_songs();
        s.p.lock().unwrap().ms = 105_000.0;
        s.play();
        s.run(25_000); // song 1, 10 s in
        assert_eq!(s.player().0, 1);
        s.cmd("8");
        assert_eq!(s.core.mode_name(), "looping", "{}", s.status());
        assert!(
            s.status().starts_with("Only"),
            "{} {:?}",
            s.status(),
            s.core.state()
        );
        s.run(10_000);
        let h = heard_since(&s, 9_000);
        assert!(h.iter().all(|x| x.is_some_and(|x| x.0 == 1)), "{h:?}");
        assert!(s.core.state().loop_from >= 0);
    }

    /// Played from the player's own controls while a loop holds it paused: the
    /// player is what's wanted.
    #[test]
    fn playing_the_player_leaves_the_loop() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(20_000);
        s.play(); // the person, in the player
        s.run(2_500);
        assert_eq!(s.core.mode_name(), "live");
        let (cur, ms, _) = s.player();
        let h = s.now_heard().unwrap();
        assert_eq!(h.0, cur);
        assert!((h.1 - ms).abs() < 200, "heard {h:?}, the player at {ms}");
    }

    /// Paused or seeked in the player while the buffer plays on after a loop: heard
    /// at once.
    #[test]
    fn pausing_or_seeking_the_player_after_live_is_heard_at_once() {
        for seek in [false, true] {
            let mut s = two_songs();
            s.play();
            s.run(40_000);
            s.cmd("4");
            s.run(20_000);
            s.cmd("live");
            s.run(3_000);
            if seek {
                s.p.lock().unwrap().ms = 90_000.0;
            } else {
                s.p.lock().unwrap().playing = false;
            }
            s.run(2_500);
            let h = heard_since(&s, 300);
            if seek {
                assert!(h.iter().all(|x| x.is_some_and(|x| x.1 > 90_000)), "{h:?}");
            } else {
                assert!(h.iter().all(|x| x.is_none()), "{h:?}");
            }
        }
    }

    /// Extending past what's buffered plays the song there (unheard: the loop plays
    /// on meanwhile), then the loop is the longer one, all of it one stretch.
    #[test]
    fn extending_past_the_buffer_plays_it_unheard() {
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(15_000);
        let was = s.core.state();
        s.cmd("move 0 32");
        assert!(s.status().starts_with("Listening to"), "{}", s.status());
        let from = s.heard.len();
        s.run(45_000);
        let st = s.core.state();
        assert_eq!(st.beats, 48, "{st:?}");
        assert_eq!(st.status, "");
        let (lo, hi) = (was.loop_from, was.loop_from + 24_000);
        assert!(
            (st.loop_from - lo).abs() < 60 && (st.loop_to - hi).abs() < 60,
            "{st:?}"
        );
        // never the player's music meanwhile: only the loop, old or new
        let h: Vec<i64> = s.heard[from..]
            .iter()
            .filter_map(|h| h.1.map(|x| x.1))
            .collect();
        assert_eq!(h.len(), s.heard.len() - from, "silence heard");
        assert!(
            h.iter().all(|&ms| (lo - 100..hi + 100).contains(&ms)),
            "{:?}",
            h.iter().max()
        );
    }

    /// A loop left playing a while, then extended (on the Mac, 2026-11-23): the
    /// paused player's silence had pushed the loop out of the newest minute of the
    /// buffer, so finding the beats there again laid the loop on silence, out of
    /// the song map ("Can't tell where in the song this loop is", and Save failed).
    /// The loop must play on unchanged while the player fetches the new part.
    #[test]
    fn extending_after_a_long_loop_keeps_it_playing() {
        let mut s = two_songs();
        s.p.lock().unwrap().ms = 60_000.0;
        s.play();
        s.run(12_000);
        s.cmd("1");
        s.run(60_000);
        // a little longer: the player fetches the next phrase, far along the buffer
        s.cmd("move 0 2");
        s.run(15_000);
        let was = s.core.state();
        assert!(was.loop_from > 0, "{was:?}");
        // back past where the buffer began: the player plays it again
        s.cmd("move -16 0");
        assert!(s.status().starts_with("Listening to"), "{}", s.status());
        let from = s.heard.len();
        s.run(30_000);
        let st = s.core.state();
        assert_eq!(st.status, "", "{st:?}");
        assert_eq!(st.beats, 22, "{st:?}");
        assert!(
            (st.loop_from - (was.loop_from - 8_000)).abs() < 60
                && (st.loop_to - was.loop_to).abs() < 60,
            "{was:?} {st:?}"
        );
        // only the loop, old or new, never silence
        let h: Vec<i64> = s.heard[from..]
            .iter()
            .filter_map(|h| h.1.map(|x| x.1))
            .collect();
        assert_eq!(h.len(), s.heard.len() - from, "silence heard");
        assert!(h
            .iter()
            .all(|&ms| (st.loop_from - 100..st.loop_to + 100).contains(&ms)));
    }

    /// Minutes of loop: the paused player's silence fills the whole buffer, and a
    /// copy of its newest minute must not crash (it underflowed past 150 s).
    #[test]
    fn a_loop_left_playing_for_minutes_can_still_be_extended() {
        let mut s = Sim::new(vec![tune(0, 400, 120.0)]);
        s.p.lock().unwrap().ms = 60_000.0;
        s.play();
        s.run(12_000);
        s.cmd("4");
        s.run(200_000);
        s.cmd("move -16 0");
        s.run(30_000);
        let st = s.core.state();
        assert_eq!(st.status, "", "{st:?}");
        assert_eq!(st.beats, 32, "{st:?}");
        assert!(st.loop_from > 0, "{st:?}");
    }

    /// Saved, then another song, then recalled: the player goes back there and the
    /// same loop plays.
    #[test]
    fn a_bookmark_comes_back() {
        let dir = std::env::temp_dir().join(format!("stemflow-sim-{}", std::process::id()));
        std::env::set_var("XDG_STATE_HOME", &dir);
        let mut s = two_songs();
        s.play();
        s.run(40_000);
        s.cmd("4");
        s.run(5_000);
        s.cmd("save Groove");
        let st = s.core.state();
        assert_eq!(st.status, "");
        let b = crate::bookmarks::all()
            .into_iter()
            .find(|b| b.name == "Groove")
            .unwrap();
        assert!((b.start_ms - st.loop_from).abs() < 50);
        s.cmd("live");
        s.p.lock().unwrap().skip();
        s.run(10_000);
        s.cmd(&format!("recall {}", b.created));
        s.run(30_000);
        let st = s.core.state();
        assert!(st.playing && st.status.is_empty(), "{st:?}");
        assert_eq!(s.player().0, 0, "back to the song");
        assert!(
            (st.loop_from - b.start_ms).abs() < 60 && (st.loop_to - b.end_ms).abs() < 60,
            "{st:?} {b:?}"
        );
        let h = heard_since(&s, 8_000);
        assert!(h.iter().all(
            |x| x.is_some_and(|x| x.0 == 0 && (b.start_ms - 60..b.end_ms + 60).contains(&x.1))
        ));
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Random sessions: what a person might do, in any order, with the rules checked
/// after every 50 ms.
#[cfg(test)]
mod random {
    use super::*;
    use proptest::prelude::*;

    #[derive(Clone, Debug)]
    pub enum Act {
        Loop(usize),
        Live,
        Top,
        Move(i64, i64),
        Repair,
        PlayPause,
        Wait(i64),
        /// In the player itself:
        Skip,
        Seek(i64),
        Pause,
        Play,
    }

    fn act() -> impl Strategy<Value = Act> {
        prop_oneof![
            3 => prop::sample::select(vec![1usize, 2, 4, 8]).prop_map(Act::Loop),
            2 => Just(Act::Live),
            1 => Just(Act::Top),
            3 => (-8i64..=4, -4i64..=16).prop_map(|(a, b)| Act::Move(a, b)),
            1 => Just(Act::Repair),
            1 => Just(Act::PlayPause),
            4 => (200i64..12_000).prop_map(Act::Wait),
            // a loop left playing a while: the paused player's silence fills the buffer
            1 => (60_000i64..200_000).prop_map(Act::Wait),
            1 => Just(Act::Skip),
            1 => (0i64..100_000).prop_map(Act::Seek),
            1 => Just(Act::Pause),
            1 => Just(Act::Play),
        ]
    }

    struct Rules {
        /// The person skipped to another song while a loop played: the loop stops.
        skipped_loop: Option<i64>,
        /// The person did something in the player at this time, while live: from a
        /// moment later, what's heard is the player.
        synced_after: Option<i64>,
        /// Since when a loop has left the player playing with nothing to wait for.
        idle_playing_since: Option<i64>,
        /// The loop playing now has had its place in the song.
        mapped: bool,
    }

    impl Rules {
        fn check(&mut self, s: &Sim, stepped: bool) -> Result<(), TestCaseError> {
            let mode = s.core.mode_name();
            let st = s.core.state();
            prop_assert_eq!(
                mode == "looping",
                s.looper.playing(),
                "the mode says looping exactly when a loop plays"
            );
            let heard = s.now_heard();
            if mode == "looping" && stepped {
                // a loop is music: never a pause in it, never before the song
                prop_assert!(heard.is_some(), "silence in a loop at {}", s.t);
                if st.loop_from >= 0 {
                    prop_assert!(st.loop_from >= -300, "a loop before the song: {:?}", st);
                }
            }
            // a loop the song map placed stays placed while it plays (after a long
            // loop, extending it laid it on silence: Mac, 2026-11-23)
            if mode != "looping" {
                self.mapped = false;
            } else if st.loop_from != -1 {
                self.mapped = true;
            } else if self.mapped && !s.core.waiting() {
                prop_assert!(false, "the loop left the song map at {}: {:?}", s.t, st);
            }
            let (cur, ms, playing) = s.player();
            // a loop never leaves the player playing unheard with nothing to wait for
            if mode == "looping" && playing && !s.core.waiting() {
                let since = *self.idle_playing_since.get_or_insert(s.t);
                // (the person may have played it: that's seen once the player has had
                // time to do what we last asked, and is asked again)
                prop_assert!(
                    s.t - since <= super::super::core::OURS_MS + s.poll_ms + 100,
                    "the player plays unheard under an idle loop since {since} (now {})",
                    s.t
                );
            } else {
                self.idle_playing_since = None;
            }
            if let Some(t0) = self.skipped_loop {
                if mode != "looping" {
                    self.skipped_loop = None;
                } else {
                    prop_assert!(
                        s.t <= t0 + s.poll_ms + 600,
                        "still looping at {} after a skip at {t0}",
                        s.t
                    );
                }
            }
            if mode != "live" {
                self.synced_after = None;
            }
            if let Some(t0) = self.synced_after.filter(|_| stepped) {
                if s.t >= t0 + s.poll_ms + 600 {
                    match heard {
                        None => {
                            prop_assert!(!playing, "silence while the player plays, at {}", s.t)
                        }
                        Some((i, h)) => {
                            prop_assert!(
                                playing,
                                "heard {i}/{h} with the player paused, at {}",
                                s.t
                            );
                            prop_assert_eq!(i, cur, "another song than the player's, at {}", s.t);
                            prop_assert!(
                                (h - ms).abs() <= 1_200,
                                "heard {} but the player is at {}, at {}",
                                h,
                                ms,
                                s.t
                            );
                        }
                    }
                }
            }
            Ok(())
        }
    }

    pub fn session(acts: Vec<Act>) -> Result<(), TestCaseError> {
        let mut s = Sim::new(vec![
            tune(0, 300, 120.0),
            tune(1, 300, 96.0),
            tune(2, 300, 140.0),
        ]);
        s.play();
        s.run(30_000);
        let mut rules = Rules {
            skipped_loop: None,
            synced_after: None,
            idle_playing_since: None,
            mapped: false,
        };
        let trace = std::env::var("SIM_TRACE").is_ok();
        for a in acts {
            if trace {
                let st = s.core.state();
                eprintln!(
                    "SIM t={} {:?} mode={} player={:?} heard={:?} loop={}..{} tail={} status={:?}",
                    s.t,
                    a,
                    s.core.mode_name(),
                    s.player(),
                    s.now_heard(),
                    st.loop_from,
                    st.loop_to,
                    st.tail,
                    st.status
                );
            }
            // what the person does in the player (that changes anything)
            let was_playing = s.player().2;
            let user = match a {
                Act::Skip | Act::Seek(_) => true,
                Act::Pause => was_playing,
                Act::Play => !was_playing,
                _ => false,
            };
            match a {
                Act::Loop(n) => s.cmd(&n.to_string()),
                Act::Live => s.cmd("live"),
                Act::Top => s.cmd("top"),
                Act::Move(a, b) => s.cmd(&format!("move {a} {b}")),
                Act::Repair => s.cmd("repair"),
                Act::PlayPause => s.cmd("playpause"),
                Act::Wait(_) => {}
                Act::Skip => s.p.lock().unwrap().skip(),
                Act::Seek(ms) => s.p.lock().unwrap().ms = ms as f64,
                Act::Pause => s.p.lock().unwrap().playing = false,
                Act::Play => s.p.lock().unwrap().playing = true,
            }
            if user && s.core.mode_name() == "live" {
                rules.synced_after = Some(s.t);
            }
            if matches!(a, Act::Skip) && s.core.mode_name() == "looping" {
                rules.skipped_loop = Some(s.t);
            }
            if matches!(a, Act::Live | Act::Top | Act::Loop(_)) {
                rules.synced_after = None; // our own doing: the buffer may play on
                rules.skipped_loop = None; // a loop asked for after the skip
            }
            if matches!(a, Act::Loop(_)) {
                rules.mapped = false; // another loop
            }
            rules.check(&s, false)?;
            let wait = if let Act::Wait(ms) = a { ms } else { 1_500 };
            for _ in 0..wait / STEP_MS {
                s.step();
                rules.check(&s, true)?;
            }
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: std::env::var("SIM_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(12),
            max_shrink_iters: 40,
            ..ProptestConfig::default()
        })]
        #[test]
        fn random_sessions(acts in prop::collection::vec(act(), 4..14)) {
            session(acts)?;
        }
    }
}
