//! Live mix control: one "balance" value, shared between the audio threads and the
//! D-Bus service (`org.stemflow.Control` on the session bus) that the GNOME
//! extension and hotkeys talk to.
//!
//! Balance runs from -1 to +1 on one line:
//!   -1 = bass solo, band fades in as it rises;  0 = the original mix;
//!   +1 = no bass (play-along), bass fades out as it rises.
//! "Bass" is really the target stem: with another one (`--target vocals` for karaoke,
//! `other` for the horns, ...) the same line runs from that stem solo to none of it. The D-Bus property names stay `Bass` and
//! `Band`; `Target` says which stem `Bass` means.
//! Each half is scaled by *perceived* loudness: 50% sounds half as loud (-10 dB),
//! so the useful quiet range isn't crammed into the bottom of the slider.
//! The target can also be boosted past full, up to `MAX_BASS` (200% = twice as loud,
//! +10 dB); the player's limiter keeps the louder mix from clipping.
//!
//! A model with a second stem (`bsrofo_sw_bass+vocals`) adds a third level of its own
//! (D-Bus `Second` names the stem, `SecondLevel` sets it, 0..1): bass, vocals and the
//! band (everything else) are then three plain levels, and "Band" no longer includes
//! the vocals. The one-line balance still moves bass against band only.

#[cfg(target_os = "linux")]
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Highest target level: 2.0 sounds twice as loud as the original (+10 dB).
pub const MAX_BASS: f32 = 2.0;

/// Perceived loudness (0.5 = half as loud, 2 = twice as loud) to amplitude: 10 dB per
/// doubling, i.e. gain = p^(log2(10)/2) ≈ p^1.66.
pub fn loudness_to_gain(p: f32) -> f32 {
    let p = p.clamp(0.0, MAX_BASS);
    if p == 0.0 {
        0.0
    } else {
        p.powf(std::f32::consts::LOG2_10 / 2.0)
    }
}

/// The truth is two perceived loudnesses (0..1): bass and band. The one-slider
/// "balance" is a view onto them; advanced mode sets each directly.
#[derive(Default)]
pub struct Control {
    /// Which stem "bass" refers to (one of `models::STEMS`). Changes when
    /// the menu switches to a model with another target.
    target: Mutex<&'static str>,
    /// The running model's id, the one the menu asked for (until the audio thread
    /// picks it up), and what happened to the last switch ("" when nothing to say).
    model: Mutex<String>,
    want: Mutex<Option<String>>,
    status: Mutex<String>,
    /// Bumped when the audio thread changes something, so the D-Bus side re-announces.
    generation: AtomicU64,
    bass: AtomicU32,
    band: AtomicU32,
    /// A two-stem model's second stem ("vocals") and its loudness (0..1).
    second: Mutex<Option<&'static str>>,
    second_level: AtomicU32,
    /// Latest model run time in microseconds, and how many runs overran the hop.
    pub compute_us: AtomicU64,
    pub overruns: AtomicU64,
    /// Beat click (src/beats.rs): on/off, beats per bar (0 = auto), feel (0.5 half
    /// time, 1, 2 double time), and what the tracker last found ("" = nothing yet).
    click: AtomicBool,
    meter: AtomicU32,
    feel: AtomicU32,
    beats: Mutex<String>,
    /// Looping, when there's a beat model for it.
    pub deck: std::sync::OnceLock<Arc<crate::loopdeck::Deck>>,
}

fn level(v: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, max)
    } else {
        1.0
    }
}

impl Control {
    pub fn new(target: &'static str, bass: f32, band: f32) -> Arc<Self> {
        let c = Control {
            target: Mutex::new(target),
            ..Default::default()
        };
        c.set_levels(bass, band);
        c.set_second_level(1.0);
        Arc::new(c)
    }
    /// Something the UIs show changed (from another thread): re-announce it.
    pub fn touch(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// The beat options as the looper takes them.
    pub fn beat_settings(&self) -> crate::beats::Settings {
        crate::beats::Settings {
            meter: self.meter() as usize,
            feel: self.feel(),
            ..Default::default()
        }
    }

    /// A `loop ...` command (`loopdeck::Command`). The meter and feel are the
    /// beat options (the click's too): set here, and the loop's bars follow.
    pub fn loop_command(&self, cmd: &str) -> anyhow::Result<()> {
        use crate::loopdeck::Command;
        let d = self
            .deck
            .get()
            .ok_or_else(|| anyhow::anyhow!("no looping: no beat model"))?;
        match cmd.parse::<Command>()? {
            Command::Meter(m) => self.set_beat_options(self.click(), m, self.feel()),
            Command::Feel(f) => self.set_beat_options(self.click(), self.meter(), f),
            _ => d.command(cmd, self.beat_settings())?,
        }
        self.touch();
        Ok(())
    }

    /// The loop panel (`loopdeck::panel`) as JSON, or `null` without looping.
    pub fn loop_panel(&self) -> String {
        self.deck.get().map_or_else(
            || "null".into(),
            |d| {
                crate::loopdeck::panel::build(
                    &d.state(),
                    self.meter(),
                    self.feel(),
                    &crate::bookmarks::all(),
                )
                .json()
            },
        )
    }

    /// The loop for drawing (`loopdeck::Deck::view_json`), or `null`.
    pub fn loop_view(&self, bins: usize) -> String {
        self.deck
            .get()
            .map_or_else(|| "null".into(), |d| d.view_json(bins))
    }

    /// The loop's state as JSON, or `null` without looping.
    pub fn loop_json(&self) -> String {
        self.deck
            .get()
            .map_or_else(|| "null".into(), |d| d.state().json())
    }

    pub fn target(&self) -> &'static str {
        *self.target.lock().unwrap()
    }
    pub fn second(&self) -> Option<&'static str> {
        *self.second.lock().unwrap()
    }
    pub fn set_second(&self, second: Option<&'static str>) {
        *self.second.lock().unwrap() = second;
    }
    pub fn second_level(&self) -> f32 {
        f32::from_bits(self.second_level.load(Ordering::Relaxed))
    }
    pub fn set_second_level(&self, v: f32) {
        self.second_level
            .store(level(v, 1.0).to_bits(), Ordering::Relaxed);
    }
    /// What the remembered levels are filed under: the target, or "bass+vocals".
    pub fn levels_key(&self) -> String {
        match self.second() {
            Some(s) => format!("{}+{s}", self.target()),
            None => self.target().to_string(),
        }
    }
    pub fn click(&self) -> bool {
        self.click.load(Ordering::Relaxed)
    }
    pub fn meter(&self) -> u32 {
        self.meter.load(Ordering::Relaxed)
    }
    pub fn feel(&self) -> f32 {
        match f32::from_bits(self.feel.load(Ordering::Relaxed)) {
            f if f == 0.5 || f == 2.0 => f,
            _ => 1.0,
        }
    }
    pub fn set_beat_options(&self, click: bool, meter: u32, feel: f32) {
        let old = (self.meter(), self.feel());
        self.click.store(click, Ordering::Relaxed);
        self.meter.store(meter.min(12), Ordering::Relaxed);
        let feel = if feel == 0.5 || feel == 2.0 {
            feel
        } else {
            1.0
        };
        self.feel.store(feel.to_bits(), Ordering::Relaxed);
        save_beat_options(self);
        // A loop playing takes the new bars too.
        if old != (self.meter(), self.feel()) {
            if let Some(d) = self.deck.get() {
                d.redecode(self.beat_settings());
            }
        }
    }
    pub fn beats(&self) -> String {
        self.beats.lock().unwrap().clone()
    }
    /// For the beat thread: tempo and meter found, announced over D-Bus when it changes.
    pub fn set_beats(&self, s: String) {
        let mut b = self.beats.lock().unwrap();
        if *b != s {
            *b = s;
            drop(b);
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn model(&self) -> String {
        self.model.lock().unwrap().clone()
    }
    pub fn set_model(&self, id: &str) {
        *self.model.lock().unwrap() = id.to_string();
    }
    pub fn status(&self) -> String {
        self.status.lock().unwrap().clone()
    }
    fn set_status(&self, s: String) {
        *self.status.lock().unwrap() = s;
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
    /// Ask the audio thread to switch to model `id` at its next hop.
    pub fn request_model(&self, id: &str, label: &str) {
        *self.want.lock().unwrap() = Some(id.to_string());
        self.set_status(format!("Loading {label}…"));
    }
    /// For the audio thread: a pending switch, if any.
    pub fn take_request(&self) -> Option<String> {
        self.want.lock().unwrap().take()
    }
    /// For the audio thread: the switch to `id` worked. A new target (or second
    /// stem) brings back the levels last used with it.
    pub fn switched(&self, id: &str, target: &'static str, second: Option<&'static str>) {
        self.set_model(id);
        save_model(id);
        let old = self.levels_key();
        *self.target.lock().unwrap() = target;
        self.set_second(second);
        let key = self.levels_key();
        if old != key {
            // a two-stem model the first time: start from the target's own levels
            match load_levels(&key).or_else(|| load_levels(target)) {
                Some((a, b, c)) => {
                    self.set_levels(a, b);
                    self.set_second_level(c.unwrap_or(1.0));
                }
                // first time with vocals, horns, ...: start with them out (play-along)
                None if target != "bass" => self.set_balance(1.0),
                None => self.set_second_level(1.0),
            }
        }
        self.set_status(String::new());
    }
    /// For the audio thread: the switch failed (and it went back to `self.model()`).
    pub fn switch_failed(&self, id: &str, err: &anyhow::Error) {
        self.set_status(format!("Couldn't load {id}: {err:#}"));
    }
    /// (bass, band) perceived loudness.
    pub fn levels(&self) -> (f32, f32) {
        (
            f32::from_bits(self.bass.load(Ordering::Relaxed)),
            f32::from_bits(self.band.load(Ordering::Relaxed)),
        )
    }
    pub fn set_levels(&self, bass: f32, band: f32) {
        self.bass
            .store(level(bass, MAX_BASS).to_bits(), Ordering::Relaxed);
        self.band
            .store(level(band, 1.0).to_bits(), Ordering::Relaxed);
    }
    /// (bass gain, band gain) as amplitudes.
    pub fn gains(&self) -> (f32, f32) {
        let (a, b) = self.levels();
        (loudness_to_gain(a), loudness_to_gain(b))
    }
    /// (bass, band, second stem) gains. Without a second stem the third is unused.
    pub fn gains3(&self) -> (f32, f32, f32) {
        let (a, b) = self.gains();
        (a, b, loudness_to_gain(self.second_level()))
    }
    /// Jump to a named mix ("solo", "full", "nobass", "novocals"); `None` if unknown.
    /// One stem: "nobass" and "novocals" both mean "remove the target", so existing
    /// hotkeys keep working in karaoke. Two stems: each names exactly what it says.
    pub fn snap(&self, name: &str) -> Option<()> {
        let boost = self.levels().0.max(1.0);
        if self.second().is_none() {
            let b = match name {
                "solo" => -1.0,
                "full" => 0.0,
                "nobass" | "remove" | "novocals" => 1.0,
                _ => return None,
            };
            self.set_balance(b);
            return Some(());
        }
        let (bass, band, second) = match name {
            "solo" => (boost, 0.0, 0.0),
            "full" => (boost, 1.0, 1.0),
            "nobass" => (0.0, 1.0, 1.0),
            "novocals" | "remove" => (boost, 1.0, 0.0),
            _ => return None,
        };
        self.set_levels(bass, band);
        self.set_second_level(second);
        Some(())
    }
    /// Balance in [-1, 1]. When both are below full (advanced mode), the quieter one
    /// decides which half of the slider it shows on.
    pub fn balance(&self) -> f32 {
        let (bass, band) = self.levels();
        if band <= bass {
            band - 1.0
        } else {
            1.0 - bass
        }
    }
    pub fn set_balance(&self, b: f32) {
        let b = if b.is_finite() {
            b.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        if b <= 0.0 {
            // keep any boost while the band fades
            self.set_levels(self.levels().0.max(1.0), 1.0 + b)
        } else {
            self.set_levels(1.0 - b, 1.0)
        }
    }
    pub fn describe(&self) -> String {
        let (bass, band) = self.levels();
        let pct = |p: f32| (p * 100.0).round() as i32;
        let t = self.target();
        let tc = format!("{}{}", t[..1].to_uppercase(), &t[1..]);
        let main = match (pct(bass), pct(band)) {
            (100, 0) => format!("{tc} solo"),
            (a, 0) if a > 100 => format!("{tc} solo · {a}%"),
            (100, 100) => "Full mix".into(),
            (0, 100) => format!("No {t}"),
            (100, b) => format!("Band {b}%"),
            (a, 100) => format!("{tc} {a}%"),
            (a, b) => format!("{tc} {a}% · Band {b}%"),
        };
        let Some(v) = self.second() else {
            return main;
        };
        // Three plain levels: name the solo, else whatever isn't at 100%.
        let cap = |s: &str| format!("{}{}", s[..1].to_uppercase(), &s[1..]);
        let parts = [
            (t.to_string(), pct(bass)),
            (v.to_string(), pct(self.second_level())),
            ("band".to_string(), pct(band)),
        ];
        let playing: Vec<_> = parts.iter().filter(|(_, p)| *p > 0).collect();
        if let [(name, p)] = playing.as_slice() {
            return match *p {
                100 => format!("{} solo", cap(name)),
                p => format!("{} solo · {p}%", cap(name)),
            };
        }
        let rest: Vec<String> = parts
            .iter()
            .filter(|(_, p)| *p != 100)
            .map(|(name, p)| match p {
                0 => format!("no {name}"),
                p => format!("{name} {p}%"),
            })
            .collect();
        if rest.is_empty() {
            "Full mix".into()
        } else {
            cap(&rest.join(" · "))
        }
    }
}

/// Where the last levels are remembered between runs, one file per target.
fn state_file(target: &str) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(base.join(match target {
        "bass" => "stemflow/levels".to_string(),
        t => format!("stemflow/levels-{t}"),
    }))
}

fn beats_file() -> Option<PathBuf> {
    Some(state_file("bass")?.with_file_name("beats"))
}

/// Click on/off, meter and feel, as "click meter feel" (e.g. "1 0 1").
pub fn load_beat_options(ctl: &Control) {
    let Some(s) = beats_file().and_then(|p| std::fs::read_to_string(p).ok()) else {
        return;
    };
    let v: Vec<&str> = s.split_whitespace().collect();
    if v.len() == 3 {
        ctl.click.store(v[0] == "1", Ordering::Relaxed);
        ctl.meter
            .store(v[1].parse().unwrap_or(0), Ordering::Relaxed);
        ctl.feel.store(
            v[2].parse::<f32>().unwrap_or(1.0).to_bits(),
            Ordering::Relaxed,
        );
    }
}

fn save_beat_options(ctl: &Control) {
    let Some(p) = beats_file() else { return };
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(
        p,
        format!("{} {} {}\n", ctl.click() as u8, ctl.meter(), ctl.feel()),
    );
}

fn model_file() -> Option<PathBuf> {
    Some(state_file("bass")?.with_file_name("model"))
}

/// The model id last switched to from the menu.
pub fn load_model() -> Option<String> {
    let s = std::fs::read_to_string(model_file()?).ok()?;
    Some(s.trim().to_string()).filter(|s| !s.is_empty())
}

fn save_model(id: &str) {
    if let Some(p) = model_file() {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, format!("{id}\n"));
    }
}

/// Last (target, band, second stem) loudness, if remembered. `key` is
/// `Control::levels_key`.
pub fn load_levels(key: &str) -> Option<(f32, f32, Option<f32>)> {
    let s = std::fs::read_to_string(state_file(key)?).ok()?;
    let mut it = s.split_whitespace().map(|v| v.parse::<f32>().ok());
    Some((it.next()??, it.next()??, it.next().flatten()))
}

pub fn save_levels(ctl: &Control) {
    let (a, b) = ctl.levels();
    if let Some(p) = state_file(&ctl.levels_key()) {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let line = match ctl.second() {
            Some(_) => format!("{a} {b} {}\n", ctl.second_level()),
            None => format!("{a} {b}\n"),
        };
        let _ = std::fs::write(p, line);
    }
}

#[cfg(target_os = "linux")]
struct Service {
    ctl: Arc<Control>,
    models: PathBuf,
    app: String,
    latency: f64,
    hop: f64,
}

#[cfg(target_os = "linux")]
impl Service {
    async fn changed(&self, em: &zbus::object_server::SignalEmitter<'_>) {
        save_levels(&self.ctl);
        self.announce(em).await;
    }
    async fn announce(&self, em: &zbus::object_server::SignalEmitter<'_>) {
        let _ = self.target_changed(em).await;
        let _ = self.model_changed(em).await;
        let _ = self.status_changed(em).await;
        let _ = self.balance_changed(em).await;
        let _ = self.bass_changed(em).await;
        let _ = self.band_changed(em).await;
        let _ = self.second_changed(em).await;
        let _ = self.second_level_changed(em).await;
        let _ = self.description_changed(em).await;
        let _ = self.click_changed(em).await;
        let _ = self.meter_changed(em).await;
        let _ = self.feel_changed(em).await;
        let _ = self.beats_changed(em).await;
        let _ = self.loop_state_changed(em).await;
    }
}

#[cfg(target_os = "linux")]
#[zbus::interface(name = "org.stemflow.Control")]
impl Service {
    /// One-slider view: -1 = bass solo, 0 = full mix, +1 = no bass.
    #[zbus(property)]
    fn balance(&self) -> f64 {
        self.ctl.balance() as f64
    }
    #[zbus(property)]
    async fn set_balance(
        &mut self,
        b: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        self.ctl.set_balance(b as f32);
        self.changed(&em).await;
    }
    /// Bass loudness, 0..2 (0.5 sounds half as loud, above 1 is a boost).
    #[zbus(property)]
    fn bass(&self) -> f64 {
        self.ctl.levels().0 as f64
    }
    #[zbus(property)]
    async fn set_bass(
        &mut self,
        v: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        let (_, band) = self.ctl.levels();
        self.ctl.set_levels(v as f32, band);
        self.changed(&em).await;
    }
    /// Band (everything but the bass) loudness, 0..1.
    #[zbus(property)]
    fn band(&self) -> f64 {
        self.ctl.levels().1 as f64
    }
    #[zbus(property)]
    async fn set_band(
        &mut self,
        v: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        let (bass, _) = self.ctl.levels();
        self.ctl.set_levels(bass, v as f32);
        self.changed(&em).await;
    }
    /// A two-stem model's second stem ("vocals"), or "" for a one-stem model.
    #[zbus(property)]
    fn second(&self) -> String {
        self.ctl.second().unwrap_or("").to_string()
    }
    /// The second stem's loudness, 0..1 (0 = taken out).
    #[zbus(property)]
    fn second_level(&self) -> f64 {
        self.ctl.second_level() as f64
    }
    #[zbus(property)]
    async fn set_second_level(
        &mut self,
        v: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        self.ctl.set_second_level(v as f32);
        self.changed(&em).await;
    }
    /// Human-readable state, e.g. "Band 15%".
    #[zbus(property)]
    fn description(&self) -> String {
        self.ctl.describe()
    }
    /// Which stem `Bass`, solo and remove refer to: "bass", "vocals" (karaoke), "other", ...
    #[zbus(property)]
    fn target(&self) -> String {
        self.ctl.target().to_string()
    }
    /// Every model in the models folder as (id, label), rescanned on each read so new
    /// exports show up without restarting anything.
    #[zbus(property(emits_changed_signal = "false"))]
    fn models(&self) -> Vec<(String, String)> {
        crate::models::scan(&self.models)
            .into_iter()
            .map(|m| (m.id, m.label))
            .collect()
    }
    /// The running model's id. Setting it switches models live (a few seconds of
    /// silence while the new one loads); `Status` says how it went.
    #[zbus(property)]
    fn model(&self) -> String {
        self.ctl.model()
    }
    #[zbus(property)]
    async fn set_model(&mut self, id: String) -> zbus::fdo::Result<()> {
        let m = crate::models::find(&self.models, &id)
            .ok_or_else(|| zbus::fdo::Error::InvalidArgs(format!("no model `{id}`")))?;
        if id != self.ctl.model() {
            self.ctl.request_model(&m.id, &m.label);
        }
        Ok(())
    }
    /// "Loading …" during a switch, an error if one failed, else empty.
    #[zbus(property)]
    fn status(&self) -> String {
        self.ctl.status()
    }
    #[zbus(property)]
    fn app(&self) -> String {
        self.app.clone()
    }
    /// Seconds from the source to the speakers (excluding device buffers).
    #[zbus(property)]
    fn latency(&self) -> f64 {
        self.latency
    }
    /// Seconds the last model run took; it must stay under `Hop` to keep up.
    #[zbus(property(emits_changed_signal = "false"))]
    fn compute(&self) -> f64 {
        self.ctl.compute_us.load(Ordering::Relaxed) as f64 / 1e6
    }
    #[zbus(property)]
    fn hop(&self) -> f64 {
        self.hop
    }
    #[zbus(property(emits_changed_signal = "false"))]
    fn overruns(&self) -> u64 {
        self.ctl.overruns.load(Ordering::Relaxed)
    }

    /// Beat click: a click on every beat mixed into the output, higher on bar lines.
    #[zbus(property)]
    fn click(&self) -> bool {
        self.ctl.click()
    }
    #[zbus(property)]
    async fn set_click(
        &mut self,
        on: bool,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        self.ctl
            .set_beat_options(on, self.ctl.meter(), self.ctl.feel());
        if !on {
            self.ctl.set_beats(String::new());
        }
        self.announce(&em).await;
    }
    /// Beats per bar for the bar lines; 0 = let the tracker choose (3-7).
    #[zbus(property)]
    fn meter(&self) -> u32 {
        self.ctl.meter()
    }
    #[zbus(property)]
    async fn set_meter(
        &mut self,
        m: u32,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        self.ctl
            .set_beat_options(self.ctl.click(), m, self.ctl.feel());
        self.announce(&em).await;
    }
    /// 1 = the beat as decoded, 0.5 = half time, 2 = double time.
    #[zbus(property)]
    fn feel(&self) -> f64 {
        self.ctl.feel() as f64
    }
    #[zbus(property)]
    async fn set_feel(
        &mut self,
        f: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) {
        self.ctl
            .set_beat_options(self.ctl.click(), self.ctl.meter(), f as f32);
        self.announce(&em).await;
    }
    /// What the tracker last found, e.g. "158 bpm · 4/bar"; "" while there's nothing.
    #[zbus(property)]
    fn beats(&self) -> String {
        self.ctl.beats()
    }

    /// The loop, as JSON (`loopdeck::State`), or `null` without looping. Changes are
    /// announced; the playhead isn't (read it while showing it).
    #[zbus(property)]
    fn loop_state(&self) -> String {
        self.ctl.loop_json()
    }

    /// A loop command: `4` (loop the last 4 bars, or resize), `live`, `again`,
    /// `move <da> <db>`, `shift <beats>`, `click on|off`, `speed <x>`, `pitch <st>`.
    async fn r#loop(
        &mut self,
        cmd: String,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) -> zbus::fdo::Result<()> {
        self.ctl
            .loop_command(&cmd)
            .map_err(|e| zbus::fdo::Error::InvalidArgs(format!("{e:#}")))?;
        let _ = self.loop_state_changed(&em).await;
        Ok(())
    }

    /// The loop for drawing (`loopdeck::Deck::view_json`) with `bins` peaks, or
    /// `null`. Read it while showing it; it isn't announced.
    fn loop_view(&self, bins: u32) -> String {
        self.ctl.loop_view(bins as usize)
    }

    /// The saved loops as a JSON array (`bookmarks::list_json`).
    fn loop_bookmarks(&self) -> String {
        crate::bookmarks::list_json()
    }

    /// The loop panel, as JSON (`loopdeck::panel`): what the extension shows. It
    /// changes when LoopState does.
    fn loop_panel(&self) -> String {
        self.ctl.loop_panel()
    }

    /// Move the one-slider balance by `step` (e.g. ±0.1); returns the new balance.
    async fn nudge(
        &mut self,
        step: f64,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) -> f64 {
        self.ctl.set_balance(self.ctl.balance() + step as f32);
        self.changed(&em).await;
        self.ctl.balance() as f64
    }

    /// Jump to "solo", "full", "nobass" or "novocals" (`Control::snap`); returns the
    /// new balance.
    async fn snap(
        &mut self,
        name: String,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) -> zbus::fdo::Result<f64> {
        self.ctl
            .snap(&name)
            .ok_or_else(|| zbus::fdo::Error::InvalidArgs(format!("unknown snap `{name}`")))?;
        self.changed(&em).await;
        Ok(self.ctl.balance() as f64)
    }
}

#[cfg(target_os = "linux")]
/// Claim `org.stemflow.Control` on the session bus. Keep the returned connection
/// alive for as long as the service should exist.
pub fn serve(
    ctl: Arc<Control>,
    models: PathBuf,
    app: &str,
    latency: f64,
    hop: f64,
) -> Result<zbus::blocking::Connection> {
    let svc = Service {
        ctl: ctl.clone(),
        models,
        app: app.to_string(),
        latency,
        hop,
    };
    let conn = zbus::blocking::connection::Builder::session()?
        .name("org.stemflow.Control")?
        .serve_at("/org/stemflow/Control", svc)?
        .build()?;
    // Re-announce whatever the audio thread changes (model switches, new target).
    let c = conn.clone();
    std::thread::spawn(move || {
        let mut seen = ctl.generation.load(Ordering::Relaxed);
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let g = ctl.generation.load(Ordering::Relaxed);
            if g == seen {
                continue;
            }
            seen = g;
            let Ok(iface) = c
                .object_server()
                .interface::<_, Service>("/org/stemflow/Control")
            else {
                return;
            };
            let svc = iface.get();
            zbus::block_on(svc.announce(iface.signal_emitter()));
        }
    });
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_scale() {
        let c = Control::new("bass", 1.0, 1.0);
        for (b, want) in [(-1.0, (1.0, 0.0)), (0.0, (1.0, 1.0)), (1.0, (0.0, 1.0))] {
            c.set_balance(b);
            assert_eq!(c.gains(), want);
            assert_eq!(c.balance(), b);
        }
        // half as loud = -10 dB
        c.set_balance(-0.5);
        let (_, band) = c.gains();
        assert!((20.0 * band.log10() + 10.0).abs() < 0.01, "{band}");
        c.set_balance(-0.85);
        assert_eq!(c.describe(), "Band 15%");
        c.set_balance(0.7);
        assert_eq!(c.describe(), "Bass 30%");
        c.set_levels(0.5, 0.2);
        assert_eq!(c.describe(), "Bass 50% · Band 20%");
        let k = Control::new("vocals", 1.0, 1.0);
        k.set_balance(1.0);
        assert_eq!(k.describe(), "No vocals");
        k.set_balance(-1.0);
        assert_eq!(k.describe(), "Vocals solo");
        // boost: twice as loud = +10 dB, kept while the band fades, capped at 200%
        c.set_levels(2.0, 1.0);
        assert!((20.0 * c.gains().0.log10() - 10.0).abs() < 0.01);
        assert_eq!((c.balance(), c.describe().as_str()), (0.0, "Bass 200%"));
        c.set_balance(-1.0);
        assert_eq!(c.describe(), "Bass solo · 200%");
        c.set_levels(5.0, 1.0);
        assert_eq!(c.levels().0, MAX_BASS);
    }

    #[test]
    fn second_stem() {
        let c = Control::new("bass", 1.0, 1.0);
        c.set_second(Some("vocals"));
        assert_eq!(c.levels_key(), "bass+vocals");
        assert_eq!(
            (c.gains3(), c.describe().as_str()),
            ((1.0, 1.0, 1.0), "Full mix")
        );
        for (snap, gains, text) in [
            ("novocals", (1.0, 1.0, 0.0), "No vocals"),
            ("solo", (1.0, 0.0, 0.0), "Bass solo"),
            ("nobass", (0.0, 1.0, 1.0), "No bass"),
            ("full", (1.0, 1.0, 1.0), "Full mix"),
        ] {
            c.snap(snap).unwrap();
            assert_eq!((c.gains3(), c.describe().as_str()), (gains, text), "{snap}");
        }
        // three independent levels
        c.set_levels(0.4, 1.0);
        c.set_second_level(0.0);
        assert_eq!(c.describe(), "Bass 40% · no vocals");
        c.set_levels(0.0, 0.0);
        c.set_second_level(1.0);
        assert_eq!(c.describe(), "Vocals solo");
        c.set_levels(2.0, 0.5);
        assert_eq!(c.describe(), "Bass 200% · band 50%");
        // one stem: "novocals" is the old remove-the-target hotkey
        let k = Control::new("vocals", 1.0, 1.0);
        k.snap("novocals").unwrap();
        assert_eq!(k.describe(), "No vocals");
    }
}
