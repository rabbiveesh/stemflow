//! Live mix control: one "balance" value, shared between the audio threads and the
//! D-Bus service (`org.stemflow.Control` on the session bus) that the GNOME
//! extension and hotkeys talk to.
//!
//! Balance runs from -1 to +1 on one line:
//!   -1 = bass solo, band fades in as it rises;  0 = the original mix;
//!   +1 = no bass (play-along), bass fades out as it rises.
//! "Bass" is really the target stem: in karaoke mode (`--target vocals`) the same
//! line runs from vocals solo to no vocals. The D-Bus property names stay `Bass` and
//! `Band`; `Target` says which stem `Bass` means.
//! Each half is scaled by *perceived* loudness: 50% sounds half as loud (-10 dB),
//! so the useful quiet range isn't crammed into the bottom of the slider.
//! The target can also be boosted past full, up to `MAX_BASS` (200% = twice as loud,
//! +10 dB); the player's limiter keeps the louder mix from clipping.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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
    /// Which stem "bass" refers to: "bass", or "vocals" in karaoke mode. Changes when
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
    /// Latest model run time in microseconds, and how many runs overran the hop.
    pub compute_us: AtomicU64,
    pub overruns: AtomicU64,
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
        Arc::new(c)
    }
    pub fn target(&self) -> &'static str {
        *self.target.lock().unwrap()
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
    /// For the audio thread: the switch to `id` worked. A new target brings back the
    /// levels last used with it.
    pub fn switched(&self, id: &str, target: &'static str) {
        self.set_model(id);
        save_model(id);
        let old = std::mem::replace(&mut *self.target.lock().unwrap(), target);
        if old != target {
            match load_levels(target) {
                Some((a, b)) => self.set_levels(a, b),
                // first time in karaoke: start with the vocals out
                None if target == "vocals" => self.set_balance(1.0),
                None => {}
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
        match (pct(bass), pct(band)) {
            (100, 0) => format!("{tc} solo"),
            (a, 0) if a > 100 => format!("{tc} solo · {a}%"),
            (100, 100) => "Full mix".into(),
            (0, 100) => format!("No {t}"),
            (100, b) => format!("Band {b}%"),
            (a, 100) => format!("{tc} {a}%"),
            (a, b) => format!("{tc} {a}% · Band {b}%"),
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

/// Last (target, band) loudness, if remembered.
pub fn load_levels(target: &str) -> Option<(f32, f32)> {
    let s = std::fs::read_to_string(state_file(target)?).ok()?;
    let mut it = s.split_whitespace().map(|v| v.parse::<f32>().ok());
    Some((it.next()??, it.next()??))
}

pub fn save_levels(ctl: &Control) {
    let (a, b) = ctl.levels();
    if let Some(p) = state_file(ctl.target()) {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, format!("{a} {b}\n"));
    }
}

/// Snap points the hotkeys and menu buttons use. "nobass" means "remove the
/// target" whatever it is, so existing hotkeys keep working in karaoke mode.
fn snap(name: &str) -> Option<f32> {
    match name {
        "solo" => Some(-1.0),
        "full" => Some(0.0),
        "nobass" | "remove" | "novocals" => Some(1.0),
        _ => None,
    }
}

struct Service {
    ctl: Arc<Control>,
    models: PathBuf,
    app: String,
    latency: f64,
    hop: f64,
}

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
        let _ = self.description_changed(em).await;
    }
}

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
    /// Human-readable state, e.g. "Band 15%".
    #[zbus(property)]
    fn description(&self) -> String {
        self.ctl.describe()
    }
    /// Which stem `Bass`, solo and remove refer to: "bass" or "vocals" (karaoke).
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

    /// Jump to "solo", "full" or "nobass"; returns the new balance.
    async fn snap(
        &mut self,
        name: String,
        #[zbus(signal_emitter)] em: zbus::object_server::SignalEmitter<'_>,
    ) -> zbus::fdo::Result<f64> {
        let b = snap(&name)
            .ok_or_else(|| zbus::fdo::Error::InvalidArgs(format!("unknown snap `{name}`")))?;
        self.ctl.set_balance(b);
        self.changed(&em).await;
        Ok(b as f64)
    }
}

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
}
