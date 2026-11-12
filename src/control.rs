//! Live mix control: one "balance" value, shared between the audio threads and the
//! D-Bus service (`org.stemflow.Control` on the session bus) that the GNOME
//! extension and hotkeys talk to.
//!
//! Balance runs from -1 to +1 on one line:
//!   -1 = bass solo, band fades in as it rises;  0 = the original mix;
//!   +1 = no bass (play-along), bass fades out as it rises.
//! Each half is scaled by *perceived* loudness: 50% sounds half as loud (-10 dB),
//! so the useful quiet range isn't crammed into the bottom of the slider.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Perceived loudness (0..1, 0.5 = half as loud) to amplitude: -10 dB per halving,
/// i.e. gain = p^(log2(10)/2) ≈ p^1.66.
pub fn loudness_to_gain(p: f32) -> f32 {
    let p = p.clamp(0.0, 1.0);
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
    bass: AtomicU32,
    band: AtomicU32,
    /// Latest model run time in microseconds, and how many runs overran the hop.
    pub compute_us: AtomicU64,
    pub overruns: AtomicU64,
}

fn unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

impl Control {
    pub fn new(bass: f32, band: f32) -> Arc<Self> {
        let c = Control::default();
        c.set_levels(bass, band);
        Arc::new(c)
    }
    /// (bass, band) perceived loudness.
    pub fn levels(&self) -> (f32, f32) {
        (
            f32::from_bits(self.bass.load(Ordering::Relaxed)),
            f32::from_bits(self.band.load(Ordering::Relaxed)),
        )
    }
    pub fn set_levels(&self, bass: f32, band: f32) {
        self.bass.store(unit(bass).to_bits(), Ordering::Relaxed);
        self.band.store(unit(band).to_bits(), Ordering::Relaxed);
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
            self.set_levels(1.0, 1.0 + b)
        } else {
            self.set_levels(1.0 - b, 1.0)
        }
    }
    pub fn describe(&self) -> String {
        let (bass, band) = self.levels();
        let pct = |p: f32| (p * 100.0).round() as i32;
        match (pct(bass), pct(band)) {
            (100, 0) => "Bass solo".into(),
            (100, 100) => "Full mix".into(),
            (0, 100) => "No bass".into(),
            (100, b) => format!("Band {b}%"),
            (a, 100) => format!("Bass {a}%"),
            (a, b) => format!("Bass {a}% · Band {b}%"),
        }
    }
}

/// Where the last levels are remembered between runs.
fn state_file() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(base.join("stemflow/levels"))
}

/// Last (bass, band) loudness, if remembered.
pub fn load_levels() -> Option<(f32, f32)> {
    let s = std::fs::read_to_string(state_file()?).ok()?;
    let mut it = s.split_whitespace().map(|v| v.parse::<f32>().ok());
    Some((it.next()??, it.next()??))
}

pub fn save_levels(ctl: &Control) {
    let (a, b) = ctl.levels();
    if let Some(p) = state_file() {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, format!("{a} {b}\n"));
    }
}

/// Snap points the hotkeys and menu buttons use.
fn snap(name: &str) -> Option<f32> {
    match name {
        "solo" => Some(-1.0),
        "full" => Some(0.0),
        "nobass" => Some(1.0),
        _ => None,
    }
}

struct Service {
    ctl: Arc<Control>,
    app: String,
    latency: f64,
    hop: f64,
}

impl Service {
    async fn changed(&self, em: &zbus::object_server::SignalEmitter<'_>) {
        save_levels(&self.ctl);
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
    /// Bass loudness, 0..1 (0.5 sounds half as loud).
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
    app: &str,
    latency: f64,
    hop: f64,
) -> Result<zbus::blocking::Connection> {
    let svc = Service {
        ctl,
        app: app.to_string(),
        latency,
        hop,
    };
    let conn = zbus::blocking::connection::Builder::session()?
        .name("org.stemflow.Control")?
        .serve_at("/org/stemflow/Control", svc)?
        .build()?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_scale() {
        let c = Control::new(1.0, 1.0);
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
    }
}
