//! Spotify Connect input: stemflow shows up as a speaker in Spotify's device list
//! (via librespot) and gets the decoded audio directly, instead of capturing an app.
//!
//! Connecting, in the order it's tried:
//! 1. Credentials cached by an earlier pairing (`~/.cache/stemflow/spotify`): the
//!    device logs straight in and is listed in every Spotify app on the account,
//!    on any network.
//! 2. Zeroconf, always on: any Spotify app on the same network lists "stemflow (host)"
//!    under Devices; picking it pairs (and caches) that account. Picking it from
//!    another account switches to that one.
//! 3. `--spotify-login`, or no zeroconf: log in once in the browser (OAuth).
//!
//! librespot needs Spotify Premium.
//!
//! The player decodes far faster than real time, so the sink blocks once it is a
//! little ahead, and a clock thread hands stemflow 20 ms blocks at the real rate,
//! filling silence while paused, exactly like a null sink's monitor would.

use anyhow::{Context, Result};
use librespot::connect::{ConnectConfig, Spirc};
use librespot::core::authentication::Credentials;
use librespot::core::cache::Cache;
use librespot::core::config::{DeviceType, SessionConfig};
use librespot::core::session::Session;
use librespot::discovery::Discovery;
use librespot::playback::audio_backend::{Sink, SinkResult};
use librespot::playback::config::PlayerConfig;
use librespot::playback::convert::Converter;
use librespot::playback::decoder::AudioPacket;
use librespot::playback::mixer::{self, MixerConfig};
use librespot::playback::player::{Player, PlayerEvent};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const RATE: usize = librespot::playback::SAMPLE_RATE as usize;
const _: () = assert!(RATE == crate::model::SAMPLE_RATE as usize);
/// How far the player may decode ahead of the clock.
const AHEAD: usize = RATE / 2;
const BLOCK: usize = RATE / 50; // 20 ms, as the capture path reads

#[derive(Clone, Debug)]
pub struct Opts {
    /// The name in Spotify's device list.
    pub name: String,
    /// Log in through the browser now instead of waiting for a pairing.
    pub login: bool,
}

/// Stereo interleaved samples between the player and the clock.
#[derive(Default)]
struct Feed {
    q: Mutex<VecDeque<f32>>,
    room: Condvar,
}

struct FeedSink {
    feed: Arc<Feed>,
    running: Arc<AtomicBool>,
}

impl Sink for FeedSink {
    fn write(&mut self, packet: AudioPacket, _: &mut Converter) -> SinkResult<()> {
        let AudioPacket::Samples(s) = packet else {
            return Ok(()); // passthrough decoder isn't enabled
        };
        let mut q = self.feed.q.lock().unwrap();
        while q.len() >= 2 * AHEAD && self.running.load(Ordering::Relaxed) {
            q = self
                .feed
                .room
                .wait_timeout(q, Duration::from_millis(100))
                .unwrap()
                .0;
        }
        q.extend(s.iter().map(|&v| v as f32));
        Ok(())
    }
}

/// Real-time clock: one 20 ms block per 20 ms, silence where the player has none.
fn clock(feed: Arc<Feed>, tx: mpsc::Sender<Vec<f32>>, running: Arc<AtomicBool>) {
    let period = Duration::from_secs_f64(BLOCK as f64 / RATE as f64);
    let mut next = Instant::now();
    while running.load(Ordering::Relaxed) {
        let mut v = Vec::with_capacity(2 * BLOCK);
        {
            let mut q = feed.q.lock().unwrap();
            let n = q.len().min(2 * BLOCK);
            v.extend(q.drain(..n));
        }
        feed.room.notify_all();
        v.resize(2 * BLOCK, 0.0);
        if tx.send(v).is_err() {
            break;
        }
        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else if now - next > Duration::from_millis(200) {
            next = now; // fell asleep (suspend, a stall): don't burst to catch up
        }
    }
}

fn cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("stemflow/spotify")
}

/// "stemflow (laptop)": every machine running stemflow is its own speaker, so the
/// name says which one.
pub fn default_name() -> String {
    match hostname() {
        Some(h) => format!("stemflow ({h})"),
        None => "stemflow".into(),
    }
}

fn hostname() -> Option<String> {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes into buf.
    if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let h = String::from_utf8_lossy(&buf[..end]);
    let h = h.split('.').next().unwrap_or_default().trim();
    (!h.is_empty() && h != "localhost").then(|| h.to_string())
}

/// librespot makes up a new device id every run, which Spotify sees as a new speaker
/// each time. Keep the first one.
fn device_id(dir: &std::path::Path) -> String {
    let path = dir.join("device_id");
    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim();
        if !id.is_empty() {
            return id.to_string();
        }
    }
    let id = SessionConfig::default().device_id;
    let _ = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, &id));
    id
}

/// Start the Connect device; its audio arrives on `tx` in 20 ms stereo blocks.
pub fn spawn(opts: Opts, tx: mpsc::Sender<Vec<f32>>, running: Arc<AtomicBool>) -> Result<()> {
    let feed = Arc::new(Feed::default());
    {
        let (f, r) = (feed.clone(), running.clone());
        std::thread::spawn(move || clock(f, tx, r));
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("starting the Spotify Connect runtime")?;
    std::thread::spawn(move || {
        if let Err(e) = rt.block_on(run(opts, feed, running)) {
            eprintln!("stemflow: Spotify Connect stopped: {e:#}");
        }
    });
    Ok(())
}

async fn run(opts: Opts, feed: Arc<Feed>, running: Arc<AtomicBool>) -> Result<()> {
    let dir = cache_dir();
    // Credentials and volume only: no audio cache.
    let cache = Cache::new(Some(&dir), Some(&dir), None, None)
        .with_context(|| format!("opening {}", dir.display()))?;
    let session_config = SessionConfig {
        device_id: device_id(&dir),
        ..SessionConfig::default()
    };

    let discovery = Discovery::builder(
        session_config.device_id.clone(),
        session_config.client_id.clone(),
    )
    .name(opts.name.clone())
    .device_type(DeviceType::Computer)
    .launch();
    let mut discovery = match discovery {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("stemflow: no zeroconf on this network ({e}); use --spotify-login");
            None
        }
    };

    let mut creds = cache.credentials();
    if opts.login || (creds.is_none() && discovery.is_none()) {
        creds = Some(oauth(&session_config.client_id)?);
    }
    match &creds {
        Some(_) => eprintln!(
            "stemflow: logging in to Spotify; pick \"{}\" under Devices in any Spotify app",
            opts.name
        ),
        None => eprintln!(
            "stemflow: open Spotify on a device on this network and pick \"{}\" under Devices (Premium needed)",
            opts.name
        ),
    }

    let mixer = mixer::find(None).context("no soft mixer")?(MixerConfig::default())?;
    let session = Session::new(session_config.clone(), Some(cache.clone()));
    let sink_feed = feed.clone();
    let sink_running = running.clone();
    let player = Player::new(
        PlayerConfig::default(),
        session.clone(),
        mixer.get_soft_volume(),
        move || {
            Box::new(FeedSink {
                feed: sink_feed.clone(),
                running: sink_running.clone(),
            }) as Box<dyn Sink>
        },
    );
    let mut events = player.get_player_event_channel();
    let connect_config = ConnectConfig {
        name: opts.name.clone(),
        device_type: DeviceType::Computer,
        // stemflow sets the levels; Spotify's slider still works on top
        initial_volume: u16::MAX,
        ..Default::default()
    };

    let mut session = session;
    let mut spirc: Option<Spirc> = None;
    let mut task: Option<std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>> = None;
    let mut connecting = creds.is_some();
    let mut restarts: Vec<Instant> = Vec::new();
    loop {
        tokio::select! {
            c = async { futures_next(discovery.as_mut()).await }, if discovery.is_some() => {
                match c {
                    Some(c) => {
                        eprintln!("stemflow: paired with a Spotify app");
                        creds = Some(c);
                        if let Some(s) = spirc.take() { let _ = s.shutdown(); }
                        if let Some(t) = task.take() { tokio::spawn(t); }
                        if !session.is_invalid() { session.shutdown(); }
                        connecting = true;
                    }
                    None => {
                        eprintln!("stemflow: zeroconf stopped; cached login still works");
                        discovery = None;
                    }
                }
            }
            _ = async {}, if connecting && creds.is_some() => {
                connecting = false;
                if session.is_invalid() {
                    session = Session::new(session_config.clone(), Some(cache.clone()));
                    player.set_session(session.clone());
                }
                match Spirc::new(connect_config.clone(), session.clone(), creds.clone().unwrap(), player.clone(), mixer.clone()).await {
                    Ok((s, t)) => {
                        eprintln!("stemflow: on Spotify as {} (\"{}\")", session.username(), opts.name);
                        spirc = Some(s);
                        task = Some(Box::pin(t));
                    }
                    Err(e) => eprintln!("stemflow: Spotify login failed: {e}"),
                }
            }
            _ = async { task.as_mut().unwrap().await }, if task.is_some() && !connecting => {
                task = None;
                spirc = None;
                restarts.retain(|t| t.elapsed() < Duration::from_secs(600));
                if restarts.len() >= 5 {
                    anyhow::bail!("the Spotify connection keeps dropping");
                }
                restarts.push(Instant::now());
                eprintln!("stemflow: Spotify connection dropped; reconnecting");
                if !session.is_invalid() { session.shutdown(); }
                connecting = true;
            }
            ev = events.recv() => match ev {
                Some(PlayerEvent::Unavailable { track_id, .. }) => {
                    eprintln!("stemflow: Spotify couldn't play {track_id} (skipped)");
                }
                Some(_) => {}
                None => anyhow::bail!("the Spotify player stopped"),
            },
        }
        if !running.load(Ordering::Relaxed) {
            break;
        }
    }
    if let Some(s) = spirc {
        let _ = s.shutdown();
    }
    Ok(())
}

async fn futures_next(d: Option<&mut Discovery>) -> Option<Credentials> {
    use futures_util::StreamExt;
    d?.next().await
}

fn oauth(client_id: &str) -> Result<Credentials> {
    eprintln!("stemflow: opening the browser to log in to Spotify");
    let token = librespot::oauth::OAuthClientBuilder::new(
        client_id,
        "http://127.0.0.1:5588/login",
        vec!["streaming"],
    )
    .open_in_browser()
    .build()?
    .get_access_token()?;
    Ok(Credentials::with_access_token(token.access_token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_paces_and_fills_silence() {
        let feed = Arc::new(Feed::default());
        feed.q
            .lock()
            .unwrap()
            .extend(std::iter::repeat_n(0.5f32, 2 * BLOCK + 10));
        let (tx, rx) = mpsc::channel();
        let running = Arc::new(AtomicBool::new(true));
        let (f, r) = (feed.clone(), running.clone());
        let t0 = Instant::now();
        let h = std::thread::spawn(move || clock(f, tx, r));
        let blocks: Vec<Vec<f32>> = rx.iter().take(10).collect();
        let dt = t0.elapsed();
        running.store(false, Ordering::Relaxed);
        drop(rx);
        h.join().unwrap();
        assert!(blocks.iter().all(|b| b.len() == 2 * BLOCK));
        assert!(blocks[0].iter().all(|&v| v == 0.5));
        assert_eq!(blocks[1][9], 0.5);
        assert_eq!(blocks[1][10], 0.0);
        assert!(blocks[2].iter().all(|&v| v == 0.0));
        // 10 blocks: the first goes out at once, then 20 ms apiece
        assert!(dt >= Duration::from_millis(170), "{dt:?}");
    }
}
