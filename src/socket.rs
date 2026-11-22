//! Live control over a Unix socket: the same knobs as the D-Bus service, for
//! platforms without a session bus (the macOS menu bar app) and for scripts.
//!
//! One request per line, one JSON reply per line, always the whole state:
//!
//! ```text
//! state                      -> {"ok":true,"balance":-1,...}
//! set balance -0.5           (also bass, band, second_level, model, click, meter, feel)
//! nudge 0.1
//! snap solo|full|nobass|novocals
//! quit                       stop stemflow (as Ctrl-C would)
//! ```
//!
//! `echo state | nc -U <socket>` works for a quick look. The socket is
//! `$STEMFLOW_SOCKET`, else `$XDG_RUNTIME_DIR/stemflow.sock` (Linux), else
//! `~/Library/Application Support/stemflow/control.sock` (macOS) or
//! `~/.local/state/stemflow/control.sock`.

use crate::control::{save_levels, Control};
use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub fn path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("STEMFLOW_SOCKET") {
        return Some(p.into());
    }
    if cfg!(target_os = "linux") {
        if let Some(d) = std::env::var_os("XDG_RUNTIME_DIR") {
            return Some(PathBuf::from(d).join("stemflow.sock"));
        }
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    Some(if cfg!(target_os = "macos") {
        home.join("Library/Application Support/stemflow/control.sock")
    } else {
        home.join(".local/state/stemflow/control.sock")
    })
}

/// What the requests act on, besides the shared `Control`.
pub struct Info {
    pub models: PathBuf,
    pub app: String,
    pub latency: f64,
    pub hop: f64,
}

/// Listen on the control socket from a background thread.
pub fn serve(ctl: Arc<Control>, info: Info) -> Result<PathBuf> {
    let path = path().context("no HOME")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if UnixStream::connect(&path).is_ok() {
        bail!(
            "another stemflow is already listening on {}",
            path.display()
        );
    }
    let _ = std::fs::remove_file(&path); // left over from a run that was killed
    let listener =
        UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;
    let info = Arc::new(info);
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (ctl, info) = (ctl.clone(), info.clone());
            std::thread::spawn(move || {
                let _ = handle(conn, &ctl, &info);
            });
        }
    });
    Ok(path)
}

fn handle(conn: UnixStream, ctl: &Control, info: &Info) -> Result<()> {
    let mut out = conn.try_clone()?;
    for line in BufReader::new(conn).lines() {
        let line = line?;
        let reply = match request(line.trim(), ctl, info) {
            Ok(()) => state(ctl, info, None),
            Err(e) => state(ctl, info, Some(&format!("{e:#}"))),
        };
        out.write_all(reply.as_bytes())?;
        out.write_all(b"\n")?;
    }
    Ok(())
}

fn num(v: Option<&str>) -> Result<f32> {
    v.context("missing value")?.parse().context("not a number")
}

fn request(line: &str, ctl: &Control, info: &Info) -> Result<()> {
    let mut words = line.splitn(3, ' ');
    let cmd = words.next().unwrap_or("");
    let (a, b) = (words.next(), words.next());
    match cmd {
        "" | "state" => return Ok(()),
        "nudge" => ctl.set_balance(ctl.balance() + num(a)?),
        "snap" => {
            let name = a.context("snap what?")?;
            ctl.snap(name)
                .with_context(|| format!("unknown snap `{name}`"))?;
        }
        "quit" => {
            // The Ctrl-C handler gives the app its sound back before exiting.
            unsafe { libc::kill(libc::getpid(), libc::SIGINT) };
            return Ok(());
        }
        "set" => match a.context("set what?")? {
            "balance" => ctl.set_balance(num(b)?),
            "bass" => ctl.set_levels(num(b)?, ctl.levels().1),
            "band" => ctl.set_levels(ctl.levels().0, num(b)?),
            "second_level" => ctl.set_second_level(num(b)?),
            "model" => {
                let id = b.context("which model?")?;
                let m = crate::models::find(&info.models, id)
                    .with_context(|| format!("no model `{id}`"))?;
                if m.id != ctl.model() {
                    ctl.request_model(&m.id, &m.label);
                }
                return Ok(());
            }
            "click" => {
                let on = matches!(b, Some("1" | "true" | "on"));
                ctl.set_beat_options(on, ctl.meter(), ctl.feel());
                if !on {
                    ctl.set_beats(String::new());
                }
                return Ok(());
            }
            "meter" => {
                ctl.set_beat_options(ctl.click(), num(b)? as u32, ctl.feel());
                return Ok(());
            }
            "feel" => {
                ctl.set_beat_options(ctl.click(), ctl.meter(), num(b)?);
                return Ok(());
            }
            k => bail!("unknown setting `{k}`"),
        },
        c => bail!("unknown request `{c}`"),
    }
    save_levels(ctl);
    Ok(())
}

fn js(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn state(ctl: &Control, info: &Info, err: Option<&str>) -> String {
    let (bass, band) = ctl.levels();
    let models: Vec<String> = crate::models::scan(&info.models)
        .into_iter()
        .map(|m| format!("[{},{}]", js(&m.id), js(&m.label)))
        .collect();
    let mut f = vec![
        format!("\"ok\":{}", err.is_none()),
        format!("\"balance\":{}", ctl.balance()),
        format!("\"bass\":{bass}"),
        format!("\"band\":{band}"),
        format!("\"second\":{}", js(ctl.second().unwrap_or(""))),
        format!("\"second_level\":{}", ctl.second_level()),
        format!("\"description\":{}", js(&ctl.describe())),
        format!("\"target\":{}", js(ctl.target())),
        format!("\"model\":{}", js(&ctl.model())),
        format!("\"status\":{}", js(&ctl.status())),
        format!("\"app\":{}", js(&info.app)),
        format!("\"latency\":{}", info.latency),
        format!("\"hop\":{}", info.hop),
        format!(
            "\"compute\":{}",
            ctl.compute_us.load(Ordering::Relaxed) as f64 / 1e6
        ),
        format!("\"overruns\":{}", ctl.overruns.load(Ordering::Relaxed)),
        format!("\"click\":{}", ctl.click()),
        format!("\"meter\":{}", ctl.meter()),
        format!("\"feel\":{}", ctl.feel()),
        format!("\"beats\":{}", js(&ctl.beats())),
        format!("\"models\":[{}]", models.join(",")),
    ];
    if let Some(e) = err {
        f.push(format!("\"error\":{}", js(e)));
    }
    format!("{{{}}}", f.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests() {
        let ctl = Control::new("bass", 1.0, 1.0);
        let info = Info {
            models: PathBuf::from("/nonexistent"),
            app: "spotify".into(),
            latency: 2.8,
            hop: 1.0,
        };
        // Levels are saved on change; keep that out of the real state dir.
        std::env::set_var("XDG_STATE_HOME", std::env::temp_dir().join("stemflow-test"));
        request("snap solo", &ctl, &info).unwrap();
        assert_eq!(ctl.balance(), -1.0);
        request("set balance 0.5", &ctl, &info).unwrap();
        request("nudge -0.25", &ctl, &info).unwrap();
        assert!((ctl.balance() - 0.25).abs() < 1e-6);
        assert!(request("snap sideways", &ctl, &info).is_err());
        assert!(request("set model nope", &ctl, &info).is_err());
        let s = state(&ctl, &info, Some("a \"quoted\" error"));
        assert!(s.starts_with("{\"ok\":false,\"balance\":0.25,"), "{s}");
        assert!(
            s.ends_with(",\"models\":[],\"error\":\"a \\\"quoted\\\" error\"}"),
            "{s}"
        );
    }
}
