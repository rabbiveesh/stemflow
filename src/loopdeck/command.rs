//! What a UI can ask of the loop, as one type: the control socket's and D-Bus's
//! `loop ...` text parses into it, and every platform's loop panel is checked against
//! its list of verbs (`Verb`), so none can quietly leave a feature out.

use anyhow::{bail, Context, Result};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, strum::EnumDiscriminants)]
#[strum_discriminants(name(Verb), derive(strum::EnumIter, strum::IntoStaticStr, Hash))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
pub enum Command {
    /// Loop the last N bars heard, or make the loop playing N bars long.
    Bars(usize),
    /// Back to the live stream.
    Live,
    /// The loop again from its top.
    Top,
    /// Move the start by `start` beats and the end by `end` (extend, trim, move).
    Move {
        start: i64,
        end: i64,
    },
    /// What the panel's ◀ ▶ step by.
    Unit(Unit),
    /// Set the ends as beat indices (the loop view's drag).
    Edges {
        a: usize,
        b: usize,
    },
    /// Bar 1 is on this beat index.
    Bar1(usize),
    Click(bool),
    /// Beats per bar (0: find out), and the feel (½: half time, 2: double): the
    /// beat options, which read the loop's bars again.
    Meter(u32),
    Feel(f32),
    Speed(f32),
    Pitch(f32),
    KeepPitch(bool),
    /// Play the loop's part of the song again and lay the loop on that.
    Repair,
    Save(String),
    Recall(i64),
    Rename {
        created: i64,
        name: String,
    },
    Delete(i64),
    /// Play or pause the player.
    #[strum_discriminants(strum(serialize = "playpause"))]
    PlayPause,
    /// Play the plain mix while live; separate only what is looped.
    StemsOnly(bool),
}

/// What the panel's ◀ ▶ step by: a beat, a bar, or the loop's own length.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    strum::EnumIter,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[strum(serialize_all = "snake_case")]
pub enum Unit {
    #[default]
    Beat,
    Bar,
    Phrase,
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.into())
    }
}

impl Verb {
    /// The word that starts the command's text.
    pub fn word(self) -> &'static str {
        self.into()
    }
}

fn on(v: Option<&str>) -> Result<bool> {
    match v {
        Some("1" | "true" | "on") => Ok(true),
        Some("0" | "false" | "off") => Ok(false),
        Some(x) => bail!("on or off, not `{x}`"),
        None => bail!("on or off?"),
    }
}

impl FromStr for Command {
    type Err = anyhow::Error;

    /// `<bars>`, `live`, `top` (or `again`), `move <start> <end>`, `unit beat|bar|phrase`,
    /// `edges <a> <b>`, `bar1 <beat>`, `click on|off`, `meter <beats per bar>`, `feel <x>`,
    /// `speed <x>`, `pitch <semitones>`, `keep_pitch on|off`, `repair`,
    /// `save <name>`, `recall <created>`, `rename <created> <name>`,
    /// `delete <created>`, `playpause`, `stems_only on|off`.
    fn from_str(cmd: &str) -> Result<Command> {
        let cmd = cmd.trim();
        let mut w = cmd.split_whitespace();
        let verb = w.next().context("loop what?")?;
        let mut num = |what: &str| -> Result<String> {
            w.next()
                .map(str::to_string)
                .with_context(|| format!("missing {what}"))
        };
        let int = |s: String| -> Result<i64> { s.parse().context("not a number") };
        let float = |s: String| -> Result<f32> {
            let v: f32 = s.parse().context("not a number")?;
            if !v.is_finite() {
                bail!("not a number");
            }
            Ok(v)
        };
        // the rest of the line, after `skip` words
        let rest = |skip: usize| {
            cmd.splitn(skip + 1, char::is_whitespace)
                .nth(skip)
                .unwrap_or("")
                .trim()
                .to_string()
        };
        Ok(match verb {
            "live" => Command::Live,
            "top" | "again" => Command::Top,
            "move" => {
                let start = int(num("start")?)?;
                Command::Move {
                    start,
                    end: int(num("end")?)?,
                }
            }
            "unit" => Command::Unit(num("unit")?.parse()?),
            "edges" => {
                let (a, b) = (int(num("start")?)?, int(num("end")?)?);
                if a < 0 || b <= a {
                    bail!("the start has to come before the end");
                }
                Command::Edges {
                    a: a as usize,
                    b: b as usize,
                }
            }
            "bar1" => Command::Bar1(int(num("beat")?)?.max(0) as usize),
            "click" => Command::Click(on(w.next())?),
            "meter" => Command::Meter(int(num("meter")?)?.clamp(0, 12) as u32),
            "feel" => {
                let f = float(num("feel")?)?;
                if ![0.5, 1.0, 2.0].contains(&f) {
                    bail!("the feel is 0.5, 1 or 2");
                }
                Command::Feel(f)
            }
            "speed" => Command::Speed(float(num("speed")?)?.clamp(0.25, 2.0)),
            "pitch" => Command::Pitch(float(num("semitones")?)?.clamp(-12.0, 12.0)),
            "keep_pitch" => Command::KeepPitch(on(w.next())?),
            "repair" => Command::Repair,
            "save" => {
                let name = rest(1);
                Command::Save(if name.is_empty() { "Loop".into() } else { name })
            }
            "recall" => Command::Recall(int(num("bookmark")?)?),
            "rename" => {
                let created = int(num("bookmark")?)?;
                let name = rest(2);
                if name.is_empty() {
                    bail!("rename to what?");
                }
                Command::Rename { created, name }
            }
            "delete" => Command::Delete(int(num("bookmark")?)?),
            "playpause" => Command::PlayPause,
            "stems_only" => Command::StemsOnly(on(w.next())?),
            n => match n.parse::<usize>() {
                Ok(bars) if (1..=64).contains(&bars) => Command::Bars(bars),
                _ => bail!("unknown loop command `{n}`"),
            },
        })
    }
}

impl fmt::Display for Command {
    /// The text that parses back into it.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let onoff = |v: bool| if v { "on" } else { "off" };
        let w = Verb::from(self).word();
        match self {
            Command::Bars(n) => write!(f, "{n}"),
            Command::Live | Command::Top | Command::Repair | Command::PlayPause => f.write_str(w),
            Command::Unit(u) => write!(f, "{w} {u}"),
            Command::Meter(m) => write!(f, "{w} {m}"),
            Command::Feel(x) => write!(f, "{w} {x}"),
            Command::Move { start, end } => write!(f, "{w} {start} {end}"),
            Command::Edges { a, b } => write!(f, "{w} {a} {b}"),
            Command::Bar1(i) => write!(f, "{w} {i}"),
            Command::Click(v) | Command::KeepPitch(v) | Command::StemsOnly(v) => {
                write!(f, "{w} {}", onoff(*v))
            }
            Command::Speed(x) | Command::Pitch(x) => write!(f, "{w} {x}"),
            Command::Save(name) => write!(f, "{w} {name}"),
            Command::Recall(c) | Command::Delete(c) => write!(f, "{w} {c}"),
            Command::Rename { created, name } => write!(f, "{w} {created} {name}"),
        }
    }
}

impl Command {
    /// One of each kind, for tests and for checking that UIs cover them all.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn examples() -> Vec<Command> {
        use strum::IntoEnumIterator;
        Verb::iter()
            .map(|v| match v {
                Verb::Bars => Command::Bars(4),
                Verb::Live => Command::Live,
                Verb::Top => Command::Top,
                Verb::Move => Command::Move { start: -4, end: 4 },
                Verb::Unit => Command::Unit(Unit::Bar),
                Verb::Edges => Command::Edges { a: 3, b: 19 },
                Verb::Bar1 => Command::Bar1(7),
                Verb::Click => Command::Click(true),
                Verb::Meter => Command::Meter(3),
                Verb::Feel => Command::Feel(0.5),
                Verb::Speed => Command::Speed(0.75),
                Verb::Pitch => Command::Pitch(-2.0),
                Verb::KeepPitch => Command::KeepPitch(false),
                Verb::Repair => Command::Repair,
                Verb::Save => Command::Save("Horn line".into()),
                Verb::Recall => Command::Recall(1_700_000_000_000),
                Verb::Rename => Command::Rename {
                    created: 1_700_000_000_000,
                    name: "Bridge, take 2".into(),
                },
                Verb::Delete => Command::Delete(1_700_000_000_000),
                Verb::PlayPause => Command::PlayPause,
                Verb::StemsOnly => Command::StemsOnly(true),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trips() {
        for c in Command::examples() {
            let t = c.to_string();
            assert_eq!(t.parse::<Command>().unwrap(), c, "{t}");
        }
        assert_eq!("again".parse::<Command>().unwrap(), Command::Top);
        assert_eq!(
            "save".parse::<Command>().unwrap(),
            Command::Save("Loop".into())
        );
        for bad in [
            "sideways",
            "move 1",
            "edges 3 1",
            "click maybe",
            "0",
            "65",
            "speed NaN",
            "",
        ] {
            assert!(bad.parse::<Command>().is_err(), "{bad}");
        }
    }
}
