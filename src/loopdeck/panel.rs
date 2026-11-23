//! The loop panel, declared once: every row of the Loops screen, what each control
//! shows and what it sends, built from the loop's state. The macOS menu
//! (macos/StemflowBar/Loop.swift), the GNOME extension (gnome/stemflow@veesh/loop.js)
//! and the Android app (android/app/.../LoopPanel.kt) only draw it: they render a
//! handful of item kinds (`Kind`) and send the lines they're given. So a feature
//! added here is on every platform at once, and the tests below hold that:
//! - every verb of `Command` is sent by some control (no feature without a UI);
//! - the rows and their items keep their kinds whatever the state (renderers build
//!   them once and update them in place);
//! - the platforms' generated lists of kinds are current, and each renderer has a
//!   case for every kind (Swift's `switch` and Kotlin's `when` are exhaustive over
//!   the generated enums, so a new kind doesn't build until it's drawn; GNOME's
//!   JavaScript is checked here).

use super::command::Unit;
use super::State;
use crate::bookmarks::Bookmark;
use serde::Serialize;

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Panel {
    pub rows: Vec<Row>,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Row {
    pub id: &'static str,
    pub hidden: bool,
    pub items: Vec<Item>,
}

/// Lines to send over the control socket (or D-Bus `Loop`, without the `loop `),
/// in order. `{name}` in one is the name asked for (`ask`).
pub type Send = Vec<String>;

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Ask {
    pub title: String,
    pub text: String,
    pub value: String,
    pub ok: String,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Opt {
    pub label: String,
    pub send: Send,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Mark {
    pub label: String,
    /// What a search matches, lower case.
    pub search: String,
    pub play: Send,
    pub rename: Send,
    pub rename_ask: Ask,
    pub delete: Send,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Group {
    pub title: String,
    pub marks: Vec<Mark>,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    /// A heading.
    Title,
    Normal,
    /// Small print.
    Small,
}

#[derive(Serialize, Debug, Clone, PartialEq, strum::EnumDiscriminants)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[strum_discriminants(name(Kind), derive(strum::EnumIter, strum::IntoStaticStr))]
#[strum_discriminants(strum(serialize_all = "snake_case"))]
pub enum Item {
    Text {
        text: String,
        style: Style,
    },
    /// `on`: a toggle's state (None: not a toggle). With `ask`, a name is asked for
    /// first.
    Button {
        label: String,
        send: Send,
        enabled: bool,
        on: Option<bool>,
        ask: Option<Ask>,
    },
    /// A menu item with a switch.
    Switch {
        label: String,
        on: bool,
        send: Send,
        tip: String,
    },
    /// One of several, `selected` (None: none of them). `title` names it where
    /// it's shown as a menu rather than side by side ("" for none).
    Choice {
        title: String,
        options: Vec<Opt>,
        selected: Option<usize>,
    },
    /// A label, ◀ ▶ and a note under them.
    Nudge {
        label: String,
        note: String,
        back: Send,
        forward: Send,
    },
    /// `send` with `{v}` for the value, `min`..`max` in steps of `step` (snapping
    /// to `snap` within two steps); `text` shows the value.
    Slider {
        label: String,
        min: f64,
        max: f64,
        step: f64,
        snap: Option<f64>,
        value: f64,
        text: String,
        send: String,
    },
    /// The loop drawn from `loop view <bins>` (polled while shown): dragging an end
    /// sends `edges` with `{a}`, `{b}` (beat indices), making a beat bar 1 `bar1`
    /// with `{beat}`.
    LoopView {
        edges: String,
        bar1: String,
    },
    /// The saved loops by recording, with a search field.
    Bookmarks {
        title: String,
        search_hint: String,
        empty: String,
        no_match: String,
        groups: Vec<Group>,
    },
}

impl Kind {
    pub fn name(self) -> &'static str {
        self.into()
    }
}

fn s(x: &str) -> String {
    x.to_string()
}

fn send(lines: &[String]) -> Send {
    lines.to_vec()
}

fn one(line: impl Into<String>) -> Send {
    vec![line.into()]
}

fn button(label: &str, line: impl Into<String>, enabled: bool) -> Item {
    Item::Button {
        label: s(label),
        send: one(line),
        enabled,
        on: None,
        ask: None,
    }
}

fn toggle(label: &str, on: bool, verb: &str) -> Item {
    Item::Button {
        label: s(label),
        send: one(format!("loop {verb} {}", if on { "off" } else { "on" })),
        enabled: true,
        on: Some(on),
        ask: None,
    }
}

fn text(t: impl Into<String>, style: Style) -> Item {
    Item::Text {
        text: t.into(),
        style,
    }
}

/// `m:ss`
pub fn mmss(ms: i64) -> String {
    crate::songmap::time(ms.max(0))
}

fn plural(n: usize, one: &str) -> String {
    format!("{n} {one}{}", if n == 1 { "" } else { "s" })
}

const METERS: [u32; 7] = [0, 2, 3, 4, 5, 6, 7];
const FEELS: [(f32, &str); 3] = [(1.0, "As heard"), (0.5, "½×"), (2.0, "2×")];

/// The panel for state `l`, with the beat options `meter` and `feel` and the saved
/// loops `marks`.
pub fn build(l: &State, meter: u32, feel: f32, marks: &[Bookmark]) -> Panel {
    let playing = l.playing;
    let m = l.meter.max(1);
    let beats = l.beats;
    let mut rows = Vec::new();
    let mut row = |id: &'static str, hidden: bool, items: Vec<Item>| {
        rows.push(Row { id, hidden, items });
    };

    // the player: what plays, play/pause it, save the loop; the saved loops
    let now = if l.title.is_empty() {
        s("The player doesn't say what's playing")
    } else if playing && l.loop_from >= 0 {
        format!(
            "{} — {} · looping {}–{}",
            l.title,
            l.artist,
            mmss(l.loop_from),
            mmss(l.loop_to)
        )
    } else {
        format!(
            "{} — {} · {}{}",
            l.title,
            l.artist,
            mmss(l.song_ms),
            if l.song_playing { "" } else { " · paused" }
        )
    };
    row("now", false, vec![text(now, Style::Small)]);
    let guess = if l.title.is_empty() {
        s("Loop")
    } else if l.loop_from >= 0 {
        format!("{} {}", l.title, mmss(l.loop_from))
    } else {
        l.title.clone()
    };
    row(
        "strip",
        false,
        vec![
            button(
                if l.song_playing { "Pause" } else { "Play" },
                "loop playpause",
                !l.title.is_empty() && !playing,
            ),
            Item::Button {
                label: s("Save…"),
                send: one("loop save {name}"),
                enabled: playing,
                on: None,
                ask: Some(Ask {
                    title: s("Save this loop"),
                    text: s("Name it to come back to it later."),
                    value: guess,
                    ok: s("Save"),
                }),
            },
        ],
    );
    row("bookmarks", false, vec![bookmarks(marks)]);

    let info = if playing {
        let (whole, extra) = (beats / m, beats % m);
        format!(
            "Looping {}{} of {m}/4 · {:.0} bpm · {:.1} s",
            plural(whole, "bar"),
            if extra > 0 {
                format!(" + {}", plural(extra, "beat"))
            } else {
                String::new()
            },
            l.bpm,
            l.length
        )
    } else if l.recalling {
        s("Going back to a bookmark…")
    } else if l.finding {
        s("Finding the bars…")
    } else if l.tail {
        s("Catching up with the song")
    } else {
        s("Loop the last…")
    };
    row("info", false, vec![text(info, Style::Title)]);
    let hint = if !l.status.is_empty() && !l.finding {
        l.status.clone()
    } else if playing {
        s("Drag an end · hold a beat (or right-click it) for bar 1")
    } else if l.finding {
        String::new()
    } else {
        s("Press a length to loop what you just heard")
    };
    row("hint", hint.is_empty(), vec![text(hint, Style::Small)]);

    let mut grab: Vec<Item> = [1usize, 2, 4, 8]
        .iter()
        .map(|&n| Item::Button {
            label: if n == 1 {
                s("1 bar")
            } else {
                format!("{n} bars")
            },
            send: one(format!("loop {n}")),
            enabled: true,
            on: Some(playing && l.bars == n),
            ask: None,
        })
        .collect();
    grab.push(button("Live", "loop live", playing || l.tail || l.finding));
    row("grab", false, grab);

    row(
        "view",
        !playing,
        vec![Item::LoopView {
            edges: s("loop edges {a} {b}"),
            bar1: s("loop bar1 {beat}"),
        }],
    );

    let units: Vec<Unit> = [Unit::Beat, Unit::Bar, Unit::Phrase].into();
    row(
        "units",
        !playing,
        vec![Item::Choice {
            title: s("Step by"),
            options: units
                .iter()
                .map(|u| Opt {
                    label: match u {
                        Unit::Beat => s("Beat"),
                        Unit::Bar => s("Bar"),
                        Unit::Phrase if beats > 0 && beats.is_multiple_of(m) => {
                            format!("{}-bar phrase", beats / m)
                        }
                        Unit::Phrase => s("Phrase"),
                    },
                    send: one(format!("loop unit {u}")),
                })
                .collect(),
            selected: units.iter().position(|&u| u == l.unit),
        }],
    );

    // Start, the whole loop, or the end: earlier or later by the unit.
    let step = match l.unit {
        Unit::Beat => 1,
        Unit::Bar => m,
        Unit::Phrase => beats.max(1),
    } as i64;
    let nudge = |label: &str, note: &str, da: i64, db: i64| Item::Nudge {
        label: s(label),
        note: s(note),
        back: one(format!("loop move {} {}", -da * step, -db * step)),
        forward: one(format!("loop move {} {}", da * step, db * step)),
    };
    row(
        "edit",
        !playing,
        vec![
            nudge("START", "◀ longer · shorter ▶", 1, 0),
            nudge("MOVE", "same length", 1, 1),
            nudge("END", "◀ shorter · longer ▶", 0, 1),
        ],
    );

    // Meter and feel re-decode the loop's bars (and the click's).
    row(
        "beat",
        false,
        vec![
            Item::Choice {
                title: s("Beats per bar"),
                options: METERS
                    .iter()
                    .map(|&v| Opt {
                        label: if v == 0 { s("Auto") } else { v.to_string() },
                        send: one(format!("loop meter {v}")),
                    })
                    .collect(),
                selected: METERS.iter().position(|&v| v == meter).or(Some(0)),
            },
            Item::Choice {
                title: s("Feel"),
                options: FEELS
                    .iter()
                    .map(|&(v, label)| Opt {
                        label: s(label),
                        send: one(format!("loop feel {v}")),
                    })
                    .collect(),
                selected: FEELS.iter().position(|&(v, _)| v == feel).or(Some(0)),
            },
        ],
    );

    // Practice speed (40-120%) and transposition (±12 semitones).
    row(
        "speed",
        false,
        vec![Item::Slider {
            label: s("Speed"),
            min: 0.4,
            max: 1.2,
            step: 0.01,
            snap: Some(1.0),
            value: l.speed as f64,
            text: format!("{}%", (l.speed * 100.0).round()),
            send: s("loop speed {v}"),
        }],
    );
    let st = l.semis.round() as i64;
    row(
        "pitch",
        false,
        vec![Item::Slider {
            label: s("Pitch"),
            min: -12.0,
            max: 12.0,
            step: 1.0,
            snap: None,
            value: l.semis as f64,
            text: if st == 0 {
                s("0 st")
            } else {
                format!("{st:+} st")
            },
            send: s("loop pitch {v}"),
        }],
    );
    row(
        "chips",
        !playing,
        vec![
            button("75%", "loop speed 0.75", true),
            button("50%", "loop speed 0.5", true),
            toggle("Keep pitch", l.keep_pitch, "keep_pitch"),
            toggle("Click", l.click, "click"),
            Item::Button {
                label: s("Reset"),
                send: send(&[s("loop speed 1"), s("loop pitch 0")]),
                enabled: true,
                on: None,
                ask: None,
            },
        ],
    );
    // From the top; Repair plays the loop's part of the song again and lays the
    // loop on that (for a glitch in it).
    row(
        "actions",
        !playing,
        vec![
            button("From the top", "loop top", true),
            button("Repair", "loop repair", true),
        ],
    );
    row(
        "stems_only",
        false,
        vec![Item::Switch {
            label: s("Stems only when looping"),
            on: l.stems_only,
            send: one(format!(
                "loop stems_only {}",
                if l.stems_only { "off" } else { "on" }
            )),
            tip: s("Hear the plain mix and save power; a loop is separated when you make it"),
        }],
    );
    Panel { rows }
}

/// The saved loops by recording, the newest saved first; in each, by time.
fn bookmarks(marks: &[Bookmark]) -> Item {
    let mut sorted: Vec<&Bookmark> = marks.iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.created));
    let mut by_song: Vec<(String, Vec<&Bookmark>)> = Vec::new();
    for b in sorted {
        let title = format!("{} — {}", b.song.title, b.song.artist);
        match by_song.iter_mut().find(|g| g.0 == title) {
            Some(g) => g.1.push(b),
            None => by_song.push((title, vec![b])),
        }
    }
    let groups = by_song
        .into_iter()
        .map(|(title, mut bs)| {
            bs.sort_by_key(|b| b.start_ms);
            Group {
                title,
                marks: bs
                    .into_iter()
                    .map(|b| Mark {
                        label: format!(
                            "{}  ·  {} · {}",
                            b.name,
                            plural(b.bars, "bar"),
                            mmss(b.start_ms)
                        ),
                        search: format!("{} {} {}", b.name, b.song.title, b.song.artist)
                            .to_lowercase(),
                        play: one(format!("loop recall {}", b.created)),
                        rename: one(format!("loop rename {} {{name}}", b.created)),
                        rename_ask: Ask {
                            title: s("Rename the loop"),
                            text: String::new(),
                            value: b.name.clone(),
                            ok: s("Rename"),
                        },
                        delete: one(format!("loop delete {}", b.created)),
                    })
                    .collect(),
            }
        })
        .collect();
    Item::Bookmarks {
        title: if marks.is_empty() {
            s("Bookmarks")
        } else {
            format!("Bookmarks ({})", marks.len())
        },
        search_hint: s("Search songs and loops"),
        empty: s("Save a loop (Save…) to come back to it later"),
        no_match: s("Nothing matches"),
        groups,
    }
}

impl Panel {
    pub fn json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "null".into())
    }

    /// Every line a control here can send, with sample values for the blanks.
    #[cfg(test)]
    pub fn lines(&self) -> Vec<String> {
        let fill = |l: &str| {
            l.replace("{name}", "Chorus")
                .replace("{v}", "0.8")
                .replace("{a}", "2")
                .replace("{b}", "18")
                .replace("{beat}", "6")
        };
        let mut out = Vec::new();
        for r in &self.rows {
            for i in &r.items {
                match i {
                    Item::Text { .. } => {}
                    Item::Button { send, .. } | Item::Switch { send, .. } => {
                        out.extend(send.iter().map(|l| fill(l)))
                    }
                    Item::Choice { options, .. } => {
                        for o in options {
                            out.extend(o.send.iter().map(|l| fill(l)));
                        }
                    }
                    Item::Nudge { back, forward, .. } => {
                        out.extend(back.iter().chain(forward).map(|l| fill(l)))
                    }
                    Item::Slider { send, .. } => out.push(fill(send)),
                    Item::LoopView { edges, bar1 } => {
                        out.push(fill(edges));
                        out.push(fill(bar1));
                    }
                    Item::Bookmarks { groups, .. } => {
                        for m in groups.iter().flat_map(|g| &g.marks) {
                            out.extend(
                                m.play
                                    .iter()
                                    .chain(&m.rename)
                                    .chain(&m.delete)
                                    .map(|l| fill(l)),
                            );
                        }
                    }
                }
            }
        }
        out
    }
}

/// The platforms' lists of kinds and rows, generated: Swift for the Mac, JavaScript
/// for GNOME.
#[cfg(test)]
pub fn swift_kinds() -> String {
    use strum::IntoEnumIterator;
    let mut o = String::from(
        "// Generated from src/loopdeck/panel.rs by `cargo test`: don't edit.\n\
         // The kinds of item in the loop panel. Loop.swift draws each one; its switch\n\
         // over them is exhaustive, so a new kind doesn't build until it's drawn.\n\n\
         enum PanelKind: String, CaseIterable {\n",
    );
    for k in Kind::iter() {
        let n = k.name();
        let camel: String = n
            .split('_')
            .enumerate()
            .map(|(i, w)| {
                if i == 0 {
                    w.to_string()
                } else {
                    w[..1].to_uppercase() + &w[1..]
                }
            })
            .collect();
        if camel == n {
            let n = if n == "switch" {
                format!("`{n}`")
            } else {
                n.to_string()
            };
            o.push_str(&format!("    case {n}\n"));
        } else {
            o.push_str(&format!("    case {camel} = \"{n}\"\n"));
        }
    }
    o.push_str("}\n");
    o
}

#[cfg(test)]
pub fn js_kinds() -> String {
    use strum::IntoEnumIterator;
    let kinds: Vec<String> = Kind::iter().map(|k| format!("'{}'", k.name())).collect();
    format!(
        "/* Generated from src/loopdeck/panel.rs by `cargo test`: don't edit.\n \
         * The kinds of item in the loop panel; loop.js draws each one (checked by\n \
         * that test). */\nvar KINDS = [{}];\n",
        kinds.join(", ")
    )
}

/// Kotlin for the Android app.
#[cfg(test)]
pub fn kotlin_kinds() -> String {
    use strum::IntoEnumIterator;
    let kinds: Vec<String> = Kind::iter()
        .map(|k| format!("    {}(\"{}\")", k.name().to_uppercase(), k.name()))
        .collect();
    format!(
        "// Generated from src/loopdeck/panel.rs by `cargo test`: don't edit.\n\
         // The kinds of item in the loop panel. LoopPanel.kt draws each one in a `when`\n\
         // over them with no `else`, so a new kind doesn't build until it's drawn.\n\n\
         package org.stemflow.app\n\n\
         enum class PanelKind(val wire: String) {{\n{};\n\n    \
         companion object {{\n        \
         fun of(s: String): PanelKind? = entries.firstOrNull {{ it.wire == s }}\n    \
         }}\n}}\n",
        kinds.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopdeck::command::{Command, Verb};
    use crate::songmap::Song;
    use std::collections::HashSet;
    use strum::IntoEnumIterator;

    fn states() -> Vec<State> {
        let base = State {
            title: s("Corcovado"),
            artist: s("Getz/Gilberto"),
            song_ms: 61_000,
            song_playing: true,
            loop_from: -1,
            loop_to: -1,
            speed: 1.0,
            keep_pitch: true,
            ..Default::default()
        };
        let looping = State {
            playing: true,
            bpm: 72.0,
            meter: 4,
            bars: 4,
            a: 10,
            b: 26,
            beats: 16,
            length: 13.3,
            loop_from: 40_000,
            loop_to: 53_300,
            unit: Unit::Bar,
            ..base.clone()
        };
        vec![
            State::default(),
            base.clone(),
            State {
                finding: true,
                ..base.clone()
            },
            State {
                tail: true,
                ..base.clone()
            },
            State {
                recalling: true,
                status: s("Going to Corcovado at 0:40…"),
                ..base.clone()
            },
            looping.clone(),
            State {
                beats: 18,
                unit: Unit::Phrase,
                speed: 0.75,
                semis: -2.0,
                click: true,
                stems_only: true,
                status: s("The song starts there: the loop can't start any earlier"),
                ..looping
            },
        ]
    }

    fn marks() -> Vec<Bookmark> {
        let song = |t: &str| Song {
            id: format!("spotify:track:{t}"),
            title: t.into(),
            artist: s("Someone"),
            album: String::new(),
            ..Default::default()
        };
        let mut v = vec![
            Bookmark::new("Intro", song("Corcovado"), 1_000, 9_000, 1_000, 4, 4, 1.0),
            Bookmark::new(
                "Bridge",
                song("Corcovado"),
                60_000,
                70_000,
                60_000,
                4,
                4,
                1.0,
            ),
            Bookmark::new("Riff", song("Wave"), 30_000, 36_000, 30_000, 4, 2, 1.0),
        ];
        for (k, b) in v.iter_mut().enumerate() {
            b.created = 1_700_000_000_000 + k as i64;
        }
        v
    }

    fn panels() -> Vec<Panel> {
        let mut out = Vec::new();
        for st in states() {
            for marks in [Vec::new(), marks()] {
                out.push(build(&st, 4, 1.0, &marks));
            }
        }
        out
    }

    /// No feature without a control: every verb is sent by some item, and every
    /// line an item sends is a command.
    #[test]
    fn every_command_has_a_control() {
        let mut seen = HashSet::new();
        for p in panels() {
            for line in p.lines() {
                let rest = line
                    .strip_prefix("loop ")
                    .unwrap_or_else(|| panic!("not a loop line: {line}"));
                let c: Command = rest.parse().unwrap_or_else(|e| panic!("{line}: {e}"));
                seen.insert(Verb::from(&c));
            }
        }
        let missing: Vec<&str> = Verb::iter()
            .filter(|v| !seen.contains(v))
            .map(|v| v.word())
            .collect();
        assert!(missing.is_empty(), "no control sends: {missing:?}");
    }

    /// The same rows with the same kinds of item, whatever the state.
    #[test]
    fn rows_keep_their_shape() {
        let shape = |p: &Panel| -> Vec<(&'static str, Vec<Kind>)> {
            p.rows
                .iter()
                .map(|r| (r.id, r.items.iter().map(Kind::from).collect()))
                .collect()
        };
        let ps = panels();
        for p in &ps {
            assert_eq!(shape(p), shape(&ps[0]));
        }
        let ids: Vec<_> = ps[0].rows.iter().map(|r| r.id).collect();
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "row ids repeat: {ids:?}");
    }

    #[test]
    fn steps_follow_the_unit() {
        let st = &states()[5]; // looping 4 bars of 4, by the bar
        let p = build(st, 4, 1.0, &[]);
        let edit = p.rows.iter().find(|r| r.id == "edit").unwrap();
        let Item::Nudge { forward, back, .. } = &edit.items[2] else {
            panic!()
        };
        assert_eq!(
            (forward[0].as_str(), back[0].as_str()),
            ("loop move 0 4", "loop move 0 -4")
        );
        let st = &states()[6]; // by the phrase, 18 beats
        let p = build(st, 4, 1.0, &[]);
        let edit = p.rows.iter().find(|r| r.id == "edit").unwrap();
        let Item::Nudge { forward, .. } = &edit.items[1] else {
            panic!()
        };
        assert_eq!(forward[0], "loop move 18 18");
    }

    /// The generated lists are current, and the renderers draw every kind.
    #[test]
    fn platforms_draw_every_kind() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut stale = Vec::new();
        for (path, want) in [
            ("macos/StemflowBar/PanelKinds.swift", swift_kinds()),
            ("gnome/stemflow@veesh/panelKinds.js", js_kinds()),
            (
                "android/app/src/main/kotlin/org/stemflow/app/PanelKinds.kt",
                kotlin_kinds(),
            ),
        ] {
            let f = root.join(path);
            if std::fs::read_to_string(&f).ok().as_deref() != Some(want.as_str()) {
                std::fs::write(&f, &want).unwrap();
                stale.push(path);
            }
        }
        assert!(stale.is_empty(), "regenerated {stale:?}: commit them");
        // GNOME: a case in the renderer for each kind
        let js = std::fs::read_to_string(root.join("gnome/stemflow@veesh/loop.js")).unwrap();
        for k in Kind::iter() {
            assert!(
                js.contains(&format!("case '{}':", k.name())),
                "loop.js draws no `{}`",
                k.name()
            );
        }
        // the Mac: the switch over PanelKind names every case (swiftc makes it
        // exhaustive; this catches a `default:` that would hide a new one)
        let swift = std::fs::read_to_string(root.join("macos/StemflowBar/Loop.swift")).unwrap();
        assert!(
            !swift.contains("default:"),
            "Loop.swift: no `default:` in the panel's switch"
        );
        // Android: Kotlin's `when` over an enum, used as a value, must name every
        // case, unless an `else` hides a new one
        let kt = std::fs::read_to_string(
            root.join("android/app/src/main/kotlin/org/stemflow/app/LoopPanel.kt"),
        )
        .unwrap();
        assert!(!kt.contains("else ->"), "LoopPanel.kt: no `else ->`");
        for k in Kind::iter() {
            let case = format!("PanelKind.{} ->", k.name().to_uppercase());
            assert!(kt.contains(&case), "LoopPanel.kt draws no `{}`", k.name());
        }
    }
}
