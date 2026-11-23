//! Saved loops, in the Android app's format (Bookmarks.kt): the recording and the
//! loop in song time, with its bars, and the beats around it (song ms; the loop is
//! beats[a]..beats[b], a bar line on beats[one]) so recalling lays them straight back
//! without analysing. Kept in `bookmarks.json` in stemflow's state folder.
//!
//! Reading takes every field with a fallback; writing merges into the entry as read,
//! so fields this version doesn't know survive a rename; an entry that doesn't parse
//! is kept as is, and a file that doesn't parse is set aside before anything is
//! written over it.

use crate::songmap::Song;
use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Bookmark {
    pub name: String,
    pub song: Song,
    pub start_ms: i64,
    pub end_ms: i64,
    pub bar1_ms: i64,
    pub meter: usize,
    pub bars: usize,
    pub feel: f32,
    /// When it was saved (ms since 1970): its identity.
    pub created: i64,
    pub beats: Vec<f64>,
    pub a: usize,
    pub b: usize,
    pub one: usize,
    pub bpm: f32,
    raw: Map<String, Value>,
}

/// Beats in song ms, the loop as indices into them, a bar line, the tempo.
pub struct Grid {
    pub beats: Vec<f64>,
    pub a: usize,
    pub b: usize,
    pub one: usize,
    pub bpm: f32,
}

const VERSION: i64 = 2;

impl Bookmark {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        song: Song,
        start_ms: i64,
        end_ms: i64,
        bar1_ms: i64,
        meter: usize,
        bars: usize,
        feel: f32,
    ) -> Self {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as i64);
        Bookmark {
            name: name.into(),
            song,
            start_ms,
            end_ms,
            bar1_ms,
            meter,
            bars,
            feel,
            created,
            beats: Vec::new(),
            a: 0,
            b: 0,
            one: 0,
            bpm: 0.0,
            raw: Map::new(),
        }
    }

    pub fn has_grid(&self) -> bool {
        self.beats.len() > 1 && self.a < self.b && self.b < self.beats.len()
    }

    pub fn with_grid(mut self, g: Grid) -> Self {
        (self.beats, self.a, self.b, self.one, self.bpm) = (g.beats, g.a, g.b, g.one, g.bpm);
        self
    }

    fn of(o: &Map<String, Value>) -> Option<Bookmark> {
        let s = |k: &str| o.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let i = |k: &str| o.get(k).and_then(Value::as_i64);
        let f = |k: &str| o.get(k).and_then(Value::as_f64);
        let start = i("startMs")?;
        Some(Bookmark {
            name: o.get("name")?.as_str()?.to_string(),
            song: Song {
                id: s("mediaId"),
                title: s("title"),
                artist: s("artist"),
                album: s("album"),
                app: s("pkg"),
            },
            start_ms: start,
            end_ms: i("endMs")?,
            bar1_ms: i("bar1Ms").unwrap_or(start),
            meter: i("meter").unwrap_or(4).max(1) as usize,
            bars: i("bars").unwrap_or(4).max(1) as usize,
            feel: f("feel").unwrap_or(1.0) as f32,
            created: i("created").unwrap_or(0),
            beats: o
                .get("beats")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default(),
            a: i("a").unwrap_or(0).max(0) as usize,
            b: i("b").unwrap_or(0).max(0) as usize,
            one: i("one").unwrap_or(0).max(0) as usize,
            bpm: f("bpm").unwrap_or(0.0) as f32,
            raw: o.clone(),
        })
    }

    fn json(&self) -> Value {
        let mut o = self.raw.clone();
        let v = o.get("v").and_then(Value::as_i64).unwrap_or(1);
        let mut put = |k: &str, v: Value| {
            o.insert(k.into(), v);
        };
        put("v", json!(if self.has_grid() { v.max(VERSION) } else { v }));
        put("name", json!(self.name));
        put("pkg", json!(self.song.app));
        put("title", json!(self.song.title));
        put("artist", json!(self.song.artist));
        put("album", json!(self.song.album));
        put("mediaId", json!(self.song.id));
        put("startMs", json!(self.start_ms));
        put("endMs", json!(self.end_ms));
        put("bar1Ms", json!(self.bar1_ms));
        put("meter", json!(self.meter));
        put("bars", json!(self.bars));
        put("feel", json!(self.feel));
        put("created", json!(self.created));
        if self.has_grid() {
            let beats: Vec<f64> = self
                .beats
                .iter()
                .map(|b| (b * 10.0).round() / 10.0)
                .collect();
            put("beats", json!(beats));
            put("a", json!(self.a));
            put("b", json!(self.b));
            put("one", json!(self.one));
            put("bpm", json!(self.bpm));
        }
        Value::Object(o)
    }

    /// For a UI: one JSON object.
    pub fn summary(&self) -> Value {
        json!({
            "name": self.name, "title": self.song.title, "artist": self.song.artist,
            "album": self.song.album, "start": self.start_ms, "end": self.end_ms,
            "bars": self.bars, "meter": self.meter, "created": self.created,
        })
    }
}

static STATE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Keep stemflow's state (bookmarks, the loop options) in `dir` rather than
/// `$XDG_STATE_HOME/stemflow`: the Android app's files folder, where its bookmarks
/// have always been.
#[allow(dead_code)] // the Android app's
pub fn set_state_dir(dir: PathBuf) {
    let _ = STATE_DIR.set(dir);
}

/// Where stemflow keeps its state.
pub fn state_dir() -> Option<PathBuf> {
    if let Some(d) = STATE_DIR.get() {
        return Some(d.clone());
    }
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))?;
    Some(base.join("stemflow"))
}

fn file() -> Option<PathBuf> {
    Some(state_dir()?.join("bookmarks.json"))
}

/// Every entry as stored, with the bookmark it holds (None if it doesn't parse).
fn load() -> Vec<(Value, Option<Bookmark>)> {
    let Some(f) = file() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&f) else {
        return Vec::new();
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Array(a)) => a
            .into_iter()
            .map(|o| {
                let b = o.as_object().and_then(Bookmark::of);
                (o, b)
            })
            .collect(),
        _ => {
            // unreadable: keep it aside rather than save over it
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let _ = std::fs::rename(&f, f.with_extension(format!("json.bad-{t}")));
            Vec::new()
        }
    }
}

pub fn all() -> Vec<Bookmark> {
    load().into_iter().filter_map(|(_, b)| b).collect()
}

/// Rewrite the file: each entry through `f` (None drops it), unparsed ones kept.
fn edit(extra: Option<&Bookmark>, f: impl Fn(Bookmark) -> Option<Bookmark>) -> Result<()> {
    let mut out = Vec::new();
    for (o, b) in load() {
        match b {
            None => out.push(o),
            Some(b) => {
                if let Some(b) = f(b) {
                    out.push(b.json())
                }
            }
        }
    }
    if let Some(b) = extra {
        out.push(b.json());
    }
    let path = file().context("no HOME")?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&Value::Array(out))?)?;
    std::fs::rename(&tmp, &path).context("writing bookmarks.json")
}

pub fn add(b: &Bookmark) -> Result<()> {
    edit(Some(b), Some)
}

/// Write `b` over the bookmark saved at the same time.
pub fn replace(b: &Bookmark) -> Result<()> {
    edit(None, |x| {
        Some(if x.created == b.created { b.clone() } else { x })
    })
}

pub fn rename(created: i64, name: &str) -> Result<()> {
    edit(None, |mut x| {
        if x.created == created {
            x.name = name.into();
        }
        Some(x)
    })
}

pub fn remove(created: i64) -> Result<()> {
    edit(None, |x| (x.created != created).then_some(x))
}

/// The bookmarks for a UI, newest song first, as a JSON array.
pub fn list_json() -> String {
    let v: Vec<Value> = all().iter().map(Bookmark::summary).collect();
    Value::Array(v).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_unknown_fields() {
        let mut o = Map::new();
        o.insert("name".into(), json!("Intro"));
        o.insert("mediaId".into(), json!("spotify:track:x"));
        o.insert("title".into(), json!("Corcovado"));
        o.insert("startMs".into(), json!(1000));
        o.insert("endMs".into(), json!(9000));
        o.insert("fromTheFuture".into(), json!(true));
        let b = Bookmark::of(&o).unwrap();
        assert_eq!(
            (b.bar1_ms, b.meter, b.bars, b.has_grid()),
            (1000, 4, 4, false)
        );
        let b = b.with_grid(Grid {
            beats: vec![500.0, 1000.0, 1500.0],
            a: 1,
            b: 2,
            one: 1,
            bpm: 120.0,
        });
        let j = b.json();
        assert_eq!(j["fromTheFuture"], json!(true));
        assert_eq!(j["v"], json!(2));
        let c = Bookmark::of(j.as_object().unwrap()).unwrap();
        assert!(c.has_grid() && c.song.same(&b.song) && c.end_ms == 9000);
    }
}
