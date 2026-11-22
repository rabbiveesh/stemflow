//! The captured app as a `loopdeck::Source`: pausing and playing it, what it plays
//! and where, seeking, and opening a recording. MPRIS on Linux; AppleScript on macOS
//! (Spotify and Music).

use crate::loopdeck::{Now, Source};
#[allow(unused_imports)]
use crate::songmap::Song;

/// The player for `grab` (the `--grab` app name).
pub fn for_app(grab: &str) -> Box<dyn Source> {
    Box::new(Player {
        needle: grab.to_lowercase(),
        #[cfg(target_os = "linux")]
        zeros: std::sync::atomic::AtomicU32::new(0),
    })
}

struct Player {
    needle: String,
    /// Linux: polls in a row where the player said position 0 while playing (some
    /// players never report it).
    #[cfg(target_os = "linux")]
    zeros: std::sync::atomic::AtomicU32,
}

#[cfg(target_os = "linux")]
mod mpris {
    use super::*;
    use std::collections::HashMap;
    use zbus::zvariant::{ObjectPath, OwnedValue, Value};

    const PATH: &str = "/org/mpris/MediaPlayer2";
    const PLAYER: &str = "org.mpris.MediaPlayer2.Player";

    fn text(v: &Value) -> String {
        match v {
            Value::Str(s) => s.to_string(),
            Value::ObjectPath(p) => p.to_string(),
            Value::Array(a) => a.iter().next().map(text).unwrap_or_default(),
            Value::Value(v) => text(v),
            _ => String::new(),
        }
    }

    impl Player {
        fn conn(&self) -> zbus::Result<Option<(zbus::blocking::Connection, String)>> {
            let c = zbus::blocking::Connection::session()?;
            let names: Vec<String> = c
                .call_method(
                    Some("org.freedesktop.DBus"),
                    "/org/freedesktop/DBus",
                    Some("org.freedesktop.DBus"),
                    "ListNames",
                    &(),
                )?
                .body()
                .deserialize()?;
            let name = names.into_iter().find(|n| {
                n.strip_prefix("org.mpris.MediaPlayer2.")
                    .is_some_and(|p| p.to_lowercase().contains(&self.needle))
            });
            Ok(name.map(|n| (c, n)))
        }

        /// Call `method` with `args` on the player; false if there is none.
        pub(super) fn call<B>(&self, method: &str, args: &B) -> bool
        where
            B: zbus::export::serde::ser::Serialize + zbus::zvariant::DynamicType,
        {
            let r = (|| -> zbus::Result<bool> {
                let Some((c, name)) = self.conn()? else {
                    return Ok(false);
                };
                c.call_method(Some(name.as_str()), PATH, Some(PLAYER), method, args)?;
                Ok(true)
            })();
            r.unwrap_or_else(|e| {
                eprintln!(
                    "stemflow: couldn't {method} {} over MPRIS: {e}",
                    self.needle
                );
                false
            })
        }

        fn props(&self) -> zbus::Result<Option<HashMap<String, OwnedValue>>> {
            let Some((c, name)) = self.conn()? else {
                return Ok(None);
            };
            let p: HashMap<String, OwnedValue> = c
                .call_method(
                    Some(name.as_str()),
                    PATH,
                    Some("org.freedesktop.DBus.Properties"),
                    "GetAll",
                    &(PLAYER,),
                )?
                .body()
                .deserialize()?;
            Ok(Some(p))
        }

        /// The song, with `mpris:trackid` (for SetPosition), and the position.
        pub(super) fn now_mpris(&self) -> Option<(Now, String)> {
            let p = self.props().ok()??;
            let md: HashMap<String, OwnedValue> =
                p.get("Metadata")?.try_clone().ok()?.try_into().ok()?;
            let get = |k: &str| md.get(k).map(|v| text(v)).unwrap_or_default();
            let title = get("xesam:title");
            if title.is_empty() {
                return None;
            }
            let trackid = get("mpris:trackid");
            // a recording's id: its URL (Spotify: open.spotify.com/track/…, which
            // `open` turns into spotify:track:…), else the player's track id
            let url = get("xesam:url");
            let id = if url.is_empty() { trackid.clone() } else { url };
            let us = p.get("Position").and_then(|v| i64::try_from(&**v).ok())?;
            let playing = p.get("PlaybackStatus").map(|v| text(v)).as_deref() == Some("Playing");
            Some((
                Now {
                    song: Song {
                        id,
                        title,
                        artist: get("xesam:artist"),
                        album: get("xesam:album"),
                    },
                    ms: us / 1000,
                    playing,
                },
                trackid,
            ))
        }

        pub(super) fn set_position(&self, trackid: &str, ms: i64) -> bool {
            match ObjectPath::try_from(trackid) {
                Ok(p) => self.call("SetPosition", &(p, ms * 1000)),
                Err(_) => false,
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl Source for Player {
    fn pause(&self) -> bool {
        self.call("Pause", &())
    }
    fn play(&self) -> bool {
        self.call("Play", &())
    }
    fn now(&self) -> Option<Now> {
        use std::sync::atomic::Ordering;
        let (n, _) = self.now_mpris()?;
        // A player that never reports its position (it says 0 while playing) can't
        // place anything in the song.
        if n.playing && n.ms == 0 {
            if self.zeros.fetch_add(1, Ordering::Relaxed) >= 4 {
                return None;
            }
        } else {
            self.zeros.store(0, Ordering::Relaxed);
        }
        Some(n)
    }
    fn seek(&self, ms: i64) -> bool {
        let Some((_, trackid)) = self.now_mpris() else {
            return false;
        };
        self.set_position(&trackid, ms)
    }
    fn open(&self, id: &str) -> bool {
        // Spotify gives its tracks as https://open.spotify.com/track/<id>
        let uri = match id.strip_prefix("https://open.spotify.com/track/") {
            Some(t) => format!("spotify:track:{}", t.split('?').next().unwrap_or(t)),
            None => id.to_string(),
        };
        !uri.is_empty() && self.call("OpenUri", &(uri,))
    }
}

#[cfg(target_os = "macos")]
impl Player {
    /// The scriptable app for the grab name.
    fn app(&self) -> Option<&'static str> {
        if self.needle.contains("spotify") {
            Some("Spotify")
        } else if self.needle.contains("music") {
            Some("Music")
        } else {
            None
        }
    }

    /// Run `body` in a `tell` to the app, only if it's running (a tell launches it);
    /// its result, or None. The first time, macOS asks whether stemflow may control
    /// the app.
    fn tell(&self, body: &str) -> Option<String> {
        let app = self.app()?;
        let script = format!(
            "if application \"{app}\" is running then\ntell application \"{app}\"\n{body}\nend tell\nend if"
        );
        match std::process::Command::new("osascript")
            .args(["-e", &script])
            .output()
        {
            Ok(o) if o.status.success() => Some(
                String::from_utf8_lossy(&o.stdout)
                    .trim_end_matches('\n')
                    .to_string(),
            ),
            Ok(o) => {
                eprintln!(
                    "stemflow: couldn't talk to {app}: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                None
            }
            Err(e) => {
                eprintln!("stemflow: couldn't run osascript: {e}");
                None
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Source for Player {
    fn pause(&self) -> bool {
        self.tell("pause").is_some()
    }
    fn play(&self) -> bool {
        self.tell("play").is_some()
    }
    fn now(&self) -> Option<Now> {
        let id = if self.app()? == "Spotify" {
            "id of current track"
        } else {
            "persistent ID of current track"
        };
        // one line, tab-separated; the position in whole ms (no decimal comma)
        let out = self.tell(&format!(
            "if player state is stopped then return \"\"\n\
             set t to current track\n\
             return ({id}) & tab & (name of t) & tab & (artist of t) & tab & (album of t) & tab & \
             ((player position * 1000) as integer) & tab & (player state as text)"
        ))?;
        let f: Vec<&str> = out.split('\t').collect();
        if f.len() < 6 || f[1].is_empty() {
            return None;
        }
        Some(Now {
            song: Song {
                id: f[0].into(),
                title: f[1].into(),
                artist: f[2].into(),
                album: f[3].into(),
            },
            ms: f[4].trim().parse().ok()?,
            playing: f[5].trim() == "playing",
        })
    }
    fn seek(&self, ms: i64) -> bool {
        self.tell(&format!(
            "set player position to {}.{:03}",
            ms / 1000,
            ms % 1000
        ))
        .is_some()
    }
    fn open(&self, id: &str) -> bool {
        // Spotify plays its track URIs; Music has no such thing to script
        self.app() == Some("Spotify")
            && id.starts_with("spotify:")
            && self
                .tell(&format!("play track \"{}\"", id.replace('"', "")))
                .is_some()
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl Source for Player {
    fn pause(&self) -> bool {
        false
    }
    fn play(&self) -> bool {
        false
    }
}
