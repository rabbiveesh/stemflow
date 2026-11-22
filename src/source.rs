//! Pausing and playing the captured app while a loop plays (`loopdeck::Source`):
//! MPRIS on Linux, AppleScript on macOS (Spotify and Music).

use crate::loopdeck::Source;

/// The player for `grab` (the `--grab` app name).
pub fn for_app(grab: &str) -> Box<dyn Source> {
    Box::new(Player {
        needle: grab.to_lowercase(),
    })
}

struct Player {
    needle: String,
}

#[cfg(target_os = "linux")]
impl Player {
    /// Call `method` on the MPRIS player whose bus name contains the app's name.
    fn call(&self, method: &str) -> bool {
        let r = (|| -> zbus::Result<bool> {
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
            let Some(name) = names.iter().find(|n| {
                n.strip_prefix("org.mpris.MediaPlayer2.")
                    .is_some_and(|p| p.to_lowercase().contains(&self.needle))
            }) else {
                return Ok(false);
            };
            c.call_method(
                Some(name.as_str()),
                "/org/mpris/MediaPlayer2",
                Some("org.mpris.MediaPlayer2.Player"),
                method,
                &(),
            )?;
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

    /// `tell application … to <verb>`, only if it's running (a tell launches it).
    /// The first time, macOS asks whether stemflow may control the app.
    fn call(&self, verb: &str) -> bool {
        let Some(app) = self.app() else {
            return false;
        };
        let script = format!(
            "if application \"{app}\" is running then tell application \"{app}\" to {verb}"
        );
        match std::process::Command::new("osascript")
            .args(["-e", &script])
            .output()
        {
            Ok(o) if o.status.success() => true,
            Ok(o) => {
                eprintln!(
                    "stemflow: couldn't {verb} {app}: {}",
                    String::from_utf8_lossy(&o.stderr).trim()
                );
                false
            }
            Err(e) => {
                eprintln!("stemflow: couldn't run osascript: {e}");
                false
            }
        }
    }
}

impl Source for Player {
    fn pause(&self) -> bool {
        #[cfg(target_os = "linux")]
        return self.call("Pause");
        #[cfg(target_os = "macos")]
        return self.call("pause");
        #[allow(unreachable_code)]
        false
    }

    fn play(&self) -> bool {
        #[cfg(target_os = "linux")]
        return self.call("Play");
        #[cfg(target_os = "macos")]
        return self.call("play");
        #[allow(unreachable_code)]
        false
    }
}
