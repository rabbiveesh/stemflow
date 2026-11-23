//! The loop deck (src/loopdeck) for the app: the same state machine the desktop
//! runs, with the player reached through a Kotlin object (`DeckSource`), and the loop
//! panel (src/loopdeck/panel.rs) that the Loops screen draws.
//!
//! The deck owns the looper; `deckLooper` hands out its handle for the audio
//! threads' `looper*` calls (push, read, ...), valid until `deckFree`.

use crate::beats::Settings;
use crate::bookmarks;
use crate::loopdeck::{panel, Command, Deck, Now, Source};
use crate::looper::Looper;
use crate::songmap::Song;
use jni::objects::{GlobalRef, JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jfloat, jint, jlong, jstring};
use jni::{JNIEnv, JavaVM};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// The player, as the app's `DeckSource` reaches it (its media session).
struct JavaSource {
    vm: JavaVM,
    obj: GlobalRef,
}

impl JavaSource {
    /// Call into Kotlin from whatever thread this is (the deck's own threads stay
    /// attached), in a local frame: those threads never return to Java to free
    /// their references.
    fn call<R>(
        &self,
        f: impl FnOnce(&mut JNIEnv, &JObject) -> jni::errors::Result<R>,
    ) -> Option<R> {
        let mut env = self.vm.attach_current_thread_permanently().ok()?;
        let r = env.with_local_frame(8, |env| f(env, self.obj.as_obj()));
        if env.exception_check().unwrap_or(false) {
            let _ = env.exception_describe();
            let _ = env.exception_clear();
            return None;
        }
        r.ok()
    }

    fn yes(&self, name: &str) -> bool {
        self.call(|env, o| env.call_method(o, name, "()Z", &[])?.z())
            .unwrap_or(false)
    }
}

impl Source for JavaSource {
    fn pause(&self) -> bool {
        self.yes("pause")
    }

    fn play(&self) -> bool {
        self.yes("play")
    }

    /// `DeckSource.now()`: tab-separated id, title, artist, album, app, ms,
    /// playing (1/0), length ms; null when the player doesn't say.
    fn now(&self) -> Option<Now> {
        let s = self.call(|env, o| {
            let v = env
                .call_method(o, "now", "()Ljava/lang/String;", &[])?
                .l()?;
            if v.is_null() {
                return Ok(None);
            }
            let s: String = env.get_string(&JString::from(v))?.into();
            Ok(Some(s))
        })??;
        let f: Vec<&str> = s.split('\t').collect();
        if f.len() < 8 {
            return None;
        }
        Some(Now {
            song: Song {
                id: f[0].into(),
                title: f[1].into(),
                artist: f[2].into(),
                album: f[3].into(),
                app: f[4].into(),
            },
            ms: f[5].parse().ok()?,
            playing: f[6] == "1",
            len_ms: f[7].parse().unwrap_or(0),
        })
    }

    fn seek(&self, ms: i64) -> bool {
        self.call(|env, o| env.call_method(o, "seek", "(J)Z", &[JValue::Long(ms)])?.z())
            .unwrap_or(false)
    }

    fn open(&self, song: &Song) -> bool {
        self.call(|env, o| {
            let id = JObject::from(env.new_string(&song.id)?);
            let app = JObject::from(env.new_string(&song.app)?);
            let what =
                JObject::from(env.new_string(format!("{} {}", song.title, song.artist).trim())?);
            env.call_method(
                o,
                "open",
                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Z",
                &[
                    JValue::Object(&id),
                    JValue::Object(&app),
                    JValue::Object(&what),
                ],
            )?
            .z()
        })
        .unwrap_or(false)
    }
}

/// The deck, and the beat options it reads bars with (the desktop keeps these in
/// its control state).
struct Handle {
    deck: Arc<Deck>,
    meter: AtomicU32,
    feel: AtomicU32,
}

impl Handle {
    fn settings(&self) -> Settings {
        Settings {
            meter: self.meter.load(Ordering::Relaxed) as usize,
            feel: f32::from_bits(self.feel.load(Ordering::Relaxed)),
            ..Default::default()
        }
    }
}

fn handle<'a>(h: jlong) -> &'a Handle {
    unsafe { &*(h as *const Handle) }
}

fn jstr(env: &mut JNIEnv, s: &str) -> jstring {
    env.new_string(s)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// A deck around looper `looper` (from `looperNew`, which this takes over: don't
/// free it), the player reached through `source`, state kept in `dir` (the app's
/// files: bookmarks.json is the one the app always had). The beat options start at
/// `meter` and `feel`.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckNew<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    looper: jlong,
    source: JObject<'l>,
    dir: JString<'l>,
    meter: jint,
    feel: jfloat,
) -> jlong {
    let r = (|| -> anyhow::Result<jlong> {
        anyhow::ensure!(looper != 0, "no looper");
        let dir: String = env.get_string(&dir)?.into();
        bookmarks::set_state_dir(dir.into());
        let src = JavaSource {
            vm: env.get_java_vm()?,
            obj: env.new_global_ref(source)?,
        };
        let l = *unsafe { Box::from_raw(looper as *mut Looper) };
        let deck = Deck::new(l, Box::new(src), Box::new(|| {}));
        // the player's song position, for the song map
        deck.watch(std::time::Duration::from_millis(500));
        let h = Handle {
            deck,
            meter: AtomicU32::new(meter.clamp(0, 12) as u32),
            feel: AtomicU32::new(feel.to_bits()),
        };
        Ok(Box::into_raw(Box::new(h)) as jlong)
    })();
    r.unwrap_or_else(|e| {
        crate::throw(&mut env, e);
        0
    })
}

#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckFree(_: JNIEnv, _: JClass, h: jlong) {
    if h != 0 {
        drop(unsafe { Box::from_raw(h as *mut Handle) });
    }
}

/// The looper's handle, for the `looper*` calls.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckLooper(
    _: JNIEnv,
    _: JClass,
    h: jlong,
) -> jlong {
    Arc::as_ptr(&handle(h).deck.looper) as jlong
}

/// A `loop ...` line from the panel (without the `loop `; see `Command`): null when
/// done, else why it can't be (a line that doesn't parse). Meter and feel are the
/// beat options: kept here, and the loop's bars follow.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckCommand<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    line: JString<'l>,
) -> jstring {
    let r = (|| -> anyhow::Result<()> {
        let line: String = env.get_string(&line)?.into();
        let line = line.trim();
        let line = line.strip_prefix("loop ").unwrap_or(line);
        let h = handle(h);
        match line.parse::<Command>()? {
            Command::Meter(m) => {
                h.meter.store(m, Ordering::Relaxed);
                h.deck.redecode(h.settings());
            }
            Command::Feel(f) => {
                h.feel.store(f.to_bits(), Ordering::Relaxed);
                h.deck.redecode(h.settings());
            }
            _ => h.deck.command(line, h.settings())?,
        }
        Ok(())
    })();
    match r {
        Ok(()) => std::ptr::null_mut(),
        Err(e) => jstr(&mut env, &format!("{e:#}")),
    }
}

/// The loop's state as JSON (`loopdeck::State`), with the beat options.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckState<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
) -> jstring {
    let h = handle(h);
    let mut v: serde_json::Value =
        serde_json::from_str(&h.deck.state().json()).unwrap_or(serde_json::Value::Null);
    if let Some(o) = v.as_object_mut() {
        let s = h.settings();
        o.insert("beat_meter".into(), s.meter.into());
        o.insert("beat_feel".into(), (s.feel as f64).into());
    }
    jstr(&mut env, &v.to_string())
}

/// The loop panel as JSON (`loopdeck::panel`): what the Loops screen draws.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckPanel<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
) -> jstring {
    let h = handle(h);
    let s = h.settings();
    let p = panel::build(&h.deck.state(), s.meter as u32, s.feel, &bookmarks::all());
    jstr(&mut env, &p.json())
}

/// The panel without a deck (stemflow off, or the bar finder still loading): the
/// saved loops and the options as the app keeps them, nothing to loop.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_stemflow_app_Native_panelIdle<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    dir: JString<'l>,
    meter: jint,
    feel: jfloat,
    speed: jfloat,
    semis: jfloat,
    keep_pitch: jboolean,
    click: jboolean,
    stems_only: jboolean,
) -> jstring {
    if let Ok(d) = env.get_string(&dir) {
        bookmarks::set_state_dir(String::from(d).into());
    }
    let st = crate::loopdeck::State {
        speed,
        semis,
        keep_pitch: keep_pitch != 0,
        click: click != 0,
        stems_only: stems_only != 0,
        loop_from: -1,
        loop_to: -1,
        ..Default::default()
    };
    let p = panel::build(&st, meter.clamp(0, 12) as u32, feel, &bookmarks::all());
    jstr(&mut env, &p.json())
}

/// A panel line that needs no deck: renaming or deleting a saved loop. Null when
/// done, else why not.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_idleCommand<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    dir: JString<'l>,
    line: JString<'l>,
) -> jstring {
    let r = (|| -> anyhow::Result<()> {
        let d: String = env.get_string(&dir)?.into();
        bookmarks::set_state_dir(d.into());
        let line: String = env.get_string(&line)?.into();
        let line = line.trim();
        match line
            .strip_prefix("loop ")
            .unwrap_or(line)
            .parse::<Command>()?
        {
            Command::Rename { created, name } => bookmarks::rename(created, &name),
            Command::Delete(created) => bookmarks::remove(created),
            c => anyhow::bail!("`{c}` needs stemflow on"),
        }
    })();
    match r {
        Ok(()) => std::ptr::null_mut(),
        Err(e) => jstr(&mut env, &format!("{e:#}")),
    }
}

/// The worker: buffer frame `frame` is what the player is at now.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckInput(
    _: JNIEnv,
    _: JClass,
    h: jlong,
    frame: jlong,
) {
    handle(h).deck.set_input(frame);
}

/// The player thread: buffer frame `frame` is what the speaker plays now, and the
/// sound reaching the ears trails what it reads by `lag` frames.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckHeard(
    _: JNIEnv,
    _: JClass,
    h: jlong,
    frame: jlong,
    lag: jlong,
) {
    let d = &handle(h).deck;
    d.set_heard(frame);
    d.set_lag(lag);
}

/// The buffer broke here (frames skipped, the model switched, capture silenced).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_deckCut<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    why: JString<'l>,
) {
    let why: String = env.get_string(&why).map(Into::into).unwrap_or_default();
    handle(h).deck.cut(&why);
}
