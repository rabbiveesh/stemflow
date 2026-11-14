# stemflow on Android (prototype)

See [PLAN.md](PLAN.md) for the feasibility plan. This folder never changes the desktop build.

## bench/ — on-device separator timing

A Rust CLI that `#[path]`-includes the desktop's `src/spec.rs`, `src/model.rs` and `src/stream.rs`, so the timings are for the exact code the app would run. It runs over `adb shell`; no APK is needed.

Setup (once):
- NDK r29 unpacked at `~/Android/android-ndk-r29`.
- `onnxruntime-android-1.31.1.aar` from Maven Central, unzipped at `~/Android/ort-android-1.31.1`.
- `rustup target add aarch64-linux-android && cargo install cargo-ndk`

```bash
cd android/bench && ANDROID_NDK_HOME=~/Android/android-ndk-r29 cargo ndk -t arm64-v8a -P 26 build --release
```

```bash
adb push target/aarch64-linux-android/release/stemflow-bench ~/Android/ort-android-1.31.1/jni/arm64-v8a/libonnxruntime.so clip.wav /data/local/tmp/stemflow/
```

```bash
adb push ../../models/ft2/htdemucs_ft_bass.onnx /data/local/tmp/stemflow/ft2_bass.onnx
```

```bash
adb shell "cd /data/local/tmp/stemflow && ORT_DYLIB_PATH=./libonnxruntime.so ./stemflow-bench ft2_bass.onnx --wav clip.wav --threads 4 --runs 12"
```

`--out bass.wav` writes the isolated stem, so you can `adb pull` it and listen.

## Results

**Pixel 10a** (Tensor G4: 1× X4 3.1 GHz, 3× A720 2.6 GHz, 4× A520 1.95 GHz; 7.4 GB RAM; Android 17), on battery, Spotify playing in the background.

Common settings: ft2 bass (htdemucs_ft fine-tune), fp32, ONNX Runtime 1.31.1 CPU EP (default MLAS), 7.8 s window, 30 s of *If I Were a Bell*.

| config | per run p50 | min | peak RSS |
|---|---|---|---|
| 4 threads | 5.2 s | 4.2 s | 3.4 GB |
| 8 threads | 7.1 s | 6.8 s | 4.2 GB |
| 4 threads pinned to big cores (taskset f0) | 8.0 s | 5.6 s | 4.2 GB |
| 3 threads pinned to A720s (taskset 70) | 9.0 s | 6.4 s | 3.8 GB |

The run-to-run spread is large; the first run took 17.8 s. The phone stayed cool (thermal status 0), so the likely cause is memory pressure: 3–4 GB RSS on a 7.4 GB phone means zram.

**Conclusion:** CPU fp32 is about **0.2× real time**, roughly 5× short of a 1 s hop. A hop long enough to keep up (about 6 s) would mean about 13 s of latency and no safety margin. CPU alone is not enough; see PLAN.md §2 for the options.

## LiteRT on the phone GPU

`tools/export_litert.py` converts an htdemucs_ft specialist (stock, or a fine-tune checkpoint such as `out/ft2/step600.th`) from PyTorch to LiteRT (`.tflite`) with `litert-torch`. Env: `uv venv -p 3.11` + `litert-torch demucs==4.0.1 torch torchaudio` (CPU wheels), kept at `~/Android/litert-env`.

```bash
cd android/tools && ~/Android/litert-env/.venv/bin/python -I export_litert.py ~/Android/litert-models bass 7.8 ../../out/ft2/step600.th
```

The bench runs `.tflite` models through LiteRT's C API (`bench/src/litert.rs`), with the desktop's STFT and streaming around it. It links the two `.so` files from Google's `litert` and `litert-gpu` 1.4.2 AARs (unzipped at `~/Android/litert-1.4.2`): build with `LITERT_LIB_DIRS=<litert>/jni/arm64-v8a:<litert-gpu>/jni/arm64-v8a`, push both `.so` files next to the binary, and run with `LD_LIBRARY_PATH=.`:

```bash
adb shell "cd /data/local/tmp/stemflow && LD_LIBRARY_PATH=. ./stemflow-bench ft2_bass_344064.tflite --gpu fp16 --wav clip.wav --hop 1.75 --out bass.wav"
```

### What the export changes, and why

All of it is mathematically the same model (torch vs stock forward 2e-8, LiteRT desktop CPU vs torch 2e-7):

| change | reason |
|---|---|
| embeddings baked in as constants; LayerScale folded into the linears | traced, they become ops the GPU delegate rejects (BROADCAST_TO, PAD, SELECT, broadcast MUL) |
| transformer on `[1, C, 1, tokens]` with 1×1 convolutions | the delegate rejects or mis-batches `[tokens, C]` / `[1, tokens, C]` |
| segment rounded to a multiple of 1024 samples | no padding in the time branch |
| channel means as 1×1 convolutions with weights 1/C | `x.mean(dim=1)` becomes MEAN → 3-D tensor → RESHAPE, which the delegate misreads: **NaN** |
| every norm rescales by max\|d\|/128 before squaring | fp16 tops out at 65504 and norm inputs reach ~1900 on real music: **silent garbage in fp16** |

With these, all 2841 ops run in one GPU partition.

### Results (Pixel 10a, ft2 bass, 30 s of *If I Were a Bell*, mono fold)

| config | per run (p50) | hop it keeps up with | start-up | peak RSS |
|---|---|---|---|---|
| GPU fp16, 7.8 s window | 1.61–1.63 s | 1.75 s (0 of 12 runs late) | 7–11 s | 1.1 GB |
| GPU fp16, 4 s window | 0.78 s | 1 s (0 of 31 runs late) | 10 s | 0.9 GB |
| GPU fp32, 7.8 s window | 2.34 s | – | – | 1.6 GB |

Accuracy of the stem against the same model on the phone's CPU:

| config | SNR | worst second |
|---|---|---|
| GPU fp32, 7.8 s | 55.7 dB | 50.4 dB |
| GPU fp16, 7.8 s | 45.3 dB | 35.8 dB |
| GPU fp16, 4 s (vs the 7.8 s reference) | 21.3 dB | 12.5 dB |

The 4 s number is not an error measure: a shorter window is a different (less informed) estimate. Whether it sounds worse needs listening or an SDR test.

Notes:
- Before the fp16 rescaling the 7.8 s window took 1.24 s; the rescaling costs ~0.4 s. GPU time by op type: norm statistics (mean + reduce_max) 28%, reshape/transpose/slice 22%, the convolutions themselves only 18%. So there is room: calibrated constant scales instead of reduce_max, and writing the encoders/decoders in 4-D so the converter stops inserting reshapes.
- `inference_preference` must stay FAST_SINGLE_ANSWER. SUSTAINED_SPEED spent 218 s (fp16) / 635 s (fp32) tuning at start-up for no faster runs.
- LiteRT on the CPU through this C API took ~48 s per run (the prebuilt benchmark tool: 3.5 s), so XNNPACK is evidently not being applied there. Unimportant for the GPU path; not investigated.
- Random-input checks (`inference_diff`) did not catch the fp16 overflow. Only real audio did.

## app/ and core/ — the live app

`core/` is a Rust `cdylib` (JNI) holding the separator on LiteRT/GPU (`bench/src/litert.rs`), the desktop's STFT and sliding-window streaming (`src/spec.rs`, `src/stream.rs`, by path), and a mixer with the desktop's loudness curve and limiter. `app/` is a Kotlin app (no AndroidX):

- **Main screen**, modelled on the GNOME menu: Bass isolation switch, what the mix is, the Solo ↔ No bass balance line (detent at the full mix), Boost to 200%, the Bass solo / Full mix / No bass snaps, a status line.
- **Notification** with the three snaps, and a **Quick Settings tile** for on/off ("Add quick tile" on the main screen).
- **Sources:** the shell-uid helper (default; works for Spotify, see below), or Android's screen-capture consent ("Use screen capture instead"; only for apps that allow capture, and the original keeps playing).
- **Threads:** reader (source → 2 s hops) → worker (separator) → player (20 ms blocks, levels applied at playback so the controls act at once). When the worker falls behind it skips to the newest hop.

Setup (once): Android SDK at `~/Android/sdk` (cmdline-tools, `platforms;android-35`, `build-tools;35.0.0`), Gradle 8.11.1 at `~/Android/gradle-8.11.1`, `app/local.properties` with `sdk.dir=`, and the two LiteRT `.so` files copied into `app/src/main/jniLibs/arm64-v8a/`.

```bash
cd android/core && ANDROID_NDK_HOME=~/Android/android-ndk-r29 LITERT_LIB_DIRS=~/Android/litert-1.4.2/litert/jni/arm64-v8a:~/Android/litert-1.4.2/litert-gpu/jni/arm64-v8a cargo ndk -t arm64-v8a -P 26 -o ../app/src/main/jniLibs build --release
```

```bash
cd android/app && ~/Android/gradle-8.11.1/bin/gradle --no-daemon -q assembleDebug && adb install -r build/outputs/apk/debug/stemflow-debug.apk
```

The model goes into the app's private files (the newest `.tflite` there is used):

```bash
adb push ~/Android/litert-models/ft2_bass_344064.tflite /data/local/tmp/ && adb shell "run-as org.stemflow.app sh -c 'mkdir -p files && cp /data/local/tmp/ft2_bass_344064.tflite files/'; rm /data/local/tmp/ft2_bass_344064.tflite"
```

Switch it on in the app (or `adb shell am start-foreground-service -n org.stemflow.app/.CaptureService`), wait for "Waiting for the capture helper", then start the helper (command in the helper section below). The mix can also be driven over adb:

```bash
adb shell am broadcast -a org.stemflow.app.CMD -p org.stemflow.app --es snap full
```

**First live run (2026-11-14, Pixel 10a, ft2, 7.8 s window, 2 s hop, GPU fp16):** Spotify → helper → separator → mix → speaker works. The model loads in 11–13 s. Steady runs take 1.55–1.75 s (the bench: 1.61 s); the first few runs after start, and runs while the UI is being used or the phone is warm (thermal status 1), took 2.2–4.2 s and trigger the skip. So a 2 s hop has too little headroom on this phone until the model is faster.

While the source is paused (digital silence) the separator is skipped, so the GPU idles; the first run after resuming is slow (about 4 s, the GPU has clocked down) and is absorbed by the skip.

Not done: karaoke and model switching (need more models on the phone), Shizuku, releasing the model's ~1 GB when switched off.

## Audio-in findings

**Spotify blocks playback capture.** Measured 2026-11-14 on the Pixel 10a with Spotify 9.1.94 playing: the probe records digital silence (−120 dBFS). `dumpsys media.audio_policy` shows Spotify's track with `Flags: 0x600` = DEEP_BUFFER | **NO_MEDIA_PROJECTION** (0x400), i.e. `ALLOW_CAPTURE_BY_SYSTEM`. The same probe captures Musicolet at −24 dBFS, so the app is fine.

Two things that looked like "capturable" and are not evidence:
- the manifest flag `ALLOW_AUDIO_PLAYBACK_CAPTURE` (Spotify has it; the player overrides it per stream),
- `dumpsys audio`'s `AudioPlaybackConfiguration ... flags=0x0` (it doesn't show the capture policy; `dumpsys media.audio_policy` does).

**With shell privileges Spotify is capturable, and the original goes silent.** Tested with scrcpy 5.0.1 (its server runs as the shell uid over adb), recording 10 s of Spotify to WAV:

| scrcpy `--audio-source` | how it captures | result |
|---|---|---|
| `playback` | AudioPolicy mix, `MIX_TYPE_PLAYERS`, route flag `LOOP_BACK` only, matching usages | **−15.8 dBFS, Spotify's music**, taken before the volume control |
| `output` | REMOTE_SUBMIX device | −52 dBFS: it is post-volume, and media volume was at 1 of 25 |

The `playback` mix has no `RENDER` flag, so matched audio is rerouted to the capturer instead of the speaker: the same "move the stream" behaviour as the desktop. (Confirmed from the policy dump; not yet confirmed by ear.) So the app design is: a small helper running as shell uid (through Shizuku, or started over adb for development) registers that mix for Spotify, excluding our own uid, and hands PCM to the app, which plays the remix normally. scrcpy's `AudioPlaybackCapture.java` (Apache-2.0) is the reference for the hidden-API calls.

### The helper (`app/src/main/java/org/stemflow/app/Helper.java`)

Our own shell-uid capturer, in the app's APK but run outside the app. It registers a `LOOP_BACK`-only mix matching the target app's uid and streams 44.1 kHz stereo PCM to the app over a loopback TCP socket (port 28711); the app needs no MediaProjection consent with this source. For development it is started over adb, once the app says it is waiting:

```bash
android/helper.sh
```

**Works end to end** (2026-11-14): Spotify arrives in the app at −20 dBFS and the app plays its own version while the original is off the speaker (the user confirmed by ear, with scrcpy's identical mix, that the original goes silent).

Findings:
- Do not call `allowPrivilegedPlaybackCapture(true)`: `AudioMix` then rejects sample rates above 16 kHz. It isn't needed for Spotify.
- When the helper exits and the mix is dropped, Spotify pauses itself (it sees its output route change), so the app sends a media Play key a second after it is switched off.
- Killing the helper (or the app closing the socket) is enough to restore normal playback.

Not yet tested: YouTube Music; Shizuku as the way to start the helper without adb.
