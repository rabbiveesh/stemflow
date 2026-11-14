# stemflow for Android — feasibility plan

Status: draft, decisions marked **[decide]** are open.

## 1. Audio in

| Route | Works today? | Notes |
|---|---|---|
| `AudioPlaybackCaptureConfiguration` (Android 10+) | Yes, for apps that allow it | Needs a MediaProjection consent dialog each session + a foreground service of type `mediaProjection`. Captures only `USAGE_MEDIA/GAME/UNKNOWN`. Apps targeting API 29+ are capturable **unless** they set `allowAudioPlaybackCapture="false"` or a player capture policy of `ALLOW_CAPTURE_BY_SYSTEM/NONE` — then we get silence. |
| Spotify | **Not for a normal app; yes with shell privileges** (both measured 2026-11-14) | Its stream carries NO_MEDIA_PROJECTION, so MediaProjection capture records silence. A capturer running as the shell uid (scrcpy's `playback` source) gets it at full level and the original is rerouted away from the speaker. Plan: a shell-uid helper via Shizuku. |
| YouTube Music | Manifest allows it | Player policy not checked yet (it wasn't playing). |
| We are the player (local files: FLAC/MP3 in a folder, or a share-to-stemflow intent) | Always works | Strong fallback, and it's also the simplest prototype path (no consent dialog, no latency constraint — can even separate ahead of playback). |
| Bluetooth/USB loopback | Not practical | Android can't be an A2DP sink without root/custom ROM; USB audio-in from a laptop defeats the point. |

**Second big risk: muting the original.** Capture is a *copy*: the source app keeps playing to the speaker. On desktop we move the stream; Android has no public API to silence another app.

Unknowns to test on the device:
- **Spotify volume:** does lowering Spotify's own volume ("volume normalisation"/in-app level) also lower what we capture?
- **Output usage:** can our output use a different `AudioAttributes` usage whose volume stream is separate from media?

If neither works, the "we are the player" route becomes the main one, not just a fallback.

## 2. Inference on the phone

Reality check on compute: Demucs runs on a **7.8 s window** (`HTDEMUCS_SEGMENT`), BS-RoFormer SW on 3 s. On the desktop: 0.21 s (Demucs) and 0.62 s (RoFormer) per run on an RTX 4060.

**Measured on the Pixel 10a** (Tensor G4, see README): ft2 fp32 on the ORT CPU EP takes **4.2–5.2 s per 7.8 s window** (4 threads), with 3.4 GB peak RSS. That's 0.2× real time, so the CPU does **not** keep up with a 1 s hop. Ways out, in order of effort:

1. **Bigger hop.** Hop = compute time + margin. A 4 s hop gives about 5–6 s latency.
   - Latency barely matters for play-along: you play to what you *hear*, which is our delayed output.
   - It only shows at start, pause and seek.
2. **fp16 / int8 weights** (174 MB → 87 MB / 44 MB). int8 dynamic quant on CPU (XNNPACK) is ~1.5–2× faster; quality must be checked against `stemflow verify`.
3. **GPU/NPU.** *Measured (README):* LiteRT on the Mali GPU in fp16, with the export rewritten for the delegate, produces a correct ft2 stem from real audio (45 dB SNR vs CPU). The full 7.8 s window takes 1.6 s per run (keeps up with a 1.75 s hop); a 4 s window takes 0.78 s (keeps up with a 1 s hop). Open: the quality cost of the 4 s window, and trimming the 7.8 s run (half its GPU time is norm statistics and reshapes, not convolutions).
   - ONNX Runtime: NNAPI is deprecated since Android 15 and weak on Tensor. QNN is Qualcomm-only, so it doesn't apply to Pixel's Tensor.
   - **LiteRT** (TFLite successor) has the GPU delegate (ML Drift) and Google's Tensor NPU path. That means converting ONNX → LiteRT: ai-edge-torch from the PyTorch model is the cleaner route, since we have the export scripts.
   - Higher ceiling, more work, so measure CPU first.
4. **Shorter Demucs window** (re-export at e.g. 4 s). That makes inference about 2× cheaper, at some quality cost.

Licensing: ship **Demucs only** (htdemucs_ft + ft1–ft3, MIT weights). BS-RoFormer SW stays desktop/dev-only.

Memory: one Demucs model at fp32 is ~1 GB peak, which is OK on an 8 GB phone. Load **one** model at a time, never the ensemble.

## 3. Code reuse

Reuse the Rust core. `spec.rs`, `stream.rs` and `model.rs` only depend on `realfft`, `ort` and `anyhow`. `ort` with `load-dynamic` can load `libonnxruntime.so` from the official `onnxruntime-android` AAR. Only `audio.rs` (PulseAudio) and `control.rs`'s D-Bus half are desktop-only.

- **Now (prototype):** `android/core` is a separate crate that `#[path]`-includes `src/spec.rs`, `src/model.rs` and `src/stream.rs`. That's zero changes to the desktop crate.
- **Later [decide]:** split the repo into a cargo workspace with `stemflow-core` (lib: spec, model, stream, mix/limiter) + the desktop bin. The desktop *behaviour* stays the same, but the files move.
- **JNI vs uniffi:** the Kotlin↔Rust surface is tiny (`start(model, fd)`, `setBalance`, `setBoost`, `setTarget`, `stats()`). Hand-written JNI via the `jni` crate, or uniffi if we'd rather have generated bindings. **[decide]**, leaning towards uniffi for less boilerplate.
- Audio I/O in Kotlin (`AudioRecord` with the capture config → ring buffer → Rust; Rust → `AudioTrack`). Or Oboe/AAudio in Rust, but capture must start from Java because of MediaProjection anyway.

## 4. UX (from `gnome/stemflow@veesh/extension.js`)

- **Main screen:** one balance slider (bass solo ← full → no bass) with snap buttons Solo / Full / No bass, a Boost slider 0–200% (limiter as on desktop), a Karaoke switch (target = vocals), and a model picker (htdemucs_ft, ft1, ft2, ft3).
- **Foreground-service notification** (required anyway for capture): a MediaStyle notification with actions Solo / Full / No bass + Stop, and the current model in the subtitle.
- Later: a Quick Settings tile for on/off.

## Prototype order (smallest thing that kills the biggest risk first)

0. **No code:** screen-record Spotify with device audio on the phone. That answers the "can we capture Spotify" question.
1. **`android/bench`:** a Rust CLI cross-compiled for `aarch64-linux-android`, using the desktop's own `model.rs`, `spec.rs` and `stream.rs`.
   - It runs the `File` pipeline on a WAV, pushed and run over `adb shell`, and prints ms per model run.
   - No APK, Gradle or SDK needed, just the NDK + the ORT Android `.so`.
   - Try CPU threads 4/8, fp32 vs fp16 vs int8, and XNNPACK EP.
   - It tells us the hop (= latency) we can afford.
2. **Minimal APK:** capture via MediaProjection, then the Rust core, then `AudioTrack`, with one hard-coded model and the balance slider.
3. Then the UI/notification, model switching and karaoke.
