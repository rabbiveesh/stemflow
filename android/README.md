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

All of it is mathematically the same model (torch vs stock forward 4e-8, LiteRT desktop CPU vs torch 2e-7):

| change | reason |
|---|---|
| embeddings baked in as constants; LayerScale folded into the linears and norms | traced, they become ops the GPU delegate rejects (BROADCAST_TO, PAD, SELECT, broadcast MUL) |
| every tensor 4-D with batch 1: time branch `[1, C, 1, T]`, frequency branch `[1, C, Fr, T]`, transformer `[1, C, 1, tokens]`; encoders, decoders and DConv re-written with 2-D convolutions (DConv's norms take one mean/variance per frequency row) | the delegate rejects or mis-batches `[tokens, C]` / `[1, tokens, C]`, and demucs' own 3-D layers made the converter insert 318 RESHAPE, 260 TRANSPOSE and 106 SLICE ops that each copy the tensor (22% of the GPU time) |
| segment rounded to a multiple of 1024 samples; the decoders crop by padding the transposed convolution | no padding or slicing in the time branch |
| the last decoders compute only the wanted source's channels | no 5-D source axis, and a quarter of the last layer's work |
| channel means as 1×1 convolutions with weights 1/C | `x.mean(dim=1)` becomes MEAN → 3-D tensor → RESHAPE, which the delegate misreads: **NaN** |
| means and maxima over a row or a whole tensor taken as a chain of poolings, ≤ 64 values per step | one MEAN / REDUCE_MAX over millions of values is a single serial GPU work item: 14 ms each for the first encoder's tensors, 28% of the GPU time in all. Pooling accumulates in fp32, so this is safe in fp16 |
| attention one head at a time, each product a 1×1 convolution whose weights are the other operand (`scores = conv(q, weight=k)`, softmax over channels, `conv(scores, weight=v)`) | as matmuls the converter emits `[heads, T, d] @ [heads, d, S]`, and the delegate reshapes any matmul with batch ≠ 1, copying the heads×T×S scores three times (35 of 65 ms per attention). Per-head matmuls become FULLY_CONNECTED with run-time weights, which aborts |
| every norm rescales by max\|d\|/128 before squaring | fp16 tops out at 65504 and norm inputs reach ~1900 on real music: **silent garbage in fp16** |

With these, all 3335 ops run in one GPU partition (7.8 s window; 3308 for 4 s).

### Results (Pixel 10a, ft2 bass, 30 s of *If I Were a Bell*, mono fold)

| config | per run (p50) | was, before the 4-D rewrite | hop it keeps up with | start-up | peak RSS |
|---|---|---|---|---|---|
| GPU fp16, 7.8 s window | 0.96–0.98 s | 1.65 s | 1 s, barely (0–1 of 31 runs late, worst 1.002 s); 1.75 s easily | 5.5 s | 0.93 GB |
| GPU fp16, 4 s window | 0.54 s | 0.80 s | 1 s (0 of 31 runs late) | 5.5 s | 0.83 GB |
| GPU fp32, 7.8 s window | 1.96 s | 2.34 s | – | 6 s | 1.3 GB |

Of the 0.96 s, the GPU takes 0.80 s (Google's `benchmark_model`: 0.80 s; 0.42 s for the 4 s window); the rest is CPU: STFT 35 ms, iSTFT 80 ms, copies. Pinned to the X4 core (`taskset 80`) the run takes 0.90 s, so the app's worker thread should ask for a big core.

Accuracy of the fp16 stem against the same model in fp32 on the GPU (which matched the phone's CPU to 55.7 dB before the rewrite):

| config | SNR | worst second |
|---|---|---|
| GPU fp16, 7.8 s | 46.8 dB | 39.6 dB |
| GPU fp16, 4 s | 47.3 dB | 39.6 dB |
| previous export, fp16, 7.8 s (vs the new one in fp32) | 46.6 dB | 38.8 dB |

A 4 s window is a different (less informed) estimate than a 7.8 s one: the two differ by 21 dB SNR. Whether it sounds worse needs listening or an SDR test.

Notes:
- Tried and dropped, as none was faster: min/max instead of abs for the norm's scale (REDUCE_MIN: 1.04 s on the GPU instead of 0.80 s); splitting the rewrite convolutions so the GLU needs no SLICE; pooling steps of 32 or 512; multiplying by 1/s instead of dividing; STFT/iSTFT of the two channels on two threads.
- Constant, calibrated norm scales instead of max\|d\| would save at most 60 ms: the export without any fp16 protection runs in 0.73 s on the GPU. Not worth an overflow that depends on the music.
- `benchmark_model --enable_op_profiling` aborts on the new graph (it worked on the old one), so there is no per-op profile of it. Not investigated.
- The LiteRT 2.x GPU accelerator (ML Drift) is untried.
- `inference_preference` must stay FAST_SINGLE_ANSWER. SUSTAINED_SPEED spent 218 s (fp16) / 635 s (fp32) tuning at start-up for no faster runs.
- LiteRT on the CPU through this C API took ~48 s per run (the prebuilt benchmark tool: 3.5 s), so XNNPACK is evidently not being applied there. Unimportant for the GPU path; not investigated.
- Random-input checks (`inference_diff`) did not catch the fp16 overflow. Only real audio did.

## app/ and core/ — the live app

`core/` is a Rust `cdylib` (JNI) holding the separator on LiteRT/GPU (`bench/src/litert.rs`), the desktop's STFT and sliding-window streaming (`src/spec.rs`, `src/stream.rs`, by path), and a mixer with the desktop's loudness curve and limiter. `app/` is a Kotlin app (no AndroidX):

- **Main screen**, modelled on the GNOME menu: Bass isolation switch, what the mix is, the Solo ↔ No bass balance line (detent at the full mix), Boost to 200%, the Bass solo / Full mix / No bass snaps, a status line.
- **Karaoke switch and Model picker**, as in the menu: models are the `<name>_<stem>_<samples>.tflite` files in the app's files; Karaoke needs a `vocals` one and switches the target (labels follow: "Vocals solo", "No vocals"). A model picked while running is swapped in live (5–8 s to load on a Pixel 10a; the old one is freed first, so memory stays near 0.9 GB). Each target remembers its own levels: bass starts at Bass solo, karaoke at No vocals.
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

### Shizuku

With [Shizuku](https://shizuku.rikka.app/) installed and started on the phone (over wireless debugging, once per boot), the app starts the helper itself: `Shizu.kt` binds `HelperService` as a Shizuku user service, which runs `Helper.run` as the shell uid. The first switch-on asks for Shizuku's permission. Without Shizuku the app waits for `android/helper.sh` as before.

**Works** (2026-11-15, Shizuku 13.6.0 on Android 17): switch on → helper starts through Shizuku → Spotify is separated, no computer involved. One trap: under Shizuku the process's `ActivityThread` reports the app's own package, and the audio server refuses to create the `AudioRecord` for a package that doesn't belong to the shell uid (`EX_SECURITY` in `ValidatedAttributionSourceState`). `Helper.fakeContext` therefore points the thread's bound application at `com.android.shell` in both the adb and the Shizuku case.

**With the faster models (same day):** ft2, 7.8 s window, 1.5 s hop: **0.87–0.94 s per run** in the app, none late over the first 13 runs. Models are freed when isolation is switched off or another model is picked (`engineFree`, `Drop for LiteRtDemucs`).

**Karaoke (same day):** `htdemucs_ft_vocals_344064.tflite`, exported with `export_litert.py OUT vocals 7.8` (stock weights), runs at 0.82–0.92 s per 1.5 s hop. Its fp16 output has not been compared numerically against fp32 as the bass model's was.

Not done: a timeout when the helper fails to start (the status just stays on "Starting the capture helper…").

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
