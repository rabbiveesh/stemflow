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
- The LiteRT 2.x GPU accelerator (ML Drift) runs this model 14% faster: see below.
- `inference_preference` must stay FAST_SINGLE_ANSWER. SUSTAINED_SPEED spent 218 s (fp16) / 635 s (fp32) tuning at start-up for no faster runs.
- LiteRT on the CPU through this C API took ~48 s per run (the prebuilt benchmark tool: 3.5 s), so XNNPACK is evidently not being applied there. Unimportant for the GPU path; not investigated.
- Random-input checks (`inference_diff`) did not catch the fp16 overflow. Only real audio did.

### BS-RoFormer SW on the phone GPU

`tools/export_bsrofo_litert.py` does the same for one head of BS-RoFormer SW (the model that won the desktop's blind tests; its weights are dev-only, see the licensing notes). It needs `rotary-embedding-torch beartype pyyaml` in the env, and the MSST checkout and `models/ext/bsrofo/` of the main checkout:

```bash
cd android/tools && ~/Android/litert-env/.venv/bin/python -I export_bsrofo_litert.py ../.. ~/Android/litert-models vocals 3
```

The bench picks `LiteRtRoformer` for files named `bsrofo*` (input and output are the spectrogram `[1, 4, 1025, frames]`, STFT as on the desktop). `--hop` plus `--lookahead` must fit in the window, which is 2.995 s for "3": `--hop 2 --lookahead 0.9`.

Results (Pixel 10a, fp16, 3 s window, 30 s of *If I Were a Bell*, bass head, mono fold):

| step | per window |
|---|---|
| first export that runs entirely on the GPU | 3.22 s |
| attention matmuls kept 4-D (`tools/tflite_matmul4d.py`) | 1.91 s |
| all convolutions on the `[1, D, bands, T]` grid | 1.81 s on the GPU, 1.89 s in the bench |

It keeps up with a 2 s hop (0 of 16 runs late, worst 1.97 s) and not with less. fp32 takes 4.1 s. Peak RSS 1.1 GB. The stem matches the desktop's RoFormer (ONNX, CUDA, same window and hop) to 68.4 dB SNR in fp32 and 46.9 dB in fp16 (worst second 36.2 dB).

What the export changes (torch vs stock forward 3e-6, LiteRT desktop vs torch 4e-4 at a peak of 209):

| change | reason |
|---|---|
| token grid `[1, D, bands, T]`, linears as 1×1 convolutions; band split as one `[bins, 1]` convolution per band; all masks laid out by one reshape and transpose | no 3-D tensors, no per-band reshapes |
| attention one head at a time, as matmuls over `[1, rows, N, 64]`, and `tflite_matmul4d.py` rewrites the file so they stay 4-D | the converter makes every matmul 3-D with batch = rows; the delegate then copies the scores three times: 67 of 133 ms per transformer |
| rotary embedding as constants, with q and k produced as their even and odd halves | the rotation becomes elementwise |
| RMSNorm divides by the token's peak before squaring (and gamma is folded into the next linear) | fp16 overflow. The division must not be a multiplication by the reciprocal: on silence that is 0 × inf, **NaN** |
| constants made contiguous | the converter silently gave a clone of a permuted tensor the data of another constant of the same shape: **wrong output, no error** |

Notes:
- Where the time goes, per transformer (24 of them), from ablations at depth 2: feed-forward 22 ms, matmuls and softmax 13 ms, and the rest of attention (its convolutions, rotation and transposes) about 37 ms by difference; the band split and mask head take 62 ms in all. The norms cost nothing measurable.
- Convolutions run at about 370 GMAC/s on the bands × frames grid, 30% slower on frames × bands, and slower on flat or other grids (1×16058, 7, 14 or 31 rows). Demucs' widest layers reach about 470 GMAC/s, so the floor for this model (404 GMAC of convolutions per 3 s) is about 1.3 s.
- Untried: a shorter window (cost is close to linear in its length).
- `benchmark_model --enable_op_profiling` aborts on this graph too.

### LiteRT 2.x GPU accelerator (ML Drift)

The bench can run on LiteRT 2.3.2's `CompiledModel` and GPU accelerator instead of 1.4.2's interpreter and GPU delegate: build it with the feature `litert2` (the two runtimes export clashing symbols, so it is one or the other). It needs `libLiteRt.so` and `libLiteRtClGlAccelerator.so` from the `litert` and `litert-gpu` 2.3.2 AARs on Google's Maven, unzipped at `~/Android/litert-2`; both go next to the binary on the phone. The same `.tflite` files run on either.

```bash
cd android/bench && ANDROID_NDK_HOME=~/Android/android-ndk-r29 LITERT_LIB_DIRS=~/Android/litert-2/litert/jni/arm64-v8a cargo ndk -t arm64-v8a -P 26 build --release --features litert2 --target-dir target/litert2
```

Per run in the bench, fp16, Pixel 10a, same clip and settings as above:

| model | LiteRT 1.4.2 | LiteRT 2.3.2 (ML Drift) | fp16 accuracy on ML Drift |
|---|---|---|---|
| Demucs ft2 bass, 7.8 s window | 0.96 s (GPU alone 0.80 s) | 0.865 s (GPU alone 0.69 s); 0 of 31 runs late at a 1 s hop, worst 0.875 s | 46.8 dB vs its own fp32, which matches 1.4.2's fp32 to 127 dB |
| BS-RoFormer SW bass, 3 s window | 1.89 s (GPU alone 1.81 s) | 1.75 s (GPU alone 1.64 s); 0 of 16 runs late at a 2 s hop, worst 1.82 s | 47.3 dB vs the desktop RoFormer |

Start-up is 4 s for Demucs (7–11 s before). fp32 Demucs takes 1.45 s. Peak RSS 0.9 GB (Demucs), 1.0 GB (RoFormer). Four load → run → drop cycles of RoFormer return to the same RSS each time.

Notes:
- GPU options go in as an opaque option named `gpu_options` whose payload is TOML (`precision = 1` for fp16, `2` for fp32); the `Lrt*GpuOptions` helpers are not exported by `libLiteRt.so`. `buffer_storage_type`, `wait_type` and `prefer_texture_weights` made RoFormer slower or no faster.
- The accelerator library is found through the dynamic linker's search path (`LD_LIBRARY_PATH=.` over adb). Inside an APK that has not been tried; if it is not found there, the environment option `kLiteRtEnvOptionTagRuntimeLibraryDir` (tag 22, a string) names the directory.
- Every start prints "NPU accelerator could not be loaded"; harmless.
- `tools/mldrift_run.c` is the minimal C runner used for the first measurements and the NPU probes. It needs the C headers of the LiteRT repo at tag `v2.3.2` (`litert/c/`, copied to `~/Android/litert-2/include`, with an empty `litert/build_common/build_config.h`): `aarch64-linux-android26-clang -O2 -I ~/Android/litert-2/include tools/mldrift_run.c -o mldrift_run -L ~/Android/litert-2/litert/jni/arm64-v8a -lLiteRt`.

**Tensor NPU (tried, not usable on-device).** `ACCEL=npu` in `mldrift_run` selects it; it needs the two `.so` files of `litert-npu-runtime-google-tensor` 2.3.2 beside the binary and `LD_LIBRARY_PATH=.:/vendor/lib64` (the phone has `libedgetpu_litert.so`). From the shell the on-device compiler does run:

| probe on `[1, 256, 62, 259]` | NPU | GPU (ML Drift, fp16) |
|---|---|---|
| `tanh(x) * sigmoid(x)` | compiles in 33 s, runs in 7 ms | ~1 ms on the old delegate for a comparable op |
| two 1×1 convolutions + GELU (a RoFormer feed-forward) | the compile service dies (`DEAD_OBJECT`), 5 of 5 tries | 26 ms |
| slice + concat; RMS norm | the compile service dies | – |
| attention (matmul + softmax) | "unrecoverable compilation errors" | 11–13 ms |
| the whole Demucs 4 s model | all 3308 ops selected, then "unrecoverable compilation errors" | – |

The system service (`vendor.google.edgetpu_vendor_service`) logs "DarwiNN compiler virtual memory limit set to 4096MB" and exits a few seconds into each compile, with no reason visible to the shell. Google's supported route is compiling ahead of time on a Linux desktop (`ai_edge_litert.aot` with the pip package `ai-edge-litert-sdk-google-tensor`, which downloads the Tensor ML SDK and its compiler from Google under their licence agreements). Tried on 2026-11-15 with the owner's go-ahead: the install fails, because the compiler archive it downloads (`tensor-ml-sdk/sdk/releases/6.1.0/litert_plugin_compiler.tar.gz`) returns 404, and earlier package versions only accept a tarball you already have. Google's page says the SDK is a beta you sign up for, so the NPU waits on that access.

### Beat This! (beat tracker) on LiteRT

`tools/export_beat_this_litert.py` converts Beat This! (CPJKU, MIT) so the beat click needs no ONNX Runtime in the APK. Same contract as the desktop's ONNX files (`tools/export_beat_this.py`): input log-mel `[1, 1500, 128]` (30 s at 50 fps), outputs beat and downbeat logits `[1, 1500]`, in that order. Same env as the other exports; `beat_this` itself is taken from the main checkout's `tools/.venv-gpu`:

```sh
cd android/tools && ~/Android/litert-env/.venv/bin/python -I export_beat_this_litert.py ~/Android/litert-models ../../tools/.venv-gpu/lib/python3.10/site-packages ../../models/beats
```

| model | file | LiteRT vs PyTorch, max abs logit error (reference clip, logits peak 11–13) | laptop CPU, 4 threads |
|---|---|---|---|
| small0 | `beat_this_small0_1500.tflite`, 9.0 MB | beat 2.5e-5, downbeat 2.5e-5 | 0.39 s |
| final0 | `beat_this_final0_1500.tflite`, 81.6 MB | beat 8.7e-5, downbeat 5.7e-5 | 0.64 s |

The one change to the model: the attention is written out in the export script, because `rotary_embedding_torch` and Beat This!'s `SumHead` wrap their maths in `torch.autocast`, which `torch.export`'s decompositions refuse ("expected compiled_fn to be GraphModule"). The rotary angles become constants per sequence length. The rewrite gives bit-identical output to the stock forward in PyTorch.

It is meant for the CPU (XNNPACK): the GPU belongs to the separator. The graph was not reshaped for the GPU delegate (3-D tensors, time folded into the batch), so the GPU is untried.

`LiteRtBeatNet` in `bench/src/litert.rs` loads it: `load(path, Accel::Cpu, threads)`, `run(&spect) -> (beat, downbeat)`. On LiteRT 2 the CPU thread count now reaches XNNPACK (opaque options named `xnnpack`, TOML `num_threads = N`; it was ignored before). The bench times it and checks it against a reference folder (`spect.f32`, `beat.f32`, `downbeat.f32`, raw f32), one run every `--hop` seconds as the app would:

```sh
adb push ~/Android/litert-models/beat_this_small0_1500.tflite ../../models/beats/ref_beat_this_small0 /data/local/tmp/stemflow/
adb shell "cd /data/local/tmp/stemflow && LD_LIBRARY_PATH=. ./stemflow-bench beat_this_small0_1500.tflite --beat-ref ref_beat_this_small0 --threads 2 --hop 3 --runs 10"
```

**Phone timing: not measured yet** (the phone was off Wi-Fi on 2026-11-16).

## app/ and core/ — the live app

`core/` is a Rust `cdylib` (JNI) holding the separator on LiteRT/GPU (`bench/src/litert.rs`), the desktop's STFT and sliding-window streaming (`src/spec.rs`, `src/stream.rs`, by path), and a mixer with the desktop's loudness curve and limiter. `app/` is a Kotlin app (no AndroidX):

- **Main screen**, modelled on the GNOME menu: Bass isolation switch, what the mix is, the Solo ↔ No bass balance line (detent at the full mix), Boost to 200%, the Bass solo / Full mix / No bass snaps, a status line.
- **Karaoke switch and Model picker**, as in the menu: models are the `<name>_<stem>_<samples>.tflite` files in the app's files; Karaoke needs a `vocals` one and switches the target (labels follow: "Vocals solo", "No vocals"). A model picked while running is swapped in live (5–8 s to load on a Pixel 10a; the old one is freed first, so memory stays near 0.9 GB). Each target remembers its own levels: bass starts at Bass solo, karaoke at No vocals.
- **Notification** with the three snaps, and a **Quick Settings tile** for on/off ("Add quick tile" on the main screen).
- **Sources:** the shell-uid helper (default; takes whatever app is playing, Spotify included, see below), or Android's screen-capture consent ("Use screen capture instead"; only for apps that allow capture, and the original keeps playing).
- **Threads:** reader (source → 2 s hops) → worker (separator) → player (20 ms blocks, levels applied at playback so the controls act at once). When the worker falls behind it skips to the newest hop.

Setup (once): Android SDK at `~/Android/sdk` (cmdline-tools, `platforms;android-35`, `build-tools;35.0.0`), Gradle 8.11.1 at `~/Android/gradle-8.11.1` and `app/local.properties` with `sdk.dir=`. The app's LiteRT 2 libraries (`libLiteRt.so`, `libLiteRtClGlAccelerator.so`) come out of the `litert` and `litert-gpu` AARs on Google's Maven at build time (`unpackLiteRt` in `app/build.gradle.kts`); nothing to copy. The build stops if `libstemflow_core.so` hasn't been built.

```bash
cd android/core && ANDROID_NDK_HOME=~/Android/android-ndk-r29 LITERT_LIB_DIRS=~/Android/litert-2/litert/jni/arm64-v8a cargo ndk -t arm64-v8a -P 26 -o ../app/src/main/jniLibs build --release --features litert2
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

**BS-RoFormer SW in the app (2026-11-15):** `core` loads files named `bsrofo*` with `LiteRtRoformer`, and `Models.kt` gives them a 2 s hop and 0.9 s lookahead (the window is 2.995 s). `bsrofo_sw_vocals_132096.tflite` loads in 7.4 s and ran at **1.85–1.87 s per 2 s hop** for six runs, 758 MB PSS. Then runs took 2.4–3.8 s and three hops were skipped; the phone was cool (thermal status 0, skin 25 °C) and another app was on screen, so most likely the GPU was shared with someone using the phone (not confirmed). The ~5 % margin is too thin for anything but an idle phone. Listening only: SW's weights are not ours to ship.

**The app on LiteRT 2.3.2 (2026-11-15):** `core` is now built with `--features litert2` and `LITERT_LIB_DIRS=~/Android/litert-2/litert/jni/arm64-v8a`, and `jniLibs` holds `libLiteRt.so` and `libLiteRtClGlAccelerator.so` instead of the two 1.4.2 libraries (never both sets). `Native.kt` loads both with `System.loadLibrary`; with `useLegacyPackaging` the accelerator is found inside the APK install without the RuntimeLibraryDir option. SW vocals: ready in 6.1 s, **1.73–1.78 s per 2 s hop**, 0 of 10 runs late on an idle phone, 893 MB PSS. ft2 bass (7.8 s window, 1.5 s hop): **0.84–0.91 s per run**, none late over 18 runs, 732 MB PSS.

Not done: a timeout when the helper fails to start (the status just stays on "Starting the capture helper…").

### The helper (`app/src/main/java/org/stemflow/app/Helper.java`)

Our own shell-uid capturer, in the app's APK but run outside the app. It registers a `LOOP_BACK`-only mix matching every player with a media, game or unknown usage, excluding stemflow's own uid (so whatever app is playing is taken, and our output doesn't loop back in), and streams 44.1 kHz stereo PCM to the app over a loopback TCP socket (port 28711); the app needs no MediaProjection consent with this source. For development it is started over adb, once the app says it is waiting:

```bash
android/helper.sh
```

**Works end to end** (2026-11-14): Spotify arrives in the app at −20 dBFS and the app plays its own version while the original is off the speaker (the user confirmed by ear, with scrcpy's identical mix, that the original goes silent).

Findings:
- Do not call `allowPrivilegedPlaybackCapture(true)`: `AudioMix` then rejects sample rates above 16 kHz. It isn't needed for Spotify.
- When the helper exits and the mix is dropped, Spotify pauses itself (it sees its output route change), so the app sends a media Play key a second after it is switched off.
- Killing the helper (or the app closing the socket) is enough to restore normal playback.

Not yet tested: YouTube Music; Shizuku as the way to start the helper without adb.

**All media, not one app (2026-11-15):** checked in `dumpsys media.audio_policy` while live: the mix's criteria are `RULE_EXCLUDE_UID <stemflow>` plus `RULE_MATCH_ATTRIBUTE_USAGE` for MEDIA, GAME and UNKNOWN; Spotify's track sits on the remote-submix output and stemflow's own track on the speaker's deep-buffer output. Only tried with Spotify playing; other players (Musicolet, YouTube Music) are untested.

**Or one player:** the main screen's "From" picker lists "Whatever is playing" plus the installed apps that offer a media service (`Sources.kt`, found through the manifest's `<queries>`); the choice is read when isolation is switched on. A chosen app is captured by uid alone (`RULE_MATCH_UID`, as before), so other sounds keep playing normally. The helper takes `PORT except|only UID...`; `android/helper.sh PACKAGE` captures only that app. Checked live with Spotify chosen (criterion `RULE_MATCH_UID 10320`, audio flowing); the picker itself has not been looked at on the phone, and it is not known whether every player shows up in the list.

**Stem picker (2026-11-16):** the Karaoke switch became a Stem picker (bass, vocals, other, ...; only stems with a model on the phone), as in the GNOME menu. Each stem keeps its own levels and last model. `bsrofo_sw_other_132096.tflite` (`export_bsrofo_litert.py ROOT OUT other`; LiteRT vs PyTorch max error 2e-4 on a peak of 44) is the horn remover.

## Beat click (beat tracking test, 2026-11-16)

`src/beats.rs` (shared with the desktop, included by path) runs Beat This! (CPJKU, MIT) on the last 30 s of input every 3 s, then the "steady" decoder from the beat shootout page (tempo HMM, bar lines every N beats where the downbeat curve agrees; a new tempo or meter must come up three times in a row before the grid follows it). On the phone the model is the LiteRT export above (`LiteRtBeatNet`, CPU/XNNPACK, 2 threads); the GPU stays the separator's. In the app, a "Beat click" switch mixes a click into the output on every beat (higher on bar lines), with a meter picker (Auto, 2–7 per bar) and a feel picker (as heard, half time, double time). The input runs ahead of the output by the separator's lookahead, so the click mostly comes from audio already analysed; between analyses it continues at the last tempo for up to 8 s.

Models: `beat_this_{small0,final0}_1500.tflite` in the app's files; `small0` unless `--es beatmodel final0` picks the other (from the next start). Control over adb: `--ez click true`, `--ei meter 3` (0 = auto), `--ef feel 2`.

Pixel 10a, idle, CPU (bench `--beat-ref`, 6 runs, p50): small0 1.30 s with 2 threads, 0.91 s with 4; final0 2.33 s / 1.60 s. Error vs PyTorch < 1e-4.

`android/beatcheck` checks the mel spectrogram and decoder on the host (ONNX Runtime): mel vs torchaudio 1.3e-4, decoder identical to the page's beats.js on clip 1.
