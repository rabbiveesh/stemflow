# Signalsmith Stretch (vendored)

Time-stretch and pitch-shift for stemflow's loop player (`src/stretch.rs`).

- `signalsmith-stretch/` and `signalsmith-linear/`: Signalsmith Audio's header-only
  libraries (MIT, Geraint Luff / Signalsmith Audio Ltd; see their LICENSE.txt).
- `wrapper.h`, `wrapper.cpp`: the C interface from the `signalsmith-stretch` crate
  0.1.3 by Colin Marc (MIT, LICENSE-wrapper.md), used here without its bindgen step:
  `src/stretch.rs` declares the few functions by hand, so building needs only a C++
  compiler (also under the Android NDK).
