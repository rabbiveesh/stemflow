// Signalsmith Stretch (header-only C++ behind a small C interface) for src/stretch.rs,
// and on macOS the Info.plist for the audio-recording permission.
fn main() {
    let dir = std::path::Path::new("third_party/signalsmith");
    println!("cargo:rerun-if-changed=third_party/signalsmith");
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .opt_level(3)
        .file(dir.join("wrapper.cpp"))
        .include(dir.join("signalsmith-stretch"))
        .include(dir)
        .warnings(false)
        .compile("signalsmith-stretch");

    // macOS: embed an Info.plist in the binary, so the system can say why we ask to
    // record another app's audio (Core Audio process taps).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let plist = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("macos/Info.plist");
        println!("cargo:rerun-if-changed=macos/Info.plist");
        println!(
            "cargo:rustc-link-arg-bins=-Wl,-sectcreate,__TEXT,__info_plist,{}",
            plist.display()
        );
    }
}
