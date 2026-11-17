// The LiteRT backend links libtensorflowlite_jni.so and libtensorflowlite_gpu_jni.so
// from Google's litert / litert-gpu AARs; LITERT_LIB_DIRS lists where they are.
fn main() {
    println!("cargo:rerun-if-env-changed=LITERT_LIB_DIRS");
    for d in std::env::var("LITERT_LIB_DIRS")
        .unwrap_or_default()
        .split(':')
    {
        if !d.is_empty() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }

    // Signalsmith Stretch for src/stretch.rs (loop speed and pitch); the C++ runtime
    // is linked in statically, so the APK needs no libc++_shared.so.
    let ss = std::path::Path::new("../../third_party/signalsmith");
    println!("cargo:rerun-if-changed=../../third_party/signalsmith");
    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .opt_level(3)
        .cpp_link_stdlib(None)
        .file(ss.join("wrapper.cpp"))
        .include(ss.join("signalsmith-stretch"))
        .include(ss)
        .warnings(false)
        .compile("signalsmith-stretch");
    // The C++ runtime, linked in statically (the APK has no libc++_shared.so): both
    // libc++ and libc++abi, as static archives from the NDK sysroot.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        let ndk = std::env::var("ANDROID_NDK_HOME").expect("ANDROID_NDK_HOME (cargo-ndk sets it)");
        let host = if cfg!(target_os = "macos") {
            "darwin-x86_64"
        } else {
            "linux-x86_64"
        };
        let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
            Ok("aarch64") => "aarch64-linux-android",
            Ok("x86_64") => "x86_64-linux-android",
            Ok("arm") => "arm-linux-androideabi",
            _ => "i686-linux-android",
        };
        // By full path: putting that folder on the search path would also make the
        // linker take its static libc.a over the system libc.so (a crash at load).
        let lib = format!("{ndk}/toolchains/llvm/prebuilt/{host}/sysroot/usr/lib/{arch}");
        println!("cargo:rustc-link-arg={lib}/libc++_static.a");
        println!("cargo:rustc-link-arg={lib}/libc++abi.a");
    }
}
