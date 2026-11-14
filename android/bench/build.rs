// The LiteRT backend links libtensorflowlite_jni.so and libtensorflowlite_gpu_jni.so
// from Google's litert / litert-gpu AARs; LITERT_LIB_DIRS lists where they are.
fn main() {
    println!("cargo:rerun-if-env-changed=LITERT_LIB_DIRS");
    for d in std::env::var("LITERT_LIB_DIRS").unwrap_or_default().split(':') {
        if !d.is_empty() {
            println!("cargo:rustc-link-search=native={d}");
        }
    }
}
