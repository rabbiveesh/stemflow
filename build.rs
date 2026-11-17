// Signalsmith Stretch (header-only C++ behind a small C interface) for src/stretch.rs.
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
}
