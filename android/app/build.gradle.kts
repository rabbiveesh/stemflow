plugins {
    id("com.android.application") version "8.7.3"
    id("org.jetbrains.kotlin.android") version "2.0.21"
}

// LiteRT 2's runtime and GPU accelerator, which libstemflow_core.so links against.
val litertVersion = "2.3.2"
val litertLibs = listOf("libLiteRt.so", "libLiteRtClGlAccelerator.so")
val litertJni = layout.buildDirectory.dir("litert-jni")
val litert: Configuration by configurations.creating { isTransitive = false }

android {
    namespace = "org.stemflow.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "org.stemflow.app"
        minSdk = 29 // AudioPlaybackCaptureConfiguration
        targetSdk = 35
        versionCode = 1
        versionName = "0.2"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // The LiteRT libraries are plain files next to ours; keep them loadable by name.
    // A checkout that still has its own copies in jniLibs gets the same bytes twice.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += litertLibs.map { "**/$it" }
        }
    }
    // LiteRT comes from Google's Maven (unpackLiteRt below); ours from cargo ndk.
    sourceSets["main"].jniLibs.srcDir(litertJni)
    // pure logic: `gradle testDebugUnitTest` on the laptop (the loop logic is Rust: cargo test)
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    // Starts the capture helper as the shell uid without adb.
    implementation("dev.rikka.shizuku:api:13.1.5")
    implementation("dev.rikka.shizuku:provider:13.1.5")
    testImplementation("junit:junit:4.13.2")
}
dependencies {
    litert("com.google.ai.edge.litert:litert:$litertVersion@aar")
    litert("com.google.ai.edge.litert:litert-gpu:$litertVersion@aar")
}

/** The two LiteRT libraries out of their AARs, so a fresh checkout needs no copying. */
val unpackLiteRt by tasks.registering(Sync::class) {
    from({ litert.map { zipTree(it) } }) {
        include(litertLibs.map { "jni/arm64-v8a/$it" })
        eachFile { path = path.removePrefix("jni/") }
    }
    includeEmptyDirs = false
    into(litertJni)
    doLast {
        val missing = litertLibs.filter { !litertJni.get().file("arm64-v8a/$it").asFile.exists() }
        check(missing.isEmpty()) { "LiteRT $litertVersion AARs lack $missing" }
    }
}

/** Fail the build, not the app at launch, when the Rust core wasn't built. */
val checkCore by tasks.registering {
    val core = file("src/main/jniLibs/arm64-v8a/libstemflow_core.so")
    doLast {
        check(core.exists()) { "No $core: build it with cargo ndk first (android/README.md)" }
    }
}

tasks.named("preBuild") { dependsOn(unpackLiteRt, checkCore) }
