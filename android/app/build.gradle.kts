plugins {
    id("com.android.application") version "8.7.3"
    id("org.jetbrains.kotlin.android") version "2.0.21"
}

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
    packaging { jniLibs { useLegacyPackaging = true } }
    // SongMap and other pure logic: `gradle testDebugUnitTest` on the laptop
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    // Starts the capture helper as the shell uid without adb.
    implementation("dev.rikka.shizuku:api:13.1.5")
    implementation("dev.rikka.shizuku:provider:13.1.5")
    testImplementation("junit:junit:4.13.2")
}
