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
}
