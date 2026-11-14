package org.stemflow.app

/** The Rust core (android/core): separator + streaming ("engine") and mixer + limiter. */
object Native {
    init {
        // Order matters: the core links against the two LiteRT libraries.
        System.loadLibrary("tensorflowlite_jni")
        System.loadLibrary("tensorflowlite_gpu_jni")
        System.loadLibrary("stemflow_core")
    }

    const val GPU_FP16 = 2

    @JvmStatic external fun engineNew(path: String, gpu: Int, hop: Float, lookahead: Float, xfade: Float): Long
    @JvmStatic external fun engineFree(h: Long)
    @JvmStatic external fun engineHop(h: Long): Int
    @JvmStatic external fun engineDelay(h: Long): Int
    @JvmStatic external fun engineProcess(h: Long, input: FloatArray, bass: FloatArray, band: FloatArray): Float
    @JvmStatic external fun mixerNew(): Long
    @JvmStatic external fun mixerMix(
        h: Long, bass: FloatArray, band: FloatArray, pos: Int, block: Int, ahead: Int,
        bassLevel: Float, bandLevel: Float, out: FloatArray,
    )
}
