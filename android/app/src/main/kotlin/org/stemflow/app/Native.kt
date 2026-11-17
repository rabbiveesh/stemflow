package org.stemflow.app

/** The Rust core (android/core): separator + streaming ("engine") and mixer + limiter. */
object Native {
    init {
        // Order matters: the core links against LiteRT (2.x, core built with the
        // `litert2` feature). The accelerator is opened by LiteRT itself when a model
        // is compiled for the GPU; loading it here first makes a missing file fail early.
        System.loadLibrary("LiteRt")
        System.loadLibrary("LiteRtClGlAccelerator")
        System.loadLibrary("stemflow_core")
    }

    const val GPU_FP16 = 2

    @JvmStatic external fun engineNew(path: String, gpu: Int, hop: Float, lookahead: Float, xfade: Float): Long
    @JvmStatic external fun engineFree(h: Long)
    @JvmStatic external fun engineHop(h: Long): Int
    @JvmStatic external fun engineDelay(h: Long): Int
    /** One hop in; that hop's target, band and (for a two-stem model) second stem out. */
    @JvmStatic external fun engineProcess(h: Long, input: FloatArray, bass: FloatArray, band: FloatArray, second: FloatArray): Float
    /** Stems per run: 1, or 2 for bass + vocals. */
    @JvmStatic external fun engineStems(h: Long): Int
    @JvmStatic external fun mixerNew(): Long
    @JvmStatic external fun mixerMix(
        h: Long, bass: FloatArray, band: FloatArray, second: FloatArray, pos: Int, block: Int, ahead: Int,
        bassLevel: Float, bandLevel: Float, secondLevel: Float, out: FloatArray,
    )

    // Looping (src/looper.rs): the last minutes of stems, bars found by Beat This!.
    @JvmStatic external fun looperNew(model: String, threads: Int): Long
    @JvmStatic external fun looperFree(h: Long)
    @JvmStatic external fun looperPush(h: Long, bass: FloatArray, band: FloatArray, second: FloatArray)
    /** Blocks for the analysis; throws if no beat was found. */
    @JvmStatic external fun looperFind(h: Long, press: Long, bars: Int, meter: Int, feel: Float)
    @JvmStatic external fun looperRead(
        h: Long, bass: FloatArray, band: FloatArray, second: FloatArray, click: FloatArray, n: Int, advance: Int,
    ): Long
    /** op: 0 back to live (on from the loop), 8 straight to live, 1 play, 2 shift a beats, 3 ends = beats a..b, 4 a bars, 5 beat a is bar 1,
     *  6 click a != 0, 7 re-decode with meter a and feel f */
    @JvmStatic external fun looperEdit(h: Long, op: Int, a: Int, b: Int, f: Float)
    @JvmStatic external fun looperView(h: Long, bins: Int): FloatArray
    /** Re-separate the loop with `engine`'s model (after a switch); blocks for seconds. */
    @JvmStatic external fun looperRestem(h: Long, engine: Long): Boolean
    @JvmStatic external fun looperTotal(h: Long): Long
    /** [start, end, bar 1, beats per bar, bars, feel] in buffer frames; empty without a loop */
    @JvmStatic external fun looperTimes(h: Long): DoubleArray
    @JvmStatic external fun looperSetTimes(h: Long, start: Double, end: Double, bar1: Double)
    @JvmStatic external fun looperGrid(h: Long): DoubleArray
    @JvmStatic external fun looperLay(h: Long, beats: DoubleArray, a: Int, b: Int, one: Int, meter: Int, bpm: Float, feel: Float)
    @JvmStatic fun looperIsPlaying(h: Long) = looperView(h, 1).let { it.isNotEmpty() && it[0] > 0f }
}
