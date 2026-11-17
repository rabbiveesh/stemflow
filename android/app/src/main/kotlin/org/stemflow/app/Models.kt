package org.stemflow.app

import android.content.Context
import java.io.File

/**
 * The models on the phone: every `<name>_<stem>_<samples>.tflite` in the app's files
 * (android/tools/export_litert.py names them so), e.g. `ft2_bass_344064.tflite`.
 * As on the desktop, the controls act on one stem at a time (bass, vocals for karaoke,
 * other for the horns, ...), and each stem remembers the model last used for it.
 */
object Models {
    class Model(val file: File, val name: String, val stem: String, val samples: Int) {
        val id get() = file.name.removeSuffix(".tflite")
        val seconds get() = samples / 44100f
        val label get() = "%s · %.1f s window".format(name, seconds)
        /**
         * Seconds of new audio per run. Measured on a Pixel 10a (GPU fp16): the 7.8 s
         * Demucs window takes 0.96 s per run and the 4 s one 0.54 s, so those leave
         * about a third of each hop as headroom for a warm phone or a busy screen.
         * BS-RoFormer takes 1.89 s for its 3 s window: 2 s hops, with almost none.
         */
        val hop get() = if (roformer) 2.0f else if (seconds > 6f) 1.5f else 1.0f
        /** Seconds of future context; hop + lookahead must fit inside the window. */
        val lookahead get() = if (roformer) 0.9f else 1.0f
        val roformer get() = name.startsWith("bsrofo")
    }

    /** The stems a model can isolate, in menu order (as src/models.rs). */
    /** "bass+vocals": one run of a two-head model gives both, each with its own level. */
    val STEMS = listOf("bass", "bass+vocals", "vocals", "other", "piano", "guitar", "drums")
    private val LABELS = mapOf(
        "bass" to "Bass", "bass+vocals" to "Bass + vocals", "vocals" to "Vocals (karaoke)", "other" to "Other (horns, strings…)",
        "piano" to "Piano", "guitar" to "Guitar", "drums" to "Drums",
    )
    fun stemLabel(stem: String) = LABELS[stem] ?: stem

    private val pattern = Regex("""(.+)_(${STEMS.joinToString("|") { Regex.escape(it) }})_(\d+)\.tflite""")

    fun all(c: Context): List<Model> =
        listOfNotNull(c.filesDir, c.getExternalFilesDir(null))
            .flatMap { it.listFiles()?.toList() ?: emptyList() }
            .mapNotNull { f ->
                pattern.matchEntire(f.name)?.let { m ->
                    Model(f, m.groupValues[1], m.groupValues[2], m.groupValues[3].toInt())
                }
            }
            .sortedWith(compareBy({ it.name }, { -it.samples }))

    private fun prefs(c: Context) = c.getSharedPreferences("models", Context.MODE_PRIVATE)

    /** The stems there is a model for. */
    fun stems(c: Context): List<String> {
        val have = all(c).map { it.stem }.toSet()
        return STEMS.filter { it in have }
    }

    /** The stem the controls act on: the one picked, while there is a model for it. */
    fun target(c: Context): String {
        val p = prefs(c)
        // "karaoke" is the switch this picker replaced.
        val saved = p.getString("stem", null) ?: if (p.getBoolean("karaoke", false)) "vocals" else "bass"
        val have = stems(c)
        return if (saved in have) saved else have.firstOrNull() ?: "bass"
    }

    fun setTarget(c: Context, stem: String) = prefs(c).edit().putString("stem", stem).remove("karaoke").apply()

    fun choices(c: Context) = all(c).filter { it.stem == target(c) }

    fun current(c: Context): Model? {
        val list = choices(c)
        val saved = prefs(c).getString("model_" + target(c), null)
        // Default: the newest file, i.e. the one pushed last.
        return list.firstOrNull { it.id == saved } ?: list.maxByOrNull { it.file.lastModified() }
    }

    /**
     * The beat tracker: a Beat This! LiteRT export (tools/export_beat_this_litert.py),
     * e.g. `beat_this_small0_1500.tflite`, pushed next to the separators. The small one
     * unless another is picked (`--es beatmodel final0`).
     */
    fun beatModel(c: Context): File? {
        val all = listOfNotNull(c.filesDir, c.getExternalFilesDir(null))
            .flatMap { it.listFiles()?.toList() ?: emptyList() }
            .filter { it.name.startsWith("beat_this_") && it.name.endsWith(".tflite") }
        val want = prefs(c).getString("beat_model", "small0")!!
        return all.firstOrNull { it.name.startsWith("beat_this_${want}_") } ?: all.firstOrNull()
    }

    fun selectBeatModel(c: Context, name: String) = prefs(c).edit().putString("beat_model", name).apply()

    fun select(c: Context, m: Model) = prefs(c).edit().putString("model_" + m.stem, m.id).apply()
}
