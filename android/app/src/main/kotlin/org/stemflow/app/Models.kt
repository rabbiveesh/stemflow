package org.stemflow.app

import android.content.Context
import java.io.File

/**
 * The models on the phone: every `<name>_<stem>_<samples>.tflite` in the app's files
 * (android/tools/export_litert.py names them so), e.g. `ft2_bass_344064.tflite`.
 * As on the desktop, karaoke is the vocals model last used, and switching it off goes
 * back to the last bass model.
 */
object Models {
    class Model(val file: File, val name: String, val stem: String, val samples: Int) {
        val id get() = file.name.removeSuffix(".tflite")
        val seconds get() = samples / 44100f
        val label get() = "%s · %.1f s window".format(name, seconds)
        /**
         * Seconds of new audio per run. Measured on a Pixel 10a (GPU fp16): the 7.8 s
         * window takes 0.96 s per run and the 4 s window 0.54 s, so these leave about
         * a third of each hop as headroom for a warm phone or a busy screen.
         */
        val hop get() = if (seconds > 6f) 1.5f else 1.0f
    }

    private val pattern = Regex("""(.+)_(bass|vocals|drums|other)_(\d+)\.tflite""")

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

    fun karaoke(c: Context) = prefs(c).getBoolean("karaoke", false) && all(c).any { it.stem == "vocals" }

    fun setKaraoke(c: Context, on: Boolean) = prefs(c).edit().putBoolean("karaoke", on).apply()

    /** The stem the controls act on. */
    fun target(c: Context) = if (karaoke(c)) "vocals" else "bass"

    fun choices(c: Context) = all(c).filter { it.stem == target(c) }

    fun current(c: Context): Model? {
        val list = choices(c)
        val saved = prefs(c).getString("model_" + target(c), null)
        // Default: the newest file, i.e. the one pushed last.
        return list.firstOrNull { it.id == saved } ?: list.maxByOrNull { it.file.lastModified() }
    }

    fun select(c: Context, m: Model) = prefs(c).edit().putString("model_" + m.stem, m.id).apply()
}
