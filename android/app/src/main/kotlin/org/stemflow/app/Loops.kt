package org.stemflow.app

import android.content.Context
import org.json.JSONObject

/**
 * Looping, as the screens and the notification reach it: lines from the loop panel
 * ("loop 4", "loop move 0 4", ...; src/loopdeck/command.rs) go to the deck, which is
 * the desktop's (src/loopdeck). The options the deck keeps (speed, pitch, click,
 * meter, feel, stems only) are kept in the app's settings too, so they're there
 * while stemflow is off and come back with it.
 */
object Loops {
    private fun dir(c: Context) = c.filesDir.path

    /** The deck's state (src/loopdeck/mod.rs `State`), or null while there's none. */
    fun state(): JSONObject? {
        val d = CaptureService.deck
        if (d == 0L) return null
        return try { JSONObject(Native.deckState(d)) } catch (e: Exception) { null }
    }

    /** The loop panel (src/loopdeck/panel.rs) as JSON. */
    fun panel(c: Context): String {
        val d = CaptureService.deck
        if (d != 0L) return Native.deckPanel(d)
        return Native.panelIdle(dir(c), Control.meter, Control.feel, Control.speed, Control.semis,
            Control.keepPitch, Control.click, Control.stemsOnLoop)
    }

    /** Send one panel line; null, or why it couldn't be done. */
    fun send(c: Context, line: String): String? {
        val d = CaptureService.deck
        Log.i(CaptureService.TAG, "loop: $line")
        if (d != 0L) {
            val err = Native.deckCommand(d, line)
            if (err != null) Log.w(CaptureService.TAG, "loop: $line: $err")
            keep(c)
            return err
        }
        return idle(c, line.trim().removePrefix("loop ").trim())
    }

    /** The options as the deck has them, into the app's settings. */
    fun keep(c: Context) {
        val s = state() ?: return
        Control.speed = s.optDouble("speed", 1.0).toFloat()
        Control.semis = s.optDouble("semis", 0.0).toFloat()
        Control.keepPitch = s.optBoolean("keep_pitch", true)
        Control.click = s.optBoolean("click", false)
        Control.stemsOnLoop = s.optBoolean("stems_only", false)
        Control.meter = s.optInt("beat_meter", 0)
        Control.feel = s.optDouble("beat_feel", 1.0).toFloat()
        Control.saveBeats(c)
    }

    /** A new deck takes the options the app kept. */
    fun restore(d: Long) {
        val on = { v: Boolean -> if (v) "on" else "off" }
        for (l in listOf("speed ${Control.speed}", "pitch ${Control.semis}", "keep_pitch ${on(Control.keepPitch)}",
            "click ${on(Control.click)}", "stems_only ${on(Control.stemsOnLoop)}")) {
            Native.deckCommand(d, l)?.let { Log.w(CaptureService.TAG, "loop options: $l: $it") }
        }
    }

    /** Without a deck: the options, the saved loops' names, and recall (once on). */
    private fun idle(c: Context, line: String): String? {
        val w = line.split(Regex("\\s+"))
        val arg = w.getOrNull(1)
        val on = arg == "on" || arg == "1" || arg == "true"
        when (w[0]) {
            "speed" -> Control.speed = (arg?.toFloatOrNull() ?: 1f).coerceIn(0.25f, 2f)
            "pitch" -> Control.semis = (arg?.toFloatOrNull() ?: 0f).coerceIn(-12f, 12f)
            "keep_pitch" -> Control.keepPitch = on
            "click" -> Control.click = on
            "stems_only" -> Control.stemsOnLoop = on
            "meter" -> Control.meter = (arg?.toIntOrNull() ?: 0).coerceIn(0, 12)
            "feel" -> Control.feel = arg?.toFloatOrNull()?.takeIf { it == 0.5f || it == 2f } ?: 1f
            "recall" -> {
                val created = arg?.toLongOrNull() ?: return "Which bookmark?"
                CaptureService.recallWhenOn(created)
                return null
            }
            "rename", "delete" -> return Native.idleCommand(dir(c), line)
            else -> return if (CaptureService.running) "Looping isn't ready yet" else "Switch stemflow on first"
        }
        Control.saveBeats(c)
        return null
    }

    /** `m:ss` */
    fun time(ms: Long) = "%d:%02d".format(ms / 60000, ms / 1000 % 60)
}
