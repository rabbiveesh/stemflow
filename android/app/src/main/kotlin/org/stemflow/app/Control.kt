package org.stemflow.app

import android.content.Context
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The mix, as on the desktop (src/control.rs): two perceived loudnesses, `bass`
 * (0..2, the target stem) and `band` (0..1, everything else). The one-line balance
 * (-1 = bass solo, 0 = full mix, +1 = no bass) is a view onto them.
 *
 * With the two-stem model ("bass+vocals") there is a third level, `second` (0..1, the
 * vocals), and "band" no longer includes them: three plain levels, as on the desktop.
 */
object Control {
    const val MAX_BASS = 2f
    @Volatile var bass = 1f; private set
    @Volatile var band = 0f; private set
    /** A two-stem model's second stem (vocals), 0..1. */
    @Volatile var second = 1f; private set
    /** The stem `bass` acts on: one of Models.STEMS. */
    @Volatile var target = "bass"; private set
    /** Three levels (bass, vocals, band) rather than one line from solo to none. */
    val three get() = target == "bass+vocals"

    fun setSecond(v: Float) {
        second = v.coerceIn(0f, 1f)
    }

    fun setLevels(bass: Float, band: Float) {
        this.bass = bass.coerceIn(0f, MAX_BASS)
        this.band = band.coerceIn(0f, 1f)
    }

    fun setBalance(balance: Float) {
        val b = if (balance.isFinite()) balance.coerceIn(-1f, 1f) else 0f
        if (b <= 0f) setLevels(maxOf(bass, 1f), 1f + b) // keep any boost while the band fades
        else setLevels(1f - b, 1f)
    }

    fun balance(): Float = if (band <= bass) band - 1f else 1f - bass

    fun snap(name: String) {
        when (name) {
            "solo" -> { setLevels(1f, 0f); if (three) second = 0f }
            "full" -> { setLevels(1f, 1f); second = 1f }
            "nobass" -> { setLevels(0f, 1f); second = 1f }
            "novocals" -> { setLevels(1f, 1f); second = 0f }
        }
    }

    fun describe(): String {
        val a = (bass * 100).roundToInt()
        val b = (band * 100).roundToInt()
        if (three) {
            val v = (second * 100).roundToInt()
            return when {
                a == 100 && b == 100 && v == 100 -> "Full mix"
                a == 100 && b == 100 && v == 0 -> "No vocals"
                a == 0 && b == 100 && v == 100 -> "No bass"
                a == 100 && b == 0 && v == 0 -> "Bass solo"
                else -> "Bass $a% · vocals $v% · band $b%"
            }
        }
        val t = target
        val tc = t.replaceFirstChar { it.uppercase() }
        return when {
            a == 100 && b == 0 -> "$tc solo"
            a > 100 && b == 0 -> "$tc solo · $a%"
            a == 100 && b == 100 -> "Full mix"
            a == 0 && b == 100 -> "No $t"
            a > 100 && b == 100 -> "Full mix · $t $a%"
            else -> "$tc $a% · band $b%"
        }
    }

    /**
     * Each target remembers its own levels, as on the desktop. A target used for the
     * first time starts where it is most useful: bass solo, or without the vocals,
     * horns, ... (play-along).
     */
    fun load(c: Context) {
        target = Models.target(c)
        val p = c.getSharedPreferences("mix", Context.MODE_PRIVATE)
        val solo = target == "bass"
        setLevels(p.getFloat("target_$target", if (solo) 1f else 0f), p.getFloat("band_$target", if (solo) 0f else 1f))
        second = p.getFloat("second_$target", 1f)
    }

    fun save(c: Context) {
        c.getSharedPreferences("mix", Context.MODE_PRIVATE).edit()
            .putFloat("target_$target", bass).putFloat("band_$target", band).putFloat("second_$target", second).apply()
    }

    /** Another stem: keep the old target's levels, bring back the new one's. */
    fun setTarget(c: Context, stem: String) {
        save(c)
        Models.setTarget(c, stem)
        load(c)
    }

    // Looping: a click over the loop (higher on bar 1), and the bar finder's hints.
    @Volatile var click = false
    /** Beats per bar, 0 = auto (3-7). */
    @Volatile var meter = 0
    /** 1 = as decoded, 0.5 = half time, 2 = double time. */
    @Volatile var feel = 1f
    /** Loop practice speed (1 = as recorded), transposition (semitones), and whether
     *  slowing down keeps the pitch. */
    @Volatile var speed = 1f
    @Volatile var semis = 0f
    @Volatile var keepPitch = true

    fun loadBeats(c: Context) {
        val p = c.getSharedPreferences("beats", Context.MODE_PRIVATE)
        click = p.getBoolean("click", false)
        meter = p.getInt("meter", 0)
        feel = p.getFloat("feel", 1f)
        speed = p.getFloat("speed", 1f)
        semis = p.getFloat("semis", 0f)
        keepPitch = p.getBoolean("keep_pitch", true)
    }

    fun saveBeats(c: Context) {
        c.getSharedPreferences("beats", Context.MODE_PRIVATE).edit()
            .putBoolean("click", click).putInt("meter", meter).putFloat("feel", feel)
            .putFloat("speed", speed).putFloat("semis", semis).putBoolean("keep_pitch", keepPitch).apply()
    }

    fun describeBeats() = "click " + (if (click) "on" else "off") +
        " · " + (if (meter == 0) "auto meter" else "$meter/bar") +
        when (feel) { 0.5f -> " · half time"; 2f -> " · double time"; else -> "" }

    fun isSnap(name: String) = when (name) {
        "solo" -> abs(bass - 1f) < 0.005f && band < 0.005f && (!three || second < 0.005f)
        "full" -> abs(bass - 1f) < 0.005f && band > 0.995f && (!three || second > 0.995f)
        "novocals" -> three && abs(bass - 1f) < 0.005f && band > 0.995f && second < 0.005f
        "nobass" -> bass < 0.005f && band > 0.995f && (!three || second > 0.995f)
        else -> false
    }
}
