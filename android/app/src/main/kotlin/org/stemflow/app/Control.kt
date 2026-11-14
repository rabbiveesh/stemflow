package org.stemflow.app

import android.content.Context
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The mix, as on the desktop (src/control.rs): two perceived loudnesses, `bass`
 * (0..2, the target stem) and `band` (0..1, everything else). The one-line balance
 * (-1 = bass solo, 0 = full mix, +1 = no bass) is a view onto them.
 */
object Control {
    const val MAX_BASS = 2f
    @Volatile var bass = 1f; private set
    @Volatile var band = 0f; private set
    /** The stem `bass` acts on: "bass", or "vocals" in karaoke. */
    @Volatile var target = "bass"; private set

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

    fun snap(name: String) = when (name) {
        "solo" -> setLevels(1f, 0f)
        "full" -> setLevels(1f, 1f)
        "nobass" -> setLevels(0f, 1f)
        else -> Unit
    }

    fun describe(): String {
        val a = (bass * 100).roundToInt()
        val b = (band * 100).roundToInt()
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
     * first time starts where it is most useful: bass solo, or no vocals for karaoke.
     */
    fun load(c: Context) {
        target = Models.target(c)
        val p = c.getSharedPreferences("mix", Context.MODE_PRIVATE)
        val solo = target == "bass"
        setLevels(p.getFloat("target_$target", if (solo) 1f else 0f), p.getFloat("band_$target", if (solo) 0f else 1f))
    }

    fun save(c: Context) {
        c.getSharedPreferences("mix", Context.MODE_PRIVATE).edit()
            .putFloat("target_$target", bass).putFloat("band_$target", band).apply()
    }

    /** Karaoke on or off: keep the old target's levels, bring back the new one's. */
    fun setKaraoke(c: Context, on: Boolean) {
        save(c)
        Models.setKaraoke(c, on)
        load(c)
    }

    fun isSnap(name: String) = when (name) {
        "solo" -> abs(bass - 1f) < 0.005f && band < 0.005f
        "full" -> abs(bass - 1f) < 0.005f && band > 0.995f
        "nobass" -> bass < 0.005f && band > 0.995f
        else -> false
    }
}
