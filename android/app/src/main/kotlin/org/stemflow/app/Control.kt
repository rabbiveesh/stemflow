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
        return when {
            a == 100 && b == 0 -> "Bass solo"
            a > 100 && b == 0 -> "Bass solo · $a%"
            a == 100 && b == 100 -> "Full mix"
            a == 0 && b == 100 -> "No bass"
            a > 100 && b == 100 -> "Full mix · bass $a%"
            else -> "Bass $a% · band $b%"
        }
    }

    fun load(c: Context) {
        val p = c.getSharedPreferences("mix", Context.MODE_PRIVATE)
        setLevels(p.getFloat("bass", 1f), p.getFloat("band", 0f))
    }

    fun save(c: Context) {
        c.getSharedPreferences("mix", Context.MODE_PRIVATE).edit()
            .putFloat("bass", bass).putFloat("band", band).apply()
    }

    fun isSnap(name: String) = when (name) {
        "solo" -> abs(bass - 1f) < 0.005f && band < 0.005f
        "full" -> abs(bass - 1f) < 0.005f && band > 0.995f
        "nobass" -> bass < 0.005f && band > 0.995f
        else -> false
    }
}
