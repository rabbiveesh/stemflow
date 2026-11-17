package org.stemflow.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs

/**
 * The loop, with a bar either side: the waveform, a tick on every beat, numbered bar
 * lines, the loop's ends and the playhead. Drag an end to move it (it snaps to the
 * beats); long-press a beat to make it bar 1.
 *
 * Draws `Native.looperView`'s array: [playing, finding, bpm, meter, bars, from, to,
 * start, end, playhead, a, b, beats, bins, (frame, bar?, index) per beat, peaks],
 * frames relative to `from`.
 */
class LoopView(c: Context) : View(c) {
    var onEdges: (Int, Int) -> Unit = { _, _ -> }
    var onBarOne: (Int) -> Unit = {}

    private var v = FloatArray(0)
    private val d = resources.displayMetrics.density
    private val wave = Paint().apply { color = Color.argb(150, 159, 182, 198) }
    private val outside = Paint().apply { color = Color.argb(70, 128, 128, 128) }
    private val inside = Paint().apply { color = Color.argb(46, 108, 180, 232) }
    private val beat = Paint().apply { color = Color.argb(170, 159, 182, 198); strokeWidth = 1f * d }
    private val bar = Paint().apply { color = Color.rgb(0x79, 0xc2, 0x67); strokeWidth = 2.5f * d }
    private val edge = Paint().apply { color = Color.rgb(0xea, 0xf2, 0xf7); strokeWidth = 3f * d }
    private val head = Paint().apply { color = Color.rgb(229, 57, 53); strokeWidth = 2f * d }
    private val text = Paint().apply { color = Color.rgb(0x79, 0xc2, 0x67); textSize = 11f * d; isAntiAlias = true }

    /** Which end is being dragged (0 none, 1 start, 2 end) and the beat it is on. */
    private var drag = 0
    private var dragBeat = -1

    /** Loop frames between where the loop is read and what is heard (see `update`). */
    private var lag = 0f

    /** `lag`: how far (in loop frames) the sound reaching the ears trails the read
     * position, so the playhead follows the ears, not the buffer (Bluetooth!). */
    fun update(view: FloatArray, lag: Float = 0f) {
        v = view
        this.lag = lag
        invalidate()
    }

    private val ok get() = v.size >= 14
    private val to get() = v[6]
    private val nBeats get() = v[12].toInt()
    private fun beatFrame(k: Int) = v[14 + 3 * k]
    private fun beatIsBar(k: Int) = v[15 + 3 * k] > 0f
    private fun beatIndex(k: Int) = v[16 + 3 * k].toInt()
    private fun x(frame: Float) = frame / to * width

    /** The beat (in view) nearest to `px`, as (position in view, global index). */
    private fun nearest(px: Float): Pair<Int, Int>? {
        if (!ok || nBeats == 0) return null
        val k = (0 until nBeats).minByOrNull { abs(x(beatFrame(it)) - px) } ?: return null
        return k to beatIndex(k)
    }

    private val gestures = GestureDetector(c, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent) = true
        override fun onLongPress(e: MotionEvent) {
            if (drag == 0) nearest(e.x)?.let { onBarOne(it.second) }
        }
    })

    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (!ok) return false
        gestures.onTouchEvent(e)
        val (start, end) = x(v[7]) to x(v[8])
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val grab = 36 * d
                drag = when {
                    abs(e.x - start) < grab && abs(e.x - start) <= abs(e.x - end) -> 1
                    abs(e.x - end) < grab -> 2
                    else -> 0
                }
                dragBeat = -1
                if (drag != 0) parent.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_MOVE -> if (drag != 0) {
                dragBeat = nearest(e.x)?.second ?: -1
                invalidate()
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (drag != 0 && dragBeat >= 0 && e.actionMasked == MotionEvent.ACTION_UP) {
                    val (a, b) = v[10].toInt() to v[11].toInt()
                    if (drag == 1 && dragBeat < b) onEdges(dragBeat, b)
                    if (drag == 2 && dragBeat > a) onEdges(a, dragBeat)
                }
                drag = 0
                dragBeat = -1
            }
        }
        return true
    }

    // The back gesture owns the screen's edges; keep it off the loop, so an end can be
    // dragged all the way to either side.
    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        super.onLayout(changed, l, t, r, b)
        systemGestureExclusionRects = listOf(android.graphics.Rect(0, 0, width, height))
    }

    override fun onDraw(cv: Canvas) {
        if (!ok) return
        val h = height.toFloat()
        val mid = h * 0.55f
        // waveform
        val bins = v[13].toInt()
        val p0 = 14 + 3 * nBeats
        val bw = width.toFloat() / bins.coerceAtLeast(1)
        for (k in 0 until bins) {
            val a = v[p0 + k] * h * 0.4f
            cv.drawRect(k * bw, mid - a, (k + 1) * bw - 0.5f, mid + a, wave)
        }
        // the loop, with the end being dragged where it would land
        var start = x(v[7])
        var end = x(v[8])
        if (drag != 0 && dragBeat >= 0) {
            val k = (0 until nBeats).firstOrNull { beatIndex(it) == dragBeat }
            if (k != null) if (drag == 1) start = x(beatFrame(k)) else end = x(beatFrame(k))
        }
        cv.drawRect(0f, 0f, start, h, outside)
        cv.drawRect(end, 0f, width.toFloat(), h, outside)
        cv.drawRect(start, 0f, end, h, inside)
        // beats and bars; bars numbered from the loop's first bar line
        val a = v[10].toInt()
        var barNo = 0
        for (k in 0 until nBeats) {
            val px = x(beatFrame(k))
            if (beatIsBar(k)) {
                cv.drawLine(px, 14 * d, px, h, bar)
                if (beatIndex(k) >= a) barNo++
                if (beatIndex(k) >= a && beatIndex(k) < v[11].toInt()) cv.drawText("$barNo", px + 3 * d, 12 * d, text)
            } else {
                cv.drawLine(px, h * 0.25f, px, h, beat)
            }
        }
        // the ends (with a grip) and the playhead
        for (px in listOf(start, end)) {
            cv.drawLine(px, 0f, px, h, edge)
            cv.drawCircle(px, h - 8 * d, 7 * d, edge)
        }
        if (v[0] > 0f) {
            // what's heard: `lag` back from the read position, wrapped into the loop
            val (s0, e0) = v[7] to v[8]
            val len = e0 - s0
            val back = if (len > 0f) (((v[9] - lag - s0) % len) + len) % len else v[9] - s0
            val ph = x(s0 + back)
            cv.drawLine(ph, 0f, ph, h, head)
        }
    }
}
