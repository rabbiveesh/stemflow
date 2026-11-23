package org.stemflow.app

import android.app.AlertDialog
import android.graphics.Color
import android.graphics.Typeface
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.LinearLayout.LayoutParams
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The loop panel (src/loopdeck/panel.rs), drawn: the same rows the Mac menu and the
 * GNOME extension draw, from the same JSON, sending the same `loop ...` lines. Only
 * the item kinds are drawn here (`PanelKind`, generated from the Rust); the `when`
 * over them has no `else`, so a kind added in Rust doesn't build until it's drawn
 * here (and `cargo test` checks that this file names every kind).
 *
 * Rows keep their ids and the kinds of their items whatever the state, so they're
 * built once and updated in place (a slider being dragged, the loop view, a search
 * being typed all stay put). `place` says where each row goes on the screen.
 */
class LoopPanel(
    private val a: MainActivity,
    /** The container a row goes in, by its id (the strip, a tab). */
    private val place: (String) -> ViewGroup,
    private val send: (List<String>) -> Unit,
    /** The loop for drawing: `Native.looperView` for `bins` peaks, and how far the
     *  ears trail the playhead (loop frames), or null without a loop. */
    private val loopData: (Int) -> Pair<FloatArray, Float>?,
    /** Rows drawn small, in a strip: no padding, items their own size, disabled
     *  buttons hidden rather than greyed. */
    private val compact: Set<String> = emptySet(),
) {
    private var shape = ""
    private val rows = mutableListOf<Pair<View, List<Widget>>>()
    private val views = mutableListOf<LoopViewWidget>()

    fun render(json: String) {
        val p = try { JSONObject(json) } catch (e: Exception) { return }
        val rs = p.optJSONArray("rows") ?: return
        val sh = (0 until rs.length()).joinToString("|") { i ->
            val r = rs.getJSONObject(i)
            val items = r.getJSONArray("items")
            r.getString("id") + ":" + (0 until items.length()).joinToString(",") { items.getJSONObject(it).getString("kind") }
        }
        if (sh != shape) build(rs, sh)
        for (i in 0 until rs.length()) {
            val r = rs.getJSONObject(i)
            val (v, ws) = rows.getOrNull(i) ?: continue
            v.visibility = if (r.optBoolean("hidden")) View.GONE else View.VISIBLE
            val items = r.getJSONArray("items")
            ws.forEachIndexed { k, w -> w.update(items.getJSONObject(k)) }
        }
    }

    /** Often (the playhead moves): redraw the loop view. */
    fun tick() = views.forEach { it.poll() }

    private fun build(rs: JSONArray, sh: String) {
        for ((v, _) in rows) (v.parent as? ViewGroup)?.removeView(v)
        rows.clear()
        views.clear()
        shape = sh
        for (i in 0 until rs.length()) {
            val r = rs.getJSONObject(i)
            val id = r.getString("id")
            val items = r.getJSONArray("items")
            val ws = (0 until items.length()).mapNotNull { k ->
                val kind = PanelKind.of(items.getJSONObject(k).getString("kind")) ?: return@mapNotNull null
                widget(kind, id in compact)
            }
            val small = id in compact
            val many = ws.size > 1
            val line = LinearLayout(a).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            ws.forEachIndexed { k, w ->
                val lp = if (small || ws.size > 4) LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
                    else if (!many) LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
                    else LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
                if (k > 0) lp.leftMargin = a.dp(if (small) 6 else 8)
                line.addView(w.view, lp)
            }
            // five chips don't fit a phone's width: let them scroll
            val v: View = if (ws.size > 4) HorizontalScrollView(a).apply {
                isHorizontalScrollBarEnabled = false
                addView(line)
            } else line
            if (!small) v.setPadding(a.dp(16), a.dp(8), a.dp(16), 0)
            place(id).addView(v, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
            rows += v to ws
        }
    }

    private fun widget(kind: PanelKind, compact: Boolean): Widget = when (kind) {
        PanelKind.TEXT -> TextWidget()
        PanelKind.BUTTON -> ButtonWidget(compact)
        PanelKind.SWITCH -> SwitchWidget()
        PanelKind.CHOICE -> ChoiceWidget()
        PanelKind.NUDGE -> NudgeWidget()
        PanelKind.SLIDER -> SliderWidget()
        PanelKind.LOOP_VIEW -> LoopViewWidget().also { views += it }
        PanelKind.BOOKMARKS -> BookmarksWidget()
    }

    private fun lines(a: JSONArray?) = (0 until (a?.length() ?: 0)).map { a!!.getString(it) }

    /** Ask for a name first (`ask`), then send `lines` with it for `{name}`. */
    private fun ask(o: JSONObject, lines: List<String>) {
        val input = EditText(a).apply { setText(o.optString("value")); selectAll() }
        AlertDialog.Builder(a)
            .setTitle(o.optString("title"))
            .apply { o.optString("text").takeIf { it.isNotEmpty() }?.let { setMessage(it) } }
            .setView(input)
            .setPositiveButton(o.optString("ok", "OK")) { _, _ ->
                val name = input.text.toString().trim().ifEmpty { o.optString("value") }
                send(lines.map { it.replace("{name}", name) })
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    abstract inner class Widget {
        abstract val view: View
        abstract fun update(o: JSONObject)
    }

    inner class TextWidget : Widget() {
        private val t = a.label("", 13f, a.muted)
        override val view: View = t
        override fun update(o: JSONObject) {
            t.text = o.optString("text")
            val style = o.optString("style")
            t.textSize = if (style == "title") 16f else if (style == "normal") 14f else 12f
            t.setTextColor(if (style == "small") a.muted else a.ink)
            t.typeface = if (style == "title") Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
    }

    inner class ButtonWidget(private val compact: Boolean) : Widget() {
        private var o = JSONObject()
        private val b: Button = a.button("", chip = true, weight = 0f) {
            val ls = lines(o.optJSONArray("send"))
            val q = o.optJSONObject("ask")
            if (q != null) ask(q, ls) else send(ls)
        }.apply { isSingleLine = true }
        override val view: View = b
        override fun update(o: JSONObject) {
            this.o = o
            b.text = o.optString("label")
            val enabled = o.optBoolean("enabled", true)
            b.isEnabled = enabled
            b.alpha = if (enabled) 1f else 0.4f
            if (compact) b.visibility = if (enabled) View.VISIBLE else View.GONE
            a.style(b, o.opt("on") == true, true)
        }
    }

    inner class SwitchWidget : Widget() {
        private var o = JSONObject()
        private val label = a.label("", 15f)
        private val tip = a.label("", 12f, a.muted)
        private val sw = Switch(a).apply {
            thumbTintList = android.content.res.ColorStateList.valueOf(a.accent)
            setOnClickListener { send(lines(o.optJSONArray("send"))) }
        }
        override val view: View = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = a.rounded(a.surface, 18)
            setPadding(a.dp(16), a.dp(12), a.dp(12), a.dp(12))
            addView(LinearLayout(a).apply {
                orientation = LinearLayout.VERTICAL
                addView(label)
                addView(tip)
            }, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            addView(sw)
        }
        override fun update(o: JSONObject) {
            this.o = o
            label.text = o.optString("label")
            tip.text = o.optString("tip")
            tip.visibility = if (tip.text.isEmpty()) View.GONE else View.VISIBLE
            sw.isChecked = o.optBoolean("on")
        }
    }

    /** Side by side when there are few; else a menu ("Beats per bar: 4 ▾"). */
    inner class ChoiceWidget : Widget() {
        private var o = JSONObject()
        private val box = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        private val chips = mutableListOf<Button>()
        private var menu: Button? = null
        override val view: View = box

        private fun options() = o.optJSONArray("options") ?: JSONArray()
        private fun pick(i: Int) = options().optJSONObject(i)?.let { send(lines(it.optJSONArray("send"))) }

        override fun update(o: JSONObject) {
            this.o = o
            val opts = options()
            val n = opts.length()
            val sel = if (o.isNull("selected")) -1 else o.optInt("selected", -1)
            if (n > 3) {
                val m = menu ?: a.button("", chip = true, weight = 0f) { open() }.also {
                    box.removeAllViews(); chips.clear(); menu = it; box.addView(it)
                }
                val t = o.optString("title")
                val v = opts.optJSONObject(sel)?.optString("label") ?: "—"
                m.text = (if (t.isEmpty()) v else "$t: $v") + "  ▾"
                return
            }
            if (chips.size != n) {
                box.removeAllViews(); chips.clear(); menu = null
                for (i in 0 until n) {
                    val b = a.button("", chip = true) { pick(i) }.apply { isSingleLine = true; setPadding(a.dp(4), 0, a.dp(4), 0) }
                    if (i > 0) (b.layoutParams as LayoutParams).leftMargin = a.dp(6)
                    chips += b
                    box.addView(b)
                }
            }
            chips.forEachIndexed { i, b ->
                b.text = opts.getJSONObject(i).optString("label")
                a.style(b, i == sel, true)
            }
        }

        private fun open() {
            val opts = options()
            val labels = (0 until opts.length()).map { opts.getJSONObject(it).optString("label") }
            val sel = if (o.isNull("selected")) -1 else o.optInt("selected", -1)
            AlertDialog.Builder(a)
                .setTitle(o.optString("title"))
                .setSingleChoiceItems(labels.toTypedArray(), sel) { d, w -> d.dismiss(); pick(w) }
                .show()
        }
    }

    inner class NudgeWidget : Widget() {
        private var o = JSONObject()
        private val name = a.label("", 11f, a.muted, bold = true).apply { gravity = Gravity.CENTER }
        private val note = a.label("", 11f, a.muted).apply { gravity = Gravity.CENTER }
        override val view: View = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            addView(name, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
            addView(LinearLayout(a).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(a.button("◀") { send(lines(o.optJSONArray("back"))) })
                addView(a.button("▶") { send(lines(o.optJSONArray("forward"))) }.also {
                    (it.layoutParams as LayoutParams).leftMargin = a.dp(6)
                })
            }, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = a.dp(6) })
            addView(note, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = a.dp(4) })
        }
        override fun update(o: JSONObject) {
            this.o = o
            name.text = o.optString("label")
            note.text = o.optString("note")
        }
    }

    inner class SliderWidget : Widget() {
        private var o = JSONObject()
        private var dragging = false
        private val name = a.label("", 13f, a.muted).apply { minWidth = a.dp(50) }
        private val value = a.label("", 13f, a.muted).apply { minWidth = a.dp(58); gravity = Gravity.END }
        private val bar = a.slider(100) {}.apply {
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                    if (fromUser) set(p)
                }
                override fun onStartTrackingTouch(s: SeekBar) { dragging = true }
                override fun onStopTrackingTouch(s: SeekBar) { dragging = false }
            })
        }
        override val view: View = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(name)
            addView(bar)
            addView(value)
        }

        private val min get() = o.optDouble("min", 0.0)
        private val step get() = o.optDouble("step", 1.0).takeIf { it > 0 } ?: 1.0

        private fun set(p: Int) {
            var v = min + p * step
            if (!o.isNull("snap")) {
                val s = o.optDouble("snap")
                if (abs(v - s) <= 2 * step) v = s
            }
            val text = if (step >= 1.0) v.roundToInt().toString() else "%.2f".format(java.util.Locale.ROOT, v)
            send(listOf(o.optString("send").replace("{v}", text)))
        }

        override fun update(o: JSONObject) {
            this.o = o
            name.text = o.optString("label")
            value.text = o.optString("text")
            bar.max = ((o.optDouble("max", 1.0) - min) / step).roundToInt().coerceAtLeast(1)
            if (!dragging) bar.progress = ((o.optDouble("value", min) - min) / step).roundToInt()
        }
    }

    inner class LoopViewWidget : Widget() {
        private var o = JSONObject()
        private val lv = LoopView(a).apply {
            onEdges = { x, y -> send(listOf(o.optString("edges").replace("{a}", "$x").replace("{b}", "$y"))) }
            onBarOne = { i -> send(listOf(o.optString("bar1").replace("{beat}", "$i"))) }
        }
        override val view: View = lv.apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, a.dp(118))
            minimumHeight = a.dp(118)
        }
        override fun update(o: JSONObject) { this.o = o }

        fun poll() {
            if (lv.visibility != View.VISIBLE || (lv.parent as? View)?.isShown != true) return
            val bins = (lv.width / (3 * a.resources.displayMetrics.density)).toInt().coerceIn(40, 600)
            val (v, lag) = loopData(bins) ?: return
            lv.update(v, lag)
        }
    }

    /** The saved loops by recording, with a search; tap one to go back there,
     *  long-press to rename or delete it. */
    inner class BookmarksWidget : Widget() {
        private var o = JSONObject()
        private var shown = ""
        private val title = a.label("", 22f, bold = true)
        private val list = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        private val search = EditText(a).apply {
            setHintTextColor(a.muted)
            setTextColor(a.ink)
            textSize = 15f
            isSingleLine = true
            background = a.rounded(a.surface, 12)
            setPadding(a.dp(14), 0, a.dp(14), 0)
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
                override fun onTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
                override fun afterTextChanged(s: Editable?) { update(o) }
            })
        }
        override val view: View = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            addView(title, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { bottomMargin = a.dp(8) })
            addView(search, LayoutParams(LayoutParams.MATCH_PARENT, a.dp(46)))
            addView(list)
        }

        override fun update(o: JSONObject) {
            this.o = o
            title.text = o.optString("title")
            search.hint = o.optString("search_hint")
            val q = search.text.toString().trim().lowercase()
            val groups = o.optJSONArray("groups") ?: JSONArray()
            val key = q + "\n" + groups.toString()
            if (key == shown) return
            shown = key
            list.removeAllViews()
            if (groups.length() == 0) {
                list.addView(a.label(o.optString("empty"), 14f, a.muted).apply { setPadding(a.dp(4), a.dp(16), a.dp(4), 0) })
                return
            }
            var any = false
            for (g in 0 until groups.length()) {
                val grp = groups.getJSONObject(g)
                val marks = grp.getJSONArray("marks")
                val keep = (0 until marks.length()).map { marks.getJSONObject(it) }
                    .filter { q.isEmpty() || it.optString("search").contains(q) || grp.optString("title").lowercase().contains(q) }
                if (keep.isEmpty()) continue
                any = true
                list.addView(card(grp.optString("title"), keep), LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = a.dp(12) })
            }
            if (!any) list.addView(a.label(o.optString("no_match"), 14f, a.muted).apply { setPadding(a.dp(4), a.dp(16), a.dp(4), 0) })
        }

        private fun card(heading: String, marks: List<JSONObject>): View {
            val (song, by) = heading.split(" — ", limit = 2).let { it[0] to it.getOrElse(1) { "" } }
            val initials = song.split(" ").filter { it.isNotEmpty() }.take(2).joinToString("") { it.take(1).uppercase() }
            val badge = a.label(initials, 16f, a.accent, bold = true).apply {
                gravity = Gravity.CENTER
                background = a.rounded(Color.rgb(0x1f, 0x3d, 0x55), 10)
            }
            val names = LinearLayout(a).apply {
                orientation = LinearLayout.VERTICAL
                addView(a.label(song, 16f, bold = true).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END })
                if (by.isNotEmpty()) addView(a.label(by, 13f, a.muted).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END })
            }
            val head = LinearLayout(a).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(a.dp(14), a.dp(14), a.dp(14), a.dp(10))
                addView(badge, LayoutParams(a.dp(44), a.dp(44)))
                addView(names, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f).apply { leftMargin = a.dp(12) })
            }
            return LinearLayout(a).apply {
                orientation = LinearLayout.VERTICAL
                background = a.rounded(a.surface, 18)
                addView(head)
                for (m in marks) addView(row(m))
            }
        }

        private fun row(m: JSONObject): View {
            val text = a.label(m.optString("label"), 15f).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END }
            return LinearLayout(a).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(a.dp(14), a.dp(10), a.dp(14), a.dp(10))
                minimumHeight = a.dp(52)
                isClickable = true
                setBackgroundResource(android.R.drawable.list_selector_background)
                addView(text, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
                addView(a.label("▶", 16f, a.accent))
                setOnClickListener { send(lines(m.optJSONArray("play"))) }
                setOnLongClickListener { menu(m); true }
            }
        }

        private fun menu(m: JSONObject) {
            AlertDialog.Builder(a)
                .setTitle(m.optString("label"))
                .setItems(arrayOf("Rename", "Delete")) { _, which ->
                    if (which == 0) ask(m.optJSONObject("rename_ask") ?: JSONObject(), lines(m.optJSONArray("rename")))
                    else AlertDialog.Builder(a)
                        .setTitle("Delete this loop?")
                        .setMessage(m.optString("label"))
                        .setPositiveButton("Delete") { _, _ -> send(lines(m.optJSONArray("delete"))) }
                        .setNegativeButton("Cancel", null)
                        .show()
                }
                .show()
        }
    }
}
