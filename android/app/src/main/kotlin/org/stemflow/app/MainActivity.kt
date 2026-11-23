package org.stemflow.app

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.app.StatusBarManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.Icon
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.LinearLayout.LayoutParams
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Three tabs over a now-playing strip:
 *  - Stems: on/off, which stem, the mix (as the GNOME menu), model and source;
 *  - Loops: loop the last bars, the loop view and its edits, meter and feel;
 *  - Bookmarks: saved loops grouped by recording; tap to go back there.
 * The strip says what is playing and whether it's live or a loop, with Save and Live
 * (or a quick "Loop 4") always at hand.
 *
 * Everything about looping (the Loops and Bookmarks tabs, the strip's Pause and Save,
 * the stems-only switch) is the loop panel shared with the desktop
 * (src/loopdeck/panel.rs), drawn by LoopPanel.kt; this screen only says where each of
 * its rows goes.
 */
class MainActivity : Activity() {
    // palette: the logo's water blue (lightened to read on dark) on navy ink; its leaf
    // green marks the bar lines in the loop view
    internal val bg = Color.rgb(0x0e, 0x18, 0x20)
    internal val surface = Color.rgb(0x16, 0x25, 0x2f)
    internal val raised = Color.rgb(0x21, 0x36, 0x45)
    internal val line = Color.rgb(0x2d, 0x47, 0x57)
    internal val ink = Color.rgb(0xea, 0xf2, 0xf7)
    internal val muted = Color.rgb(0x9f, 0xb6, 0xc6)
    internal val accent = Color.rgb(0x6c, 0xb4, 0xe8)
    internal val onAccent = Color.rgb(0x06, 0x12, 0x1b)
    internal val bar = Color.rgb(0x0b, 0x14, 0x1b)

    private val ui = Handler(Looper.getMainLooper())
    private var dragging = false

    // tabs
    private val pages = mutableListOf<View>()
    private val tabButtons = mutableListOf<TextView>()
    private var tab = 1

    // Stems
    private lateinit var power: Switch
    private lateinit var onLine: TextView
    private lateinit var stemChips: LinearLayout
    private var shownStems = listOf<String>()
    private lateinit var desc: TextView
    private lateinit var spinner: ProgressBar
    private lateinit var balance: SeekBar
    private lateinit var boost: SeekBar
    private lateinit var boostValue: TextView
    private lateinit var noLabel: TextView
    private val snaps = mutableMapOf<String, Button>()
    // bass + vocals: three plain levels instead of the solo-to-none line and boost
    private lateinit var oneRows: View
    private lateinit var threeRows: View
    private lateinit var bassBar: SeekBar
    private lateinit var vocalsBar: SeekBar
    private lateinit var bandBar: SeekBar
    private lateinit var bassValue: TextView
    private lateinit var vocalsValue: TextView
    private lateinit var bandValue: TextView
    private lateinit var modelPicker: Picker
    private lateinit var monoIn: Button
    private lateinit var monoOut: Button
    private var modelIds = listOf<String>()
    private lateinit var sourcePicker: Picker
    private var sourcePkgs = listOf<String?>()
    private lateinit var health: TextView

    // Loops: the loop panel's rows go in these
    private lateinit var loopDesc: TextView
    private lateinit var loopsBox: LinearLayout
    private lateinit var bookmarksBox: LinearLayout
    private lateinit var stemsLoopBox: LinearLayout
    private lateinit var nowBox: LinearLayout
    private lateinit var stripBox: LinearLayout
    private lateinit var panel: LoopPanel

    // now playing
    private lateinit var npOff: TextView
    private lateinit var npLive: Button
    private lateinit var npLoop: Button

    private val tick = object : Runnable {
        override fun run() { refresh(); ui.postDelayed(this, 300) }
    }
    private val loopTick = object : Runnable {
        override fun run() { panel.tick(); ui.postDelayed(this, 40) }
    }

    internal fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    // ---------------------------------------------------------------- building blocks

    internal fun rounded(color: Int, radius: Int, stroke: Int = 0) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radius).toFloat()
        if (stroke != 0) setStroke(dp(1), stroke)
    }

    internal fun label(s: String, size: Float = 15f, color: Int = ink, bold: Boolean = false) = TextView(this).apply {
        text = s
        textSize = size
        setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun lp(w: Int = LayoutParams.MATCH_PARENT, h: Int = LayoutParams.WRAP_CONTENT, weight: Float = 0f) =
        LayoutParams(w, h, weight)

    private fun margins(v: View, l: Int = 0, t: Int = 0, r: Int = 0, b: Int = 0): View {
        val p = (v.layoutParams as? LayoutParams) ?: lp()
        p.setMargins(dp(l), dp(t), dp(r), dp(b))
        v.layoutParams = p
        return v
    }

    private fun row(vararg views: View, gap: Int = 8) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        views.forEachIndexed { i, v ->
            addView(v)
            if (i > 0) (v.layoutParams as? LayoutParams)?.leftMargin = dp(gap)
        }
    }

    private fun column(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        views.forEach { addView(it) }
    }

    private fun card(vararg views: View) = column(*views).apply {
        background = rounded(surface, 18)
        setPadding(dp(16), dp(14), dp(16), dp(14))
    }

    /** A filled button (`primary`: the accent) or an outlined chip. */
    internal fun button(s: String, primary: Boolean = false, chip: Boolean = false, weight: Float = 1f, onClick: () -> Unit) =
        Button(this).apply {
            text = s
            isAllCaps = false
            textSize = if (chip) 13f else 14f
            stateListAnimator = null
            minHeight = 0
            minimumHeight = 0
            setPadding(dp(14), 0, dp(14), 0)
            layoutParams = lp(if (weight > 0) 0 else LayoutParams.WRAP_CONTENT, dp(if (chip) 38 else 46), weight)
            style(this, primary, chip)
            setOnClickListener { onClick(); refresh() }
        }

    internal fun style(b: Button, on: Boolean, chip: Boolean) {
        b.background = when {
            on -> rounded(accent, if (chip) 19 else 12)
            chip -> rounded(Color.TRANSPARENT, 19, line)
            else -> rounded(raised, 12)
        }
        b.setTextColor(if (on) onAccent else ink)
    }

    internal fun slider(max: Int, onUser: (Int) -> Unit) = SeekBar(this).apply {
        this.max = max
        layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f)
        progressTintList = ColorStateList.valueOf(accent)
        thumbTintList = ColorStateList.valueOf(accent)
        progressBackgroundTintList = ColorStateList.valueOf(raised)
        setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                if (fromUser) { onUser(p); Control.save(this@MainActivity); refresh() }
            }
            override fun onStartTrackingTouch(s: SeekBar) { dragging = true }
            override fun onStopTrackingTouch(s: SeekBar) { dragging = false }
        })
    }

    /**
     * A choice shown as "value ▾" that opens a list (the platform dropdown's box and
     * type don't fit this screen).
     */
    private inner class Picker(
        val title: String,
        val items: () -> List<String>,
        val selected: () -> Int,
        /** true: a plain value at the right of a settings row; false: a pill. */
        val plain: Boolean = false,
        val onPick: (Int) -> Unit,
    ) {
        val view: Button = button("", chip = true, weight = if (plain) 1f else 0f) { open() }.apply {
            isSingleLine = true
            ellipsize = android.text.TextUtils.TruncateAt.END
            if (plain) {
                gravity = Gravity.CENTER_VERTICAL or Gravity.END
                textSize = 15f
                setPadding(dp(8), 0, 0, 0)
            }
        }

        init {
            if (plain) view.background = null
        }

        private fun open() {
            val list = items()
            if (list.isEmpty()) return
            AlertDialog.Builder(this@MainActivity)
                .setTitle(title)
                .setSingleChoiceItems(list.toTypedArray(), selected()) { d, w ->
                    d.dismiss(); onPick(w); refresh()
                }
                .show()
        }

        fun update() {
            view.text = (items().getOrNull(selected()) ?: "—") + "  ▾"
        }
    }

    private fun header(title: String, right: View) = row(
        label(title, 22f, bold = true).apply { layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f) },
        right,
    ).apply { setPadding(dp(20), dp(18), dp(20), dp(6)) }

    private fun page(vararg views: View) = ScrollView(this).apply {
        isFillViewport = true
        addView(column(*views).apply { setPadding(0, 0, 0, dp(16)) })
    }

    private fun pad(v: View, h: Int = 16): View = margins(v, h, 12, h, 0)

    // ---------------------------------------------------------------- the screens

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.init(this)
        Control.load(this)
        Control.loadBeats(this)
        tab = getPreferences(Context.MODE_PRIVATE).getInt("tab", 1)
        window.statusBarColor = bg
        window.navigationBarColor = bar

        val pagesFrame = FrameLayout(this).apply { layoutParams = lp(LayoutParams.MATCH_PARENT, 0, 1f) }
        listOf(stemsPage(), loopsPage(), bookmarksPage()).forEach {
            pages += it
            pagesFrame.addView(it, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        }

        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            setBackgroundColor(bg)
            addView(pagesFrame)
            addView(nowPlaying())
            addView(tabBar())
        })
        panel = LoopPanel(this, ::placeRow, ::sendLoop, { bins ->
            val h = CaptureService.looper
            if (h == 0L || !CaptureService.running) null
            else Native.looperView(h, bins).takeIf { it.size >= 14 }?.let { it to CaptureService.unheardFrames() * Control.speed }
        }, compact = setOf("now", "strip"))
        showTab(tab)
    }

    /** Where the loop panel's rows go: the strip, the Stems tab, Bookmarks, Loops. */
    private fun placeRow(id: String): LinearLayout = when (id) {
        "now" -> nowBox
        "strip" -> stripBox
        "bookmarks" -> bookmarksBox
        "stems_only" -> stemsLoopBox
        else -> loopsBox
    }

    /** The loop panel's lines, to the deck. */
    private fun sendLoop(lines: List<String>) {
        for (l in lines) {
            val w = l.removePrefix("loop ").trim()
            if (w.startsWith("save") && !NowPlaying.granted(this)) { askAccess(); return }
            val off = w.startsWith("recall") && !CaptureService.running
            val err = Loops.send(this, l)
            if (err != null) { Toast.makeText(this, err, Toast.LENGTH_LONG).show(); return }
            if (off) start() // switch on; the bookmark is recalled once the audio flows
            if (w.startsWith("recall") || w.firstOrNull()?.isDigit() == true) showTab(1)
        }
        refresh()
    }

    private fun stemsPage(): View {
        power = Switch(this).apply {
            thumbTintList = ColorStateList.valueOf(accent)
            setOnClickListener { if (isChecked) start() else stop() }
        }
        onLine = label("", 13f, muted).apply { setPadding(dp(20), 0, dp(20), 0) }
        stemChips = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val chipsScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(stemChips)
            setPadding(dp(20), 0, dp(20), 0)
            clipToPadding = false
        }

        desc = label("", 17f, bold = true)
        spinner = ProgressBar(this, null, android.R.attr.progressBarStyleSmall).apply {
            isIndeterminate = true
            visibility = View.GONE
        }
        balance = slider(200) { p ->
            var b = p / 100f - 1f
            if (abs(b) < 0.04f) b = 0f // detent at the original mix
            Control.setBalance(b)
        }
        // Boost the target past the original, up to twice as loud; the limiter keeps
        // the result from clipping.
        boost = slider(100) { p -> Control.setLevels(1f + p / 100f * (Control.MAX_BASS - 1f), Control.band) }
        boostValue = label("100%", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        noLabel = label("No bass", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        val snapRow = row()
        for ((t, name) in listOf("Bass solo" to "solo", "Full mix" to "full", "No bass" to "nobass", "No vocals" to "novocals")) {
            val b = button(t) { Control.snap(name); Control.save(this) }
            snaps[name] = b
            snapRow.addView(b)
            if (snapRow.childCount > 1) (b.layoutParams as LayoutParams).leftMargin = dp(8)
        }
        oneRows = column(
            margins(row(label("Solo", 13f, muted).apply { minWidth = dp(44) }, balance, noLabel, gap = 4), t = 14),
            margins(row(label("Boost", 13f, muted).apply { minWidth = dp(44) }, boost, boostValue, gap = 4), t = 10),
        )
        bassBar = slider(200) { p -> Control.setLevels(p / 100f, Control.band) }
        vocalsBar = slider(100) { p -> Control.setSecond(p / 100f) }
        bandBar = slider(100) { p -> Control.setLevels(Control.bass, p / 100f) }
        bassValue = label("", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        vocalsValue = label("", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        bandValue = label("", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        threeRows = column(
            margins(row(label("Bass", 13f, muted).apply { minWidth = dp(54) }, bassBar, bassValue, gap = 4), t = 14),
            margins(row(label("Vocals", 13f, muted).apply { minWidth = dp(54) }, vocalsBar, vocalsValue, gap = 4), t = 10),
            margins(row(label("Band", 13f, muted).apply { minWidth = dp(54) }, bandBar, bandValue, gap = 4), t = 10),
        )
        val mix = card(
            row(spinner, desc, gap = 6),
            oneRows,
            threeRows,
            margins(snapRow, t = 14),
        )

        modelPicker = Picker("Model", { Models.choices(this).map { it.label } },
            { Models.choices(this).indexOfFirst { it.id == Models.current(this)?.id } }, plain = true) { pos ->
            Models.choices(this).getOrNull(pos)?.let { if (it.id != Models.current(this)?.id) Models.select(this, it) }
        }
        sourcePicker = Picker("Isolate from", { listOf("Whatever is playing") + sourceLabels },
            { sourcePkgs.indexOf(Sources.current(this)?.pkg).coerceAtLeast(0) }, plain = true) { pos ->
            val pkg = sourcePkgs.getOrNull(pos)
            if (pkg != Sources.current(this)?.pkg) Sources.select(this, Sources.apps(this).firstOrNull { it.pkg == pkg })
        }
        fun divider() = View(this).apply { setBackgroundColor(raised); layoutParams = lp(LayoutParams.MATCH_PARENT, dp(1)) }
        monoIn = button("Input", chip = true, weight = 0f) {
            Models.current(this)?.let { Models.setMono(this, it, !Models.monoIn(this, it), Models.monoOut(this, it)) }
        }
        monoOut = button("Output", chip = true, weight = 0f) {
            Models.current(this)?.let { Models.setMono(this, it, Models.monoIn(this, it), !Models.monoOut(this, it)) }
        }
        val pickers = column(
            row(label("Model", 14f, muted).apply { minWidth = dp(64) }, modelPicker.view).apply { setPadding(dp(16), dp(4), dp(12), dp(4)) },
            divider(),
            row(label("Mono", 14f, muted).apply { minWidth = dp(64) }, monoIn, monoOut).apply {
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(16), dp(10), dp(12), dp(10))
            },
            divider(),
            row(label("From", 14f, muted).apply { minWidth = dp(64) }, sourcePicker.view).apply { setPadding(dp(16), dp(4), dp(12), dp(4)) },
        ).apply { background = rounded(surface, 18) }
        // the loop panel's "stems only when looping" switch goes here
        stemsLoopBox = column()
        health = label("", 12f, muted).apply { setPadding(dp(20), dp(12), dp(20), 0) }
        val extras = row(
            button("Add quick tile", chip = true, weight = 0f) { addTile() },
            button("Use screen capture instead", chip = true, weight = 0f) { startProjection() },
        ).apply { setPadding(dp(16), dp(14), dp(16), 0) }

        return page(
            header("Stems", power),
            onLine,
            label("Isolate", 13f, muted).apply { setPadding(dp(20), dp(16), dp(20), dp(8)) },
            chipsScroll,
            pad(mix),
            pad(pickers),
            margins(stemsLoopBox, t = 4),
            health,
            extras,
        )
    }

    private fun loopsPage(): View {
        loopDesc = label("", 13f, muted)
        loopsBox = column()
        return page(header("Loops", loopDesc), loopsBox)
    }

    private fun bookmarksPage(): View {
        bookmarksBox = column().apply { setPadding(dp(4), dp(18), dp(4), 0) }
        val foot = label(
            "Tap a loop to go back there: the player seeks a few seconds before it, you hear the run-up, " +
                "and the loop comes back with its bars. Long-press to rename or delete.",
            12f, muted,
        ).apply { setPadding(dp(20), dp(14), dp(20), 0) }
        return page(bookmarksBox, foot)
    }

    private fun nowPlaying(): View {
        npOff = label("", 14f, bold = true).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END }
        nowBox = column()
        stripBox = column()
        npLive = button("Live", primary = true, chip = true, weight = 0f) { sendLoop(listOf("loop live")) }
        npLoop = button("Loop 4", primary = true, chip = true, weight = 0f) { sendLoop(listOf("loop 4")) }
        val texts = column(npOff, nowBox).apply { layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f) }
        val icon = android.widget.ImageView(this).apply {
            setImageResource(R.drawable.ic_now_playing)
            imageTintList = ColorStateList.valueOf(accent)
            background = rounded(raised, 8)
            setPadding(dp(8), dp(8), dp(8), dp(8))
            layoutParams = lp(dp(36), dp(36))
        }
        return row(icon, texts, stripBox, npLive, npLoop, gap = 6).apply {
            background = rounded(surface, 14)
            setPadding(dp(14), dp(8), dp(10), dp(8))
            layoutParams = lp().apply { setMargins(dp(12), dp(6), dp(12), dp(8)) }
        }
    }

    private fun tabBar(): View {
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(bar)
            layoutParams = lp(LayoutParams.MATCH_PARENT, dp(62))
        }
        val icons = listOf(R.drawable.ic_tab_stems, R.drawable.ic_tab_loops, R.drawable.ic_tab_bookmarks)
        listOf("Stems", "Loops", "Bookmarks").forEachIndexed { i, name ->
            val t = label(name, 12f).apply {
                gravity = Gravity.CENTER
                layoutParams = lp(0, LayoutParams.MATCH_PARENT, 1f)
                setCompoundDrawablesWithIntrinsicBounds(0, icons[i], 0, 0)
                compoundDrawablePadding = dp(2)
                setPadding(0, dp(7), 0, dp(5))
                isClickable = true
                setOnClickListener { showTab(i) }
            }
            tabButtons += t
            bar.addView(t)
        }
        return bar
    }

    private fun showTab(i: Int) {
        tab = i
        getPreferences(Context.MODE_PRIVATE).edit().putInt("tab", i).apply()
        pages.forEachIndexed { k, v -> v.visibility = if (k == i) View.VISIBLE else View.GONE }
        tabButtons.forEachIndexed { k, t ->
            t.setTextColor(if (k == i) accent else muted)
            t.compoundDrawableTintList = ColorStateList.valueOf(if (k == i) accent else muted)
            t.typeface = if (k == i) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
    }

    override fun onResume() { super.onResume(); loadSources(); ui.post(tick); ui.post(loopTick) }
    override fun onPause() { super.onPause(); ui.removeCallbacks(tick); ui.removeCallbacks(loopTick) }

    private var sourceLabels = listOf<String>()

    private fun loadSources() {
        val apps = Sources.apps(this)
        sourcePkgs = listOf<String?>(null) + apps.map { it.pkg }
        sourceLabels = apps.map { it.label }
        sourcePicker.update()
    }

    // ---------------------------------------------------------------- refreshing

    private fun refresh() {
        val running = CaptureService.running
        power.isChecked = running
        val wanted = Models.current(this)
        style(monoIn, wanted != null && Models.monoIn(this, wanted), true)
        style(monoOut, wanted != null && Models.monoOut(this, wanted), true)
        val loading = running &&
            (CaptureService.loading || (wanted != null && wanted.file.path != CaptureService.loadedModel))
        spinner.visibility = if (loading) View.VISIBLE else View.GONE
        desc.text = if (loading) "Loading ${wanted?.label ?: "model"}…" else Control.describe()
        if (!dragging) {
            balance.progress = ((Control.balance() + 1f) * 100).roundToInt()
            boost.progress = ((Control.bass - 1f).coerceAtLeast(0f) / (Control.MAX_BASS - 1f) * 100).roundToInt()
        }
        boostValue.text = "${(maxOf(Control.bass, 1f) * 100).roundToInt()}%"
        val s = CaptureService.status
        val lines = s.lines()
        onLine.text = when {
            running -> "On · " + lines.first()
            s != "Off" -> s // an error stays on screen
            else -> "Off · ${Shizu.state()}"
        }
        health.text = if (running) lines.drop(1).joinToString(" · ") else ""
        val t = Control.target
        val three = Control.three
        val tc = if (three) "Bass" else t.replaceFirstChar { it.uppercase() }
        snaps["solo"]?.text = "$tc solo"
        snaps["nobass"]?.text = if (three) "No bass" else "No $t"
        snaps["novocals"]?.visibility = if (three) View.VISIBLE else View.GONE
        oneRows.visibility = if (three) View.GONE else View.VISIBLE
        threeRows.visibility = if (three) View.VISIBLE else View.GONE
        if (!dragging) {
            bassBar.progress = (Control.bass * 100).roundToInt()
            vocalsBar.progress = (Control.second * 100).roundToInt()
            bandBar.progress = (Control.band * 100).roundToInt()
        }
        bassValue.text = "${(Control.bass * 100).roundToInt()}%"
        vocalsValue.text = "${(Control.second * 100).roundToInt()}%"
        bandValue.text = "${(Control.band * 100).roundToInt()}%"
        noLabel.text = "No $t"
        for ((name, b) in snaps) style(b, Control.isSnap(name), false)
        loopDesc.text = Control.describe()

        // stem chips: one per stem with a model
        val have = Models.stems(this)
        if (have != shownStems) {
            shownStems = have
            stemChips.removeAllViews()
            for (stem in have) {
                val b = button(Models.stemLabel(stem), chip = true, weight = 0f) { Control.setTarget(this, stem) }
                b.tag = stem
                stemChips.addView(b)
                if (stemChips.childCount > 1) (b.layoutParams as LayoutParams).leftMargin = dp(8)
            }
            if (have.isEmpty()) stemChips.addView(label("No model on the phone", 14f, muted))
        }
        for (i in 0 until stemChips.childCount) (stemChips.getChildAt(i) as? Button)?.let { style(it, it.tag == t, true) }
        modelPicker.update() // the source picker updates on resume and when picked
        panel.render(Loops.panel(this))
        refreshNowPlaying()
    }

    /** The strip: the panel's song line, Pause and Save, and Live or a quick Loop 4. */
    private fun refreshNowPlaying() {
        val st = Loops.state()
        val on = st != null
        npOff.visibility = if (on) View.GONE else View.VISIBLE
        nowBox.visibility = if (on) View.VISIBLE else View.GONE
        stripBox.visibility = if (on) View.VISIBLE else View.GONE
        npOff.text = when {
            !CaptureService.running -> "Off · switch on under Stems"
            CaptureService.loopStatus.isNotEmpty() -> CaptureService.loopStatus
            else -> "Loading the bar finder…"
        }
        val looping = st?.optBoolean("playing") == true
        val busy = st?.optBoolean("recalling") == true || st?.optBoolean("finding") == true || st?.optBoolean("tail") == true
        npLive.visibility = if (looping || busy) View.VISIBLE else View.GONE
        npLoop.visibility = if (on && !looping && !busy) View.VISIBLE else View.GONE
    }

    // ---------------------------------------------------------------- actions

    /** Saving needs to know the song: ask for notification access first. */
    private fun askAccess() {
        AlertDialog.Builder(this)
            .setTitle("Allow notification access")
            .setMessage("Bookmarks need to know which song is playing and where, and to seek it later. " +
                "Android only lets an app read the player for that with notification access " +
                "(stemflow doesn't read your notifications).\n\nTurn on \"stemflow bookmarks\", then make the loop again.")
            .setPositiveButton("Open settings") { _, _ ->
                startActivity(Intent(android.provider.Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun notificationsOk(): Boolean {
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return true
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        return false
    }

    private fun start() {
        if (!notificationsOk()) return
        val go = {
            CaptureService.status = "Starting…"
            startForegroundService(Intent(this, CaptureService::class.java)); Unit
        }
        if (Shizu.available() && !Shizu.granted()) Shizu.request { go() } else go()
    }

    private fun stop() = stopService(Intent(this, CaptureService::class.java))

    /** For apps that allow playback capture: no helper needed, but the original keeps playing. */
    private fun startProjection() {
        if (CaptureService.running || !notificationsOk()) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 2)
            return
        }
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        @Suppress("DEPRECATION")
        startActivityForResult(mpm.createScreenCaptureIntent(), 2)
    }

    private fun addTile() {
        getSystemService(StatusBarManager::class.java).requestAddTileService(
            ComponentName(this, StemTile::class.java), "Bass isolation",
            Icon.createWithResource(this, R.drawable.ic_stemflow), mainExecutor,
        ) {}
    }

    @Deprecated("plain Activity, no androidx")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != 2 || resultCode != RESULT_OK || data == null) return
        startForegroundService(
            Intent(this, CaptureService::class.java).putExtra("code", resultCode).putExtra("data", data)
        )
    }
}
