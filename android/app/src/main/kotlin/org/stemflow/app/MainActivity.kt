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
import android.util.Log
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
 *  - Loops: loop the last 2/4/8 bars, the loop view and its edits, meter and feel;
 *  - Bookmarks: saved loops grouped by recording; tap to go back there.
 * The strip says what is playing and whether it's live or a loop, with Save and Live
 * (or a quick "Loop 4") always at hand.
 */
class MainActivity : Activity() {
    // palette: the logo's water blue (lightened to read on dark) on navy ink; its leaf
    // green marks the bar lines in the loop view
    private val bg = Color.rgb(0x0e, 0x18, 0x20)
    private val surface = Color.rgb(0x16, 0x25, 0x2f)
    private val raised = Color.rgb(0x21, 0x36, 0x45)
    private val line = Color.rgb(0x2d, 0x47, 0x57)
    private val ink = Color.rgb(0xea, 0xf2, 0xf7)
    private val muted = Color.rgb(0x9f, 0xb6, 0xc6)
    private val accent = Color.rgb(0x6c, 0xb4, 0xe8)
    private val onAccent = Color.rgb(0x06, 0x12, 0x1b)
    private val bar = Color.rgb(0x0b, 0x14, 0x1b)

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
    private var modelIds = listOf<String>()
    private lateinit var sourcePicker: Picker
    private var sourcePkgs = listOf<String?>()
    private lateinit var health: TextView

    // Loops
    private lateinit var loopDesc: TextView
    private val grabButtons = mutableMapOf<Int, Button>()
    private lateinit var loopInfo: TextView
    private lateinit var loopHint: TextView
    private lateinit var loopView: LoopView
    private lateinit var loopEdit: LinearLayout
    private lateinit var clickChip: Button
    private lateinit var meterPicker: Picker
    private lateinit var feelPicker: Picker
    private lateinit var speedBar: SeekBar
    private lateinit var speedValue: TextView
    private lateinit var pitchBar: SeekBar
    private lateinit var pitchValue: TextView
    private lateinit var keepChip: Button
    private var loopBars = 4

    // Bookmarks
    private lateinit var bookmarkCount: TextView
    private lateinit var bookmarkList: LinearLayout
    private lateinit var search: EditText
    private var shownBookmarks = ""

    // now playing
    private lateinit var npTitle: TextView
    private lateinit var npLine: TextView
    private lateinit var npSave: Button
    private lateinit var npLive: Button
    private lateinit var npLoop: Button
    private var songCheck = 0L
    private var song: NowPlaying.Song? = null
    private var songMs: Long? = null

    private val tick = object : Runnable {
        override fun run() { refresh(); ui.postDelayed(this, 300) }
    }
    private val loopTick = object : Runnable {
        override fun run() { refreshLoop(); ui.postDelayed(this, 40) }
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    // ---------------------------------------------------------------- building blocks

    private fun rounded(color: Int, radius: Int, stroke: Int = 0) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radius).toFloat()
        if (stroke != 0) setStroke(dp(1), stroke)
    }

    private fun label(s: String, size: Float = 15f, color: Int = ink, bold: Boolean = false) = TextView(this).apply {
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
    private fun button(s: String, primary: Boolean = false, chip: Boolean = false, weight: Float = 1f, onClick: () -> Unit) =
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
            setOnClickListener { onClick(); refresh(); refreshLoop() }
        }

    private fun style(b: Button, on: Boolean, chip: Boolean) {
        b.background = when {
            on -> rounded(accent, if (chip) 19 else 12)
            chip -> rounded(Color.TRANSPARENT, 19, line)
            else -> rounded(raised, 12)
        }
        b.setTextColor(if (on) onAccent else ink)
    }

    private fun slider(max: Int, onUser: (Int) -> Unit) = SeekBar(this).apply {
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
                    d.dismiss(); onPick(w); refresh(); refreshLoop()
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
        showTab(tab)
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
        val divider = View(this).apply { setBackgroundColor(raised); layoutParams = lp(LayoutParams.MATCH_PARENT, dp(1)) }
        val pickers = column(
            row(label("Model", 14f, muted).apply { minWidth = dp(64) }, modelPicker.view).apply { setPadding(dp(16), dp(4), dp(12), dp(4)) },
            divider,
            row(label("From", 14f, muted).apply { minWidth = dp(64) }, sourcePicker.view).apply { setPadding(dp(16), dp(4), dp(12), dp(4)) },
        ).apply { background = rounded(surface, 18) }
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
            health,
            extras,
        )
    }

    private fun loopsPage(): View {
        loopDesc = label("", 13f, muted)
        val grabs = row()
        for (n in listOf(2, 4, 8)) {
            val b = button("$n bars") { loopBars = n; CaptureService.loopLast(this, n) }
            grabButtons[n] = b
            grabs.addView(b)
            if (grabs.childCount > 1) (b.layoutParams as LayoutParams).leftMargin = dp(8)
        }
        grabs.setPadding(dp(16), 0, dp(16), 0)

        val edit = { op: Int, a: Int, b: Int, f: Float ->
            CaptureService.looper.takeIf { it != 0L }?.let { Native.looperEdit(it, op, a, b, f) }; Unit
        }
        loopInfo = label("", 15f, bold = true)
        loopHint = label("", 12f, muted)
        loopView = LoopView(this).apply {
            layoutParams = lp(LayoutParams.MATCH_PARENT, dp(118))
            onEdges = { a, b -> edit(3, a, b, 0f); refreshLoop() }
            onBarOne = { i -> edit(5, i, 0, 0f); refreshLoop() }
        }
        val nudge = row(
            button("◀ beat") { edit(2, -1, 0, 0f) },
            button("beat ▶") { edit(2, 1, 0, 0f) },
            button("− bar") { loopBars = (loopBars - 1).coerceAtLeast(1); edit(4, loopBars, 0, 0f) },
            button("+ bar") { loopBars += 1; edit(4, loopBars, 0, 0f) },
        )
        loopEdit = column(margins(loopView, t = 12), margins(nudge, t = 12))
        val loopCard = card(loopInfo, loopHint, loopEdit)

        val meters = listOf(0, 2, 3, 4, 5, 6, 7)
        meterPicker = Picker("Beats per bar", { meters.map { if (it == 0) "Auto meter" else "$it/4" } },
            { meters.indexOf(Control.meter).coerceAtLeast(0) }) { pos ->
            if (Control.meter != meters[pos]) {
                Control.meter = meters[pos]; Control.saveBeats(this)
                edit(7, Control.meter, 0, Control.feel)
            }
        }
        val feels = listOf(1f, 0.5f, 2f)
        feelPicker = Picker("Feel", { listOf("As heard", "Half time", "Double time") },
            { feels.indexOf(Control.feel).coerceAtLeast(0) }) { pos ->
            if (Control.feel != feels[pos]) {
                Control.feel = feels[pos]; Control.saveBeats(this)
                edit(7, Control.meter, 0, Control.feel)
            }
        }
        clickChip = button("Click", chip = true, weight = 0f) {
            Control.click = !Control.click; Control.saveBeats(this)
            edit(6, if (Control.click) 1 else 0, 0, 0f)
        }
        val feel = row(meterPicker.view, feelPicker.view, clickChip).apply { setPadding(dp(16), dp(14), dp(16), 0) }
        // pills as in the mockup: sized to their text, not stretched across the row

        // Practice speed (40-120%) and transposition (±12 semitones), for the loop.
        speedBar = slider(80) { p -> Control.speed = (40 + p) / 100f; applySpeed() }
        speedValue = label("100%", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        pitchBar = slider(24) { p -> Control.semis = (p - 12).toFloat(); applySpeed() }
        pitchValue = label("0 st", 13f, muted).apply { minWidth = dp(58); gravity = Gravity.END }
        keepChip = button("Keep pitch", chip = true, weight = 0f) { Control.keepPitch = !Control.keepPitch; applySpeed() }
        val speedRows = column(
            row(label("Speed", 13f, muted).apply { minWidth = dp(50) }, speedBar, speedValue, gap = 4),
            margins(row(label("Pitch", 13f, muted).apply { minWidth = dp(50) }, pitchBar, pitchValue, gap = 4), t = 6),
            margins(row(
                button("75%", chip = true, weight = 0f) { Control.speed = 0.75f; applySpeed() },
                button("50%", chip = true, weight = 0f) { Control.speed = 0.5f; applySpeed() },
                keepChip,
                button("Reset", chip = true, weight = 0f) { Control.speed = 1f; Control.semis = 0f; applySpeed() },
            ), t = 10),
        ).apply { setPadding(dp(20), dp(14), dp(16), 0) }

        val top = button("From the top") { edit(1, 0, 0, 0f) }
        val actions = row(top).apply { setPadding(dp(16), dp(14), dp(16), 0) }

        return page(
            header("Loops", loopDesc),
            label("Loop the last", 13f, muted).apply { setPadding(dp(20), dp(6), dp(20), dp(8)) },
            grabs,
            pad(loopCard),
            feel,
            speedRows,
            actions,
        )
    }

    private fun bookmarksPage(): View {
        bookmarkCount = label("", 13f, muted)
        search = EditText(this).apply {
            hint = "Search songs and loops"
            setHintTextColor(muted)
            setTextColor(ink)
            textSize = 15f
            isSingleLine = true
            background = rounded(surface, 12)
            setPadding(dp(14), 0, dp(14), 0)
            layoutParams = lp(LayoutParams.MATCH_PARENT, dp(46))
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) { refreshBookmarks(force = true) }
            })
        }
        bookmarkList = column()
        val foot = label(
            "Tap a loop to go back there: the player seeks a few seconds before it, you hear the run-up, " +
                "and the loop comes back with its bars. Long-press to rename or delete.",
            12f, muted,
        ).apply { setPadding(dp(20), dp(14), dp(20), 0) }
        return page(header("Bookmarks", bookmarkCount), pad(search), bookmarkList, foot)
    }

    private fun nowPlaying(): View {
        npTitle = label("", 14f, bold = true).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END }
        npLine = label("", 12f, muted).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END }
        npSave = button("Save", chip = true, weight = 0f) { saveLoop() }
        npLive = button("Live", primary = true, chip = true, weight = 0f) { CaptureService.goLive(this) }
        npLoop = button("Loop 4", primary = true, chip = true, weight = 0f) { loopBars = 4; CaptureService.loopLast(this, 4); showTab(1) }
        val texts = column(npTitle, npLine).apply { layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f) }
        val icon = android.widget.ImageView(this).apply {
            setImageResource(R.drawable.ic_now_playing)
            imageTintList = ColorStateList.valueOf(accent)
            background = rounded(raised, 8)
            setPadding(dp(8), dp(8), dp(8), dp(8))
            layoutParams = lp(dp(36), dp(36))
        }
        return row(icon, texts, npSave, npLive, npLoop, gap = 10).apply {
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
        if (i == 2) refreshBookmarks(force = true)
    }

    override fun onResume() { super.onResume(); loadSources(); refreshBookmarks(force = true); ui.post(tick); ui.post(loopTick) }
    override fun onPause() { super.onPause(); ui.removeCallbacks(tick); ui.removeCallbacks(loopTick) }

    private var sourceLabels = listOf<String>()

    private fun loadSources() {
        val apps = Sources.apps(this)
        sourcePkgs = listOf<String?>(null) + apps.map { it.pkg }
        sourceLabels = apps.map { it.label }
        sourcePicker.update()
    }

    /** Speed and pitch to the looper (and the screen), saved for next time. */
    private fun applySpeed() {
        Control.saveBeats(this)
        CaptureService.looper.takeIf { it != 0L }?.let {
            Native.looperEdit(it, 9, (Control.semis * 100).roundToInt(), if (Control.keepPitch) 1 else 0, Control.speed)
        }
        refresh()
    }

    // ---------------------------------------------------------------- refreshing

    private fun refresh() {
        val running = CaptureService.running
        power.isChecked = running
        val wanted = Models.current(this)
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
        meterPicker.update()
        feelPicker.update()
        style(clickChip, Control.click, true)
        if (!dragging) {
            speedBar.progress = (Control.speed * 100).roundToInt() - 40
            pitchBar.progress = Control.semis.roundToInt() + 12
        }
        speedValue.text = "${(Control.speed * 100).roundToInt()}%"
        pitchValue.text = "%+d st".format(Control.semis.roundToInt()).replace("+0", "0")
        style(keepChip, Control.keepPitch, true)
        refreshNowPlaying()
    }

    /** The strip: the song and where it is, or the loop; Save + Live, or Loop 4. */
    private fun refreshNowPlaying() {
        val now = System.currentTimeMillis()
        if (now - songCheck > 1000) {
            songCheck = now
            val ctl = NowPlaying.controller(this)
            song = ctl?.let { NowPlaying.song(it) } ?: CaptureService.anchor?.song
            songMs = ctl?.let { NowPlaying.positionMs(it) }
        }
        val h = CaptureService.looper
        val v = if (h != 0L && CaptureService.running) Native.looperView(h, 1) else FloatArray(0)
        val looping = v.size >= 14 && v[0] > 0f
        val range = CaptureService.loopSongRange()
        npTitle.text = song?.title ?: if (CaptureService.running) "stemflow" else "Off"
        npLine.text = when {
            CaptureService.loopStatus.isNotEmpty() -> CaptureService.loopStatus
            v.size >= 2 && v[1] > 0f -> "Finding the bars…"
            looping && range != null -> "Looping %s–%s · paused".format(Bookmarks.time(range[0]), Bookmarks.time(range[1]))
            looping -> "Looping · %d bars".format(v[4].toInt())
            CaptureService.running -> listOfNotNull("Live", song?.artist?.ifEmpty { null }, songMs?.let { Bookmarks.time(it) }).joinToString(" · ")
            else -> "Switch on under Stems"
        }
        npSave.visibility = if (v.size >= 14) View.VISIBLE else View.GONE
        npLive.visibility = if (looping) View.VISIBLE else View.GONE
        npLoop.visibility = if (CaptureService.running && !looping && h != 0L) View.VISIBLE else View.GONE
    }

    /** The loop card: what was found, or why not. */
    private fun refreshLoop() {
        val h = CaptureService.looper
        val bins = (loopView.width / (3 * resources.displayMetrics.density)).toInt().coerceIn(40, 600)
        val v = if (h != 0L && CaptureService.running) Native.looperView(h, bins) else FloatArray(0)
        val has = v.size >= 14
        loopEdit.visibility = if (has) View.VISIBLE else View.GONE
        if (has) {
            loopView.update(v)
            loopBars = v[4].toInt().coerceAtLeast(1)
        }
        for ((n, b) in grabButtons) style(b, has && v[0] > 0f && n == loopBars, false)
        val finding = v.size >= 2 && v[1] > 0f
        loopInfo.text = when {
            has -> "%s · %d bar%s of %d/4".format(if (v[0] > 0f) "Looping" else "Kept", v[4].toInt(),
                if (v[4].toInt() == 1) "" else "s", v[3].toInt())
            finding -> "Finding the bars…"
            CaptureService.running -> if (h == 0L) "Loading the bar finder…" else "No loop yet"
            else -> "Switch isolation on under Stems"
        }
        loopHint.text = when {
            CaptureService.loopStatus.isNotEmpty() -> CaptureService.loopStatus
            has -> "%.0f bpm · drag an end · long-press a beat for bar 1".format(v[2])
            CaptureService.running && h != 0L -> "Press a length to loop what you just heard"
            else -> ""
        }
    }

    /** Loops grouped by recording; tap to go back, long-press to rename or delete. */
    private fun refreshBookmarks(force: Boolean = false) {
        val q = search.text.toString().trim().lowercase()
        val all = Bookmarks.all(this)
        val list = all.filter {
            q.isEmpty() || listOf(it.name, it.title, it.artist, it.album).any { s -> s.lowercase().contains(q) }
        }
        val key = q + "|" + list.joinToString(",") { "${it.created}:${it.name}" }
        if (!force && key == shownBookmarks) return
        shownBookmarks = key
        val songs = all.groupBy { "${it.title}|${it.artist}|${it.album}" }.size
        bookmarkCount.text = "%d loop%s · %d song%s".format(all.size, if (all.size == 1) "" else "s", songs, if (songs == 1) "" else "s")
        bookmarkList.removeAllViews()
        if (all.isEmpty()) {
            bookmarkList.addView(label("Save a loop (Save, in the strip below) to come back to it later.", 14f, muted)
                .apply { setPadding(dp(20), dp(16), dp(20), 0) })
            return
        }
        val groups = list.groupBy { "${it.title}|${it.artist}|${it.album}" }
            .values.sortedByDescending { g -> g.maxOf { it.created } }
        for (g in groups) {
            val first = g.first()
            val initials = first.title.split(" ").filter { it.isNotEmpty() }.take(2).joinToString("") { it.take(1).uppercase() }
            val badge = label(initials, 16f, accent, bold = true).apply {
                gravity = Gravity.CENTER
                background = rounded(Color.rgb(0x1f, 0x3d, 0x55), 10)
                layoutParams = lp(dp(44), dp(44))
            }
            val names = column(
                label(first.title, 16f, bold = true).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END },
                label(listOf(first.artist, first.album).filter { it.isNotEmpty() }.joinToString(" · "), 13f, muted)
                    .apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END },
            ).apply { layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f) }
            val card = column(row(badge, names, gap = 12).apply { setPadding(dp(14), dp(14), dp(14), dp(10)) }).apply {
                background = rounded(surface, 18)
            }
            for (b in g.sortedBy { it.startMs }) card.addView(bookmarkRow(b))
            bookmarkList.addView(pad(card))
        }
    }

    private fun bookmarkRow(b: Bookmarks.Bookmark): View {
        val at = label(Bookmarks.time(b.startMs), 13f, accent, bold = true).apply { minWidth = dp(52) }
        val texts = column(
            label(b.name, 15f).apply { isSingleLine = true; ellipsize = android.text.TextUtils.TruncateAt.END },
            label("%d bars · %d/4 · %s–%s".format(b.bars, b.meter, Bookmarks.time(b.startMs), Bookmarks.time(b.endMs)), 12f, muted),
        ).apply { layoutParams = lp(0, LayoutParams.WRAP_CONTENT, 1f) }
        val play = label("▶", 16f, accent)
        return row(at, texts, play, gap = 10).apply {
            setPadding(dp(14), dp(10), dp(14), dp(10))
            minimumHeight = dp(56)
            isClickable = true
            setBackgroundResource(android.R.drawable.list_selector_background)
            setOnClickListener { CaptureService.recall(this@MainActivity, b); showTab(1) }
            setOnLongClickListener { bookmarkMenu(b); true }
        }
    }

    private fun bookmarkMenu(b: Bookmarks.Bookmark) {
        AlertDialog.Builder(this)
            .setTitle(b.name)
            .setItems(arrayOf("Rename", "Delete")) { _, which ->
                if (which == 0) {
                    val input = EditText(this).apply { setText(b.name); selectAll() }
                    AlertDialog.Builder(this)
                        .setTitle("Rename")
                        .setView(input)
                        .setPositiveButton("Rename") { _, _ ->
                            Bookmarks.rename(this, b, input.text.toString().ifBlank { b.name }); refreshBookmarks(true)
                        }
                        .setNegativeButton("Cancel", null)
                        .show()
                } else {
                    AlertDialog.Builder(this)
                        .setTitle("Delete \"${b.name}\"?")
                        .setPositiveButton("Delete") { _, _ -> Bookmarks.remove(this, b); refreshBookmarks(true) }
                        .setNegativeButton("Cancel", null)
                        .show()
                }
            }
            .show()
    }

    // ---------------------------------------------------------------- actions

    /** Name the loop and save it; first ask for notification access if it's missing. */
    private fun saveLoop() {
        if (!NowPlaying.granted(this)) {
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
            return
        }
        val range = CaptureService.loopSongRange()
        val input = EditText(this).apply {
            setText(listOfNotNull(song?.title, range?.let { Bookmarks.time(it[0]) }).joinToString(" · "))
            selectAll()
        }
        AlertDialog.Builder(this)
            .setTitle("Save this loop")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                val name = input.text.toString().ifBlank { "Loop" }
                val err = try {
                    CaptureService.saveBookmark(this, name)
                } catch (e: Exception) {
                    Log.e("stemflow", "bookmark save failed", e)
                    "Not saved: ${e.message}"
                }
                Toast.makeText(this, err ?: "Saved \u201c$name\u201d", Toast.LENGTH_LONG).show()
                refreshBookmarks(force = true)
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
