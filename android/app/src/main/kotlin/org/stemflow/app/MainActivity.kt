package org.stemflow.app

import android.Manifest
import android.app.Activity
import android.app.StatusBarManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.drawable.Icon
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.LinearLayout.LayoutParams
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The GNOME menu (gnome/stemflow@veesh/extension.js) as a phone screen: the on/off
 * switch, what the mix is, one balance line from bass solo through the full mix to
 * no bass, the boost, the three snaps, and a status line.
 */
class MainActivity : Activity() {
    private lateinit var power: Switch
    private lateinit var desc: TextView
    private lateinit var balance: SeekBar
    private lateinit var boost: SeekBar
    private lateinit var boostValue: TextView
    private lateinit var status: TextView
    private lateinit var karaoke: Switch
    private lateinit var modelPicker: Spinner
    private var modelIds = listOf<String>()
    private val snaps = mutableMapOf<String, Button>()
    private lateinit var noLabel: TextView
    private var dragging = false
    private val ui = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() { refresh(); ui.postDelayed(this, 300) }
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun row(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(0, dp(10), 0, dp(10))
        views.forEach { addView(it) }
    }

    private fun label(text: String, size: Float = 15f) = TextView(this).apply { this.text = text; textSize = size }

    private fun slider(max: Int, onUser: (Int) -> Unit) = SeekBar(this).apply {
        this.max = max
        layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                if (fromUser) { onUser(p); Control.save(this@MainActivity); refresh() }
            }
            override fun onStartTrackingTouch(s: SeekBar) { dragging = true }
            override fun onStopTrackingTouch(s: SeekBar) { dragging = false }
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Control.load(this)
        Control.target = Models.target(this)

        power = Switch(this).apply {
            text = "Bass isolation"
            textSize = 20f
            setOnClickListener { if (isChecked) start() else stop() }
        }
        desc = label("", 17f)
        balance = slider(200) { p ->
            var b = p / 100f - 1f
            if (abs(b) < 0.04f) b = 0f // detent at the original mix
            Control.setBalance(b)
        }
        // Boost the bass past the original, up to twice as loud; the limiter keeps
        // the result from clipping.
        boost = slider(100) { p -> Control.setLevels(1f + p / 100f * (Control.MAX_BASS - 1f), Control.band) }
        boostValue = label("100%").apply { minWidth = dp(52); gravity = Gravity.END }
        status = label("", 13f).apply { alpha = 0.7f }
        noLabel = label("  No bass")

        val snapRow = row()
        for ((text, name) in listOf("Bass solo" to "solo", "Full mix" to "full", "No bass" to "nobass")) {
            val b = Button(this).apply {
                this.text = text
                isAllCaps = false
                layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
                setOnClickListener { Control.snap(name); Control.save(this@MainActivity); refresh() }
            }
            snaps[name] = b
            snapRow.addView(b)
        }

        // As in the menu: Karaoke switches the target to vocals (the vocals model last
        // used), and Model picks among the models for the current target.
        karaoke = Switch(this).apply {
            text = "Karaoke"
            textSize = 16f
            setOnClickListener {
                Models.setKaraoke(this@MainActivity, isChecked)
                Control.target = Models.target(this@MainActivity)
                refresh()
            }
        }
        modelPicker = Spinner(this).apply {
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p: AdapterView<*>, v: View?, pos: Int, id: Long) {
                    val c = this@MainActivity
                    Models.choices(c).firstOrNull { it.id == modelIds.getOrNull(pos) }?.let {
                        if (it.id != Models.current(c)?.id) Models.select(c, it)
                    }
                }
                override fun onNothingSelected(p: AdapterView<*>) {}
            }
        }

        val extras = row(
            Button(this).apply {
                text = "Add quick tile"
                isAllCaps = false
                setOnClickListener { addTile() }
            },
            Button(this).apply {
                text = "Use screen capture instead"
                isAllCaps = false
                setOnClickListener { startProjection() }
            },
        )

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
            addView(power)
            addView(desc.apply { setPadding(0, dp(14), 0, dp(4)) })
            addView(row(label("Solo  "), balance, noLabel))
            addView(row(label("Boost "), boost, boostValue))
            addView(snapRow)
            addView(status.apply { setPadding(0, dp(14), 0, dp(14)) })
            addView(karaoke)
            addView(row(label("Model "), modelPicker))
            addView(extras)
        }
        // targetSdk 35 draws edge to edge; the outer view takes the system-bar insets
        // (which replace its own padding), the inner one keeps the margins.
        setContentView(LinearLayout(this).apply {
            fitsSystemWindows = true
            addView(content, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        })
    }

    override fun onResume() { super.onResume(); ui.post(tick) }
    override fun onPause() { super.onPause(); ui.removeCallbacks(tick) }

    private fun refresh() {
        power.isChecked = CaptureService.running
        desc.text = Control.describe()
        if (!dragging) {
            balance.progress = ((Control.balance() + 1f) * 100).roundToInt()
            boost.progress = ((Control.bass - 1f).coerceAtLeast(0f) / (Control.MAX_BASS - 1f) * 100).roundToInt()
        }
        boostValue.text = "${(maxOf(Control.bass, 1f) * 100).roundToInt()}%"
        for ((name, b) in snaps) b.alpha = if (Control.isSnap(name)) 1f else 0.6f
        status.text = if (CaptureService.running) CaptureService.status else "Off · ${Shizu.state()}"
        val t = Control.target
        val tc = t.replaceFirstChar { it.uppercase() }
        power.text = if (t == "bass") "Bass isolation" else "Karaoke (vocals)"
        snaps["solo"]?.text = "$tc solo"
        snaps["nobass"]?.text = "No $t"
        noLabel.text = "  No $t"
        val hasVocals = Models.all(this).any { it.stem == "vocals" }
        karaoke.isEnabled = hasVocals
        karaoke.isChecked = Models.karaoke(this)
        karaoke.text = if (hasVocals) "Karaoke" else "Karaoke (no vocals model on the phone)"
        val choices = Models.choices(this)
        val ids = choices.map { it.id }
        if (ids != modelIds) {
            modelIds = ids
            modelPicker.adapter = ArrayAdapter(
                this, android.R.layout.simple_spinner_dropdown_item,
                if (choices.isEmpty()) listOf("no $t model on the phone") else choices.map { it.label },
            )
        }
        val cur = ids.indexOf(Models.current(this)?.id)
        if (cur >= 0 && modelPicker.selectedItemPosition != cur) modelPicker.setSelection(cur)
    }

    private fun notificationsOk(): Boolean {
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return true
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        return false
    }

    private fun start() {
        if (!notificationsOk()) return
        val go = { startForegroundService(Intent(this, CaptureService::class.java)); Unit }
        // Shizuku is running but hasn't been asked yet: ask, then start either way
        // (without it the service waits for android/helper.sh).
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
            Icon.createWithResource(this, android.R.drawable.ic_media_play), mainExecutor,
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
