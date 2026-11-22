/* stemflow — live bass isolation control, GNOME Shell 42.
 *
 * Talks to `stemflow live` over D-Bus (org.stemflow.Control) and starts/stops it as
 * the systemd user unit stemflow.service. All the audio logic, the loudness scale and
 * the remembered levels live in stemflow; this is a remote control.
 */
const { GObject, St, Gio, GLib, Clutter, Meta, Shell } = imports.gi;
const Main = imports.ui.main;
const PanelMenu = imports.ui.panelMenu;
const PopupMenu = imports.ui.popupMenu;
const Slider = imports.ui.slider;
const ExtensionUtils = imports.misc.extensionUtils;
const Me = ExtensionUtils.getCurrentExtension();

const BUS_NAME = 'org.stemflow.Control';
const MAX_BASS = 2; // the daemon's limit: twice as loud as the original
const OBJ_PATH = '/org/stemflow/Control';
const UNIT = 'stemflow.service';
const STEP = 0.1; // hotkey / scroll step on the -1..1 line

// The clef is a -symbolic icon tinted water-blue through CSS (yellow when the GPU
// falls behind); the leaf and drops sit on top in fixed colours. The OSD gets a
// pre-coloured version.
const CLEF = 'color: #5aa9e0;';
const CLEF_WARN = 'color: #f6d32d;';
const CLEF_DIM = 'color: #5aa9e0; opacity: 0.5;'; // starting up
function gicon(name) {
    return Gio.icon_new_for_string(`${Me.path}/icons/${name}.svg`);
}
const icon = () => gicon('stemflow-light');

const IFACE = `
<node>
  <interface name="org.stemflow.Control">
    <property name="Balance" type="d" access="readwrite"/>
    <property name="Bass" type="d" access="readwrite"/>
    <property name="Band" type="d" access="readwrite"/>
    <property name="Second" type="s" access="read"/>
    <property name="SecondLevel" type="d" access="readwrite"/>
    <property name="Description" type="s" access="read"/>
    <property name="Target" type="s" access="read"/>
    <property name="Model" type="s" access="readwrite"/>
    <property name="Status" type="s" access="read"/>
    <property name="App" type="s" access="read"/>
    <property name="Latency" type="d" access="read"/>
    <property name="Compute" type="d" access="read"/>
    <property name="Hop" type="d" access="read"/>
    <property name="Overruns" type="t" access="read"/>
    <property name="Click" type="b" access="readwrite"/>
    <property name="Meter" type="u" access="readwrite"/>
    <property name="Feel" type="d" access="readwrite"/>
    <property name="Beats" type="s" access="read"/>
    <property name="LoopState" type="s" access="read"/>
    <method name="Loop"><arg type="s" direction="in"/></method>
    <method name="Nudge"><arg type="d" direction="in"/><arg type="d" direction="out"/></method>
    <method name="Snap"><arg type="s" direction="in"/><arg type="d" direction="out"/></method>
  </interface>
</node>`;
const ControlProxy = Gio.DBusProxy.makeProxyWrapper(IFACE);

function systemd(method, onError = null) {
    Gio.DBus.session.call(
        'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
        'org.freedesktop.systemd1.Manager', method,
        new GLib.Variant('(ss)', [UNIT, 'replace']), null,
        Gio.DBusCallFlags.NONE, -1, null,
        (conn, res) => {
            try {
                conn.call_finish(res);
            } catch (e) {
                if (onError)
                    onError(e.message);
                else
                    Main.notify('stemflow', `${method} failed: ${e.message}`);
            }
        });
}

/* The model stemflow will load on start: the one last picked, as the daemon
 * remembers it (~/.local/state/stemflow/model), or '' for its default. */
function lastModel() {
    try {
        const [, bytes] = GLib.file_get_contents(
            GLib.build_filenamev([GLib.get_user_state_dir(), 'stemflow', 'model']));
        return new TextDecoder().decode(bytes).trim();
    } catch (e) {
        return '';
    }
}

/* The unit's ActiveState ('active', 'activating', 'failed', 'inactive', …). */
function unitState(cb) {
    const sd = (path, iface, method, args, done) => Gio.DBus.session.call(
        'org.freedesktop.systemd1', path, iface, method, args, null,
        Gio.DBusCallFlags.NONE, -1, null, (conn, res) => {
            try {
                done(conn.call_finish(res).deep_unpack());
            } catch (e) {
                cb('inactive'); // not loaded: it isn't running
            }
        });
    sd('/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager', 'GetUnit',
        new GLib.Variant('(s)', [UNIT]), ([path]) =>
            sd(path, 'org.freedesktop.DBus.Properties', 'Get',
                new GLib.Variant('(ss)', ['org.freedesktop.systemd1.Unit', 'ActiveState']),
                ([v]) => cb(v.deep_unpack())));
}

/* The stems a model can isolate, as src/models.rs knows them: a model's id ends in
 * `_<stem>`, or `_<stem>+<stem>` for a model that separates two at once; anything
 * else is a bass model. */
const STEMS = ['bass', 'bass+vocals', 'vocals', 'other', 'piano', 'guitar', 'drums'];
const STEM_LABELS = {
    bass: 'Bass', 'bass+vocals': 'Bass + vocals', vocals: 'Vocals (karaoke)',
    other: 'Other (horns, strings…)', piano: 'Piano', guitar: 'Guitar', drums: 'Drums',
};
function stemOf(id) {
    const two = id.match(/_([a-z]+\+[a-z]+)$/);
    if (two)
        return two[1];
    return STEMS.find(s => id.endsWith(`_${s}`)) || 'bass';
}

/* A switch row that flips without closing the menu (GNOME's own closes it, which
 * is jarring for switches that only rearrange the menu). */
function stayOpen(item) {
    item.activate = () => item.toggle();
    return item;
}

/* A row of buttons sharing the width: [[label, onClick], …]. */
function buttonRow(specs) {
    const row = new PopupMenu.PopupBaseMenuItem({ activate: false, reactive: false });
    const box = new St.BoxLayout({ x_expand: true, style: 'spacing: 6px;' });
    const buttons = specs.map(([label, act]) => {
        const b = new St.Button({ label, style_class: 'button', x_expand: true, can_focus: true });
        b.connect('clicked', act);
        box.add_child(b);
        return b;
    });
    row.add_child(box);
    row.buttons = buttons;
    return row;
}

/* A labelled slider row: [label  ───○───  value]. */
const SliderRow = GObject.registerClass(
class StemflowSliderRow extends PopupMenu.PopupBaseMenuItem {
    _init(label, onChange) {
        super._init({ activate: false });
        this._label = new St.Label({ text: label, y_align: Clutter.ActorAlign.CENTER, style: 'min-width: 3.5em;' });
        this.slider = new Slider.Slider(0);
        this.slider.x_expand = true;
        this._value = new St.Label({ y_align: Clutter.ActorAlign.CENTER, style: 'min-width: 3em; text-align: right;' });
        this.add_child(this._label);
        this.add_child(this.slider);
        this.add_child(this._value);
        this._quiet = false;
        this.slider.connect('notify::value', () => {
            if (!this._quiet)
                onChange(this.slider.value);
        });
    }

    get dragging() {
        return this.slider._dragging;
    }

    set(v, text) {
        if (!this.dragging) {
            this._quiet = true;
            this.slider.value = v;
            this._quiet = false;
        }
        this._value.text = text;
    }
});

const Indicator = GObject.registerClass(
class StemflowIndicator extends PanelMenu.Button {
    _init(settings) {
        super._init(0.0, 'stemflow');
        this._settings = settings;
        this._proxy = null;
        // From "on" until stemflow shows up on the bus (it does once its model has
        // loaded, 5-10 s): {since, model}. _startError: why the last start failed.
        this._starting = null;
        this._startError = '';
        this._startTimer = 0;
        this._back = 0; // where toggle-solo / toggle-nobass return to

        this._icon = new St.Widget({ layout_manager: new Clutter.BinLayout(), y_align: Clutter.ActorAlign.CENTER });
        this._clef = new St.Icon({ gicon: gicon('stemflow-clef-symbolic'), style_class: 'system-status-icon', style: CLEF });
        this._icon.add_child(this._clef);
        this._icon.add_child(new St.Icon({ gicon: gicon('stemflow-accents'), style_class: 'system-status-icon' }));
        this.add_child(this._icon);
        this.connect('scroll-event', (_a, ev) => this._onScroll(ev));

        this._switch = stayOpen(new PopupMenu.PopupSwitchMenuItem('Bass isolation', false));
        this._switch.connect('toggled', (_i, on) => this.setRunning(on));
        this.menu.addMenuItem(this._switch);

        this._desc = new PopupMenu.PopupMenuItem('', { reactive: false });
        this.menu.addMenuItem(this._desc);

        // Simple mode: one line from bass solo through the full mix to no bass.
        this._one = new SliderRow('', pos => {
            let b = 2 * pos - 1;
            if (Math.abs(b) < 0.04)
                b = 0; // detent at the original mix
            this._set('Balance', b);
        });
        this._one._label.text = 'Solo';
        this._one._value.text = 'No bass';
        this.menu.addMenuItem(this._one);

        // Boost the target past the original, up to twice as loud (+10 dB); the
        // daemon limits the result so it can't clip.
        this._boost = new SliderRow('Boost', v => this._set('Bass', 1 + v * (MAX_BASS - 1)));
        this.menu.addMenuItem(this._boost);

        // Advanced mode: each level on its own (the target slider runs to MAX_BASS).
        this._bass = new SliderRow('Bass', v => this._set('Bass', v * MAX_BASS));
        this._band = new SliderRow('Band', v => this._set('Band', v));
        // A two-stem model (bass + vocals) always shows three plain sliders: bass,
        // that stem (labelled with its name) and the band (everything else).
        this._second = new SliderRow('', v => this._set('SecondLevel', v));
        this.menu.addMenuItem(this._bass);
        this.menu.addMenuItem(this._second);
        this.menu.addMenuItem(this._band);

        const snaps = new PopupMenu.PopupBaseMenuItem({ activate: false, reactive: false });
        const box = new St.BoxLayout({ x_expand: true, style: 'spacing: 6px;' });
        this._snapButtons = {};
        for (const [label, name] of [['Bass solo', 'solo'], ['Full mix', 'full'], ['No bass', 'nobass'], ['No vocals', 'novocals']]) {
            const b = new St.Button({ label, style_class: 'button', x_expand: true, can_focus: true });
            b.connect('clicked', () => this._snap(name));
            box.add_child(b);
            this._snapButtons[name] = b;
        }
        snaps.add_child(box);
        this.menu.addMenuItem(snaps);
        this._snaps = snaps;

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._status = new PopupMenu.PopupMenuItem('', { reactive: false });
        this._status.label.style = 'font-size: smaller;';
        this.menu.addMenuItem(this._status);

        // Beat click: a click on every beat (higher on bar 1), to judge the beat
        // tracking by ear. The submenu sets the bar length and half/double time, and
        // its title says what the tracker found.
        this._click = stayOpen(new PopupMenu.PopupSwitchMenuItem('Beat click', false));
        this._click.connect('toggled', (_i, on) => this._set('Click', on));
        this.menu.addMenuItem(this._click);
        this._beatOpts = new PopupMenu.PopupSubMenuMenuItem('Meter and feel');
        this.menu.addMenuItem(this._beatOpts);
        this._meterItems = new Map();
        for (const m of [0, 2, 3, 4, 5, 6, 7]) {
            const item = new PopupMenu.PopupMenuItem(m === 0 ? 'Auto meter' : `${m} beats per bar`);
            item.activate = () => this._set('Meter', m);
            this._beatOpts.menu.addMenuItem(item);
            this._meterItems.set(m, item);
        }
        this._beatOpts.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._feelItems = new Map();
        for (const [f, label] of [[1, 'As heard'], [0.5, 'Half time'], [2, 'Double time']]) {
            const item = new PopupMenu.PopupMenuItem(label);
            item.activate = () => this._set('Feel', f);
            this._beatOpts.menu.addMenuItem(item);
            this._feelItems.set(f, item);
        }

        // Looping (src/loopdeck.rs): loop the last bars heard; while a loop plays the
        // same buttons resize it, and the row below moves either end a beat.
        this._loopSep = new PopupMenu.PopupSeparatorMenuItem();
        this.menu.addMenuItem(this._loopSep);
        this._loopHead = new PopupMenu.PopupMenuItem('', { reactive: false });
        this.menu.addMenuItem(this._loopHead);
        this._loopBars = buttonRow([1, 2, 4, 8].map(n => [n === 1 ? '1 bar' : `${n} bars`, () => this.loop(`${n}`)])
            .concat([['Live', () => this.loop('live')]]));
        this.menu.addMenuItem(this._loopBars);
        this._loopEdit = buttonRow([['◂ Start', 'move -1 0'], ['Start ▸', 'move 1 0'], ['◂ End', 'move 0 -1'],
            ['End ▸', 'move 0 1'], ['Again', 'again']].map(([l, c]) => [l, () => this.loop(c)]));
        this.menu.addMenuItem(this._loopEdit);
        // 0.5x .. 1.5x, with a detent at 1.
        this._loopSpeed = new SliderRow('Speed', v => {
            let x = 0.5 + v;
            if (Math.abs(x - 1) < 0.03)
                x = 1;
            this.loop(`speed ${x.toFixed(3)}`);
        });
        this.menu.addMenuItem(this._loopSpeed);
        this._loopClick = stayOpen(new PopupMenu.PopupSwitchMenuItem('Click in the loop', false));
        this._loopClick.connect('toggled', (_i, on) => this.loop(`click ${on ? 'on' : 'off'}`));
        this.menu.addMenuItem(this._loopClick);

        // Stem: which part of the music the controls act on (vocals = karaoke,
        // other = the horns in a jazz combo). Picking one switches to the model last
        // used for it (or the first one); only stems with a model are listed.
        this._stems = new PopupMenu.PopupSubMenuMenuItem('Stem');
        this.menu.addMenuItem(this._stems);
        this._lastModel = {};
        this._pending = null; // the model asked for, until it is loaded (or fails)

        // Experiments: switch models live. The list comes from stemflow, which rescans
        // its models folder each time, so new exports appear without reloading anything.
        this._models = new PopupMenu.PopupSubMenuMenuItem('Model');
        this.menu.addMenuItem(this._models);
        this._exp = stayOpen(new PopupMenu.PopupSwitchMenuItem('Experiments', settings.get_boolean('experiments')));
        this._exp.connect('toggled', (_i, on) => settings.set_boolean('experiments', on));
        this.menu.addMenuItem(this._exp);
        this._expId = settings.connect('changed::experiments', () => this._sync());

        this._adv = stayOpen(new PopupMenu.PopupSwitchMenuItem('Separate sliders', settings.get_boolean('advanced')));
        this._adv.connect('toggled', (_i, on) => settings.set_boolean('advanced', on));
        this.menu.addMenuItem(this._adv);
        this._advId = settings.connect('changed::advanced', () => this._sync());

        this.menu.connect('open-state-changed', (_m, open) => {
            if (open) {
                this._refreshStats();
                this._timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 2, () => {
                    this._refreshStats();
                    return GLib.SOURCE_CONTINUE;
                });
            } else if (this._timer) {
                GLib.source_remove(this._timer);
                this._timer = 0;
            }
        });

        this._watch = Gio.bus_watch_name(Gio.BusType.SESSION, BUS_NAME, Gio.BusNameWatcherFlags.NONE,
            () => this._appeared(), () => this._vanished());
        this._sync();
    }

    _appeared() {
        new ControlProxy(Gio.DBus.session, BUS_NAME, OBJ_PATH, (proxy, err) => {
            if (err) {
                log(`stemflow: ${err}`);
                return;
            }
            this._proxy = proxy;
            this._stopStarting('');
            this._propsId = proxy.connect('g-properties-changed', () => this._sync());
            this._sync();
        });
    }

    _vanished() {
        if (this._proxy && this._propsId)
            this._proxy.disconnect(this._propsId);
        this._proxy = null;
        this._sync();
    }

    setRunning(on) {
        if (!on) {
            this._stopStarting('');
            systemd('StopUnit');
            return;
        }
        if (this._proxy || this._starting)
            return;
        this._startError = '';
        this._starting = { since: GLib.get_monotonic_time(), model: lastModel() };
        systemd('StartUnit', e => this._stopStarting(`Couldn't start: ${e}`));
        // Tick the elapsed time, and notice the service dying before it got going.
        this._startTimer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 1, () => {
            unitState(state => {
                const secs = (GLib.get_monotonic_time() - (this._starting?.since ?? 0)) / 1e6;
                if (this._starting && secs > 2 && (state === 'failed' || state === 'inactive'))
                    this._stopStarting("stemflow stopped while starting (journalctl --user -u stemflow)");
            });
            this._sync();
            return GLib.SOURCE_CONTINUE;
        });
        this._sync();
    }

    _stopStarting(err) {
        if (this._startTimer)
            GLib.source_remove(this._startTimer);
        this._startTimer = 0;
        this._starting = null;
        this._startError = err;
        this._sync();
    }

    _set(prop, v) {
        if (this._proxy)
            this._proxy[prop] = v;
    }

    _snap(name) {
        if (!this._proxy)
            return;
        this._proxy.SnapRemote(name, () => this._osd());
    }

    nudge(step) {
        if (!this._proxy)
            return this._osdOff();
        this._proxy.NudgeRemote(step, () => this._osd());
    }

    // Jump to a snap point, or back to where you were if already there.
    toggle(name, at) {
        if (!this._proxy)
            return this._osdOff();
        const b = this._proxy.Balance;
        if (Math.abs(b - at) < 1e-3) {
            this._set('Balance', this._back);
            this._osd(this._back);
        } else {
            this._back = b;
            this._snap(name);
        }
    }

    /* The loop's state (JSON from the daemon), or null: off, no beat model, or an
     * older daemon. */
    _loopState() {
        const j = this._proxy?.LoopState;
        if (!j)
            return null;
        try {
            return JSON.parse(j);
        } catch (e) {
            return null;
        }
    }

    loop(cmd) {
        if (!this._proxy)
            return this._osdOff();
        this._proxy.LoopRemote(cmd, (_r, err) => {
            if (err)
                Main.notify('stemflow', err.message.replace(/^.*InvalidArgs: /, ''));
        });
    }

    /* Super+Alt+L: loop the last 4 bars, or back to live if a loop plays (or is
     * being found). */
    toggleLoop() {
        const l = this._loopState();
        if (!l)
            return this._proxy ? Main.notify('stemflow', 'No looping: no beat model') : this._osdOff();
        const live = l.playing || l.finding;
        this.loop(live ? 'live' : '4');
        Main.osdWindowManager.show(-1, icon(), live ? 'Live' : 'Looping the last 4 bars', null);
    }

    _syncLoop(on) {
        const l = on ? this._loopState() : null;
        for (const w of [this._loopSep, this._loopHead, this._loopBars, this._loopEdit, this._loopSpeed, this._loopClick])
            w.visible = !!l;
        if (!l)
            return;
        this._loopEdit.visible = l.playing;
        this._loopBars.buttons[4].reactive = l.playing || l.tail || l.finding;
        let head;
        if (l.playing) {
            head = `Looping ${l.bars} bar${l.bars === 1 ? '' : 's'} · ${Math.round(l.bpm)} bpm · ` +
                `${l.meter}/bar · ${l.length.toFixed(1)} s` + (l.speed !== 1 ? ` · ${l.speed.toFixed(2)}×` : '');
        } else if (l.finding) {
            head = 'Finding the bars…';
        } else if (l.tail) {
            head = 'Catching up with the song';
        } else {
            head = 'Loop the last…';
        }
        if (l.status && !l.finding)
            head = l.status;
        this._loopHead.label.text = head;
        this._loopSpeed.set(l.speed - 0.5, `${l.speed.toFixed(2)}×`);
        this._loopClick.setToggleState(!!l.click);
    }

    _onScroll(ev) {
        const dir = ev.get_scroll_direction();
        if (dir === Clutter.ScrollDirection.UP || dir === Clutter.ScrollDirection.RIGHT)
            this.nudge(STEP / 2);
        else if (dir === Clutter.ScrollDirection.DOWN || dir === Clutter.ScrollDirection.LEFT)
            this.nudge(-STEP / 2);
        return Clutter.EVENT_STOP;
    }

    _osd(balance) {
        if (!this._proxy)
            return;
        const b = balance ?? this._proxy.Balance;
        // Description arrives with the property change; read it on the next tick.
        GLib.idle_add(GLib.PRIORITY_DEFAULT, () => {
            if (this._proxy)
                Main.osdWindowManager.show(-1, icon(), this._proxy.Description, (b + 1) / 2, 1);
            return GLib.SOURCE_REMOVE;
        });
    }

    _osdOff() {
        Main.osdWindowManager.show(-1, icon(), 'stemflow is off', null);
    }

    _refreshStats() {
        if (!this._proxy)
            return;
        // Compute/Overruns change every hop, so they're read on demand, not signalled.
        Gio.DBus.session.call(BUS_NAME, OBJ_PATH, 'org.freedesktop.DBus.Properties', 'GetAll',
            new GLib.Variant('(s)', ['org.stemflow.Control']), null, Gio.DBusCallFlags.NONE, -1, null,
            (conn, res) => {
                try {
                    const [props] = conn.call_finish(res).deep_unpack();
                    const v = k => props[k].deep_unpack();
                    const over = v('Overruns');
                    if (props.Models)
                        this._fillModels(v('Models'), v('Model'));
                    this._status.label.text =
                        `${v('App')} · ${v('Latency').toFixed(1)} s behind · ` +
                        `GPU ${v('Compute').toFixed(2)} s of ${v('Hop').toFixed(1)} s` +
                        (over > 0 ? ` · ${over} dropouts` : '');
                    this._clef.style = over > 0 ? CLEF_WARN : CLEF;
                } catch (e) {
                    this._status.label.text = '';
                }
            });
    }

    _setMode(target) {
        const list = this._modelList || [];
        const known = id => list.some(([m]) => m === id);
        const id = known(this._lastModel[target]) ? this._lastModel[target]
            : (list.find(([m]) => stemOf(m) === target) || [])[0];
        if (id)
            this._setModel(id);
        else
            Main.notify('stemflow', `No ${target} model in the models folder`);
    }

    _loading() {
        return !!this._proxy && (this._proxy.Status || '').startsWith('Loading');
    }

    _fillModels(models, current) {
        this._modelList = models;
        if (current)
            this._lastModel[stemOf(current)] = current;
        // While a model loads, the pickers already show the one asked for.
        const loading = this._loading();
        const shown = loading && this._pending ? this._pending : current;
        const key = JSON.stringify([models, shown, loading]);
        if (key === this._modelsKey)
            return; // rebuilding while a submenu is open would close it
        this._modelsKey = key;
        this._shownModel = current;
        const stem = stemOf(shown || '');
        const tail = loading ? ' · loading…' : '';
        const dot = on => (on ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE);

        this._stems.menu.removeAll();
        for (const s of STEMS.filter(t => models.some(([id]) => stemOf(id) === t))) {
            const item = new PopupMenu.PopupMenuItem(STEM_LABELS[s]);
            item.setOrnament(dot(s === stem));
            item.connect('activate', () => {
                if (s !== stem)
                    this._setMode(s);
            });
            this._stems.menu.addMenuItem(item);
        }
        this._stems.label.text = `Stem: ${STEM_LABELS[stem]}${tail}`;

        this._models.menu.removeAll();
        for (const [id, label] of models.filter(([id]) => stemOf(id) === stem)) {
            const item = new PopupMenu.PopupMenuItem(label);
            item.setOrnament(dot(id === shown));
            item.connect('activate', () => this._setModel(id));
            this._models.menu.addMenuItem(item);
        }
        const cur = models.find(([id]) => id === shown);
        this._models.label.text = `Model: ${cur ? cur[1] : shown}${tail}`;
    }

    _setModel(id) {
        this._pending = id;
        Gio.DBus.session.call(BUS_NAME, OBJ_PATH, 'org.freedesktop.DBus.Properties', 'Set',
            new GLib.Variant('(ssv)', ['org.stemflow.Control', 'Model', new GLib.Variant('s', id)]),
            null, Gio.DBusCallFlags.NONE, -1, null, (conn, res) => {
                try {
                    conn.call_finish(res);
                } catch (e) {
                    this._pending = null;
                    Main.notify('stemflow', e.message);
                }
            });
    }

    _sync() {
        const on = this._proxy !== null && this._proxy.Description !== null;
        const adv = this._settings.get_boolean('advanced');
        const starting = !on && this._starting;
        this._switch.setToggleState(on || !!starting);
        // Off: the clef takes the bar's own colour (white); on: water-blue.
        if (!on || this._clef.style !== CLEF_WARN)
            this._clef.style = on ? CLEF : '';
        for (const w of [this._desc, this._snaps, this._one, this._boost, this._bass, this._band, this._second, this._status, this._models, this._stems])
            w.visible = on;
        // Older daemons have no beat click.
        const beats = on && this._proxy.Click !== null;
        this._click.visible = this._beatOpts.visible = beats;
        if (beats) {
            const p = this._proxy;
            this._click.setToggleState(!!p.Click);
            this._click.label.text = p.Click && p.Beats ? `Beat click · ${p.Beats}` : 'Beat click';
            const dot = v => (v ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE);
            for (const [m, item] of this._meterItems)
                item.setOrnament(dot(m === p.Meter));
            for (const [f, item] of this._feelItems)
                item.setOrnament(dot(f === p.Feel));
            const meter = p.Meter ? `${p.Meter}/bar` : 'auto meter';
            const feel = p.Feel === 0.5 ? ' · half time' : p.Feel === 2 ? ' · double time' : '';
            this._beatOpts.label.text = `Meter and feel: ${meter}${feel}`;
        }
        // An older daemon has no Model property: no pickers.
        this._stems.visible = on && !!this._proxy.Model;
        this._models.visible = on && this._settings.get_boolean('experiments') && !!this._proxy.Model;
        this._syncLoop(on);
        if (!on) {
            this._status.label.text = '';
            // Starting, or why it didn't: one line under the switch.
            this._desc.visible = !!starting || !!this._startError;
            if (starting) {
                const secs = Math.floor((GLib.get_monotonic_time() - starting.since) / 1e6);
                const what = starting.model ? ` ${starting.model}` : ' the model';
                this._desc.label.text = `Loading${what}… ${secs} s`;
            } else {
                this._desc.label.text = this._startError;
            }
            this._clef.style = starting ? CLEF_DIM : '';
            return;
        }
        // Older daemons have no Second.
        const second = this._proxy.Second || '';
        const three = !!second;
        this._one.visible = this._boost.visible = !adv && !three;
        this._bass.visible = this._band.visible = adv || three;
        this._second.visible = three;
        this._snapButtons.novocals.visible = three;
        const p = this._proxy;
        // "Bass" is whichever stem stemflow targets (the Stem picker, or
        // `stemflow live --target vocals`). Older daemons have no Target.
        const t = p.Target || 'bass';
        const T = t[0].toUpperCase() + t.slice(1);
        this._switch.label.text = three ? `${T} + ${second} isolation`
            : `${T} isolation${t === 'vocals' ? ' (karaoke)' : ''}`;
        this._bass._label.text = T;
        this._snapButtons.solo.label = `${T} solo`;
        this._snapButtons.nobass.label = `No ${t}`;
        // "Loading …" or a load error, while there is one.
        this._desc.label.text = p.Status || p.Description;
        // Loading started or ended, or the model switched: the pickers follow.
        const loading = this._loading();
        if (this._wasLoading && !loading)
            this._pending = null;
        this._wasLoading = loading;
        if (p.Model && this._modelList)
            this._fillModels(this._modelList, p.Model);
        this._one.set((p.Balance + 1) / 2, `No ${t}`);
        this._boost.set(Math.max(0, p.Bass - 1) / (MAX_BASS - 1),
            p.Bass > 1 ? `+${Math.round((p.Bass - 1) * 100)}%` : 'off');
        this._bass.set(p.Bass / MAX_BASS, `${Math.round(p.Bass * 100)}%`);
        this._band.set(p.Band, `${Math.round(p.Band * 100)}%`);
        if (three) {
            this._second._label.text = second[0].toUpperCase() + second.slice(1);
            const lv = p.SecondLevel;
            this._second.set(lv, `${Math.round(lv * 100)}%`);
            this._snapButtons.novocals.label = `No ${second}`;
        }
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        if (this._startTimer)
            GLib.source_remove(this._startTimer);
        Gio.bus_unwatch_name(this._watch);
        this._settings.disconnect(this._advId);
        this._settings.disconnect(this._expId);
        this._vanished();
        super.destroy();
    }
});

let indicator = null;
let settings = null;
const KEYS = ['slide-left', 'slide-right', 'toggle-solo', 'toggle-nobass', 'toggle-running', 'toggle-loop'];

function init() {
}

function enable() {
    settings = ExtensionUtils.getSettings();
    indicator = new Indicator(settings);
    Main.panel.addToStatusArea('stemflow', indicator);
    const actions = {
        'slide-left': () => indicator.nudge(-STEP),
        'slide-right': () => indicator.nudge(STEP),
        'toggle-solo': () => indicator.toggle('solo', -1),
        'toggle-nobass': () => indicator.toggle('nobass', 1),
        'toggle-running': () => indicator.setRunning(!indicator._proxy && !indicator._starting),
        'toggle-loop': () => indicator.toggleLoop(),
    };
    for (const k of KEYS)
        Main.wm.addKeybinding(k, settings, Meta.KeyBindingFlags.NONE,
            Shell.ActionMode.NORMAL | Shell.ActionMode.OVERVIEW, actions[k]);
}

function disable() {
    for (const k of KEYS)
        Main.wm.removeKeybinding(k);
    indicator.destroy();
    indicator = null;
    settings = null;
}
