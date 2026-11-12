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
    <property name="Description" type="s" access="read"/>
    <property name="Target" type="s" access="read"/>
    <property name="Model" type="s" access="readwrite"/>
    <property name="Status" type="s" access="read"/>
    <property name="App" type="s" access="read"/>
    <property name="Latency" type="d" access="read"/>
    <property name="Compute" type="d" access="read"/>
    <property name="Hop" type="d" access="read"/>
    <property name="Overruns" type="t" access="read"/>
    <method name="Nudge"><arg type="d" direction="in"/><arg type="d" direction="out"/></method>
    <method name="Snap"><arg type="s" direction="in"/><arg type="d" direction="out"/></method>
  </interface>
</node>`;
const ControlProxy = Gio.DBusProxy.makeProxyWrapper(IFACE);

function systemd(method) {
    Gio.DBus.session.call(
        'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
        'org.freedesktop.systemd1.Manager', method,
        new GLib.Variant('(ss)', [UNIT, 'replace']), null,
        Gio.DBusCallFlags.NONE, -1, null,
        (conn, res) => {
            try {
                conn.call_finish(res);
            } catch (e) {
                Main.notify('stemflow', `${method} failed: ${e.message}`);
            }
        });
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
        this._back = 0; // where toggle-solo / toggle-nobass return to

        this._icon = new St.Widget({ layout_manager: new Clutter.BinLayout(), y_align: Clutter.ActorAlign.CENTER });
        this._clef = new St.Icon({ gicon: gicon('stemflow-clef-symbolic'), style_class: 'system-status-icon', style: CLEF });
        this._icon.add_child(this._clef);
        this._icon.add_child(new St.Icon({ gicon: gicon('stemflow-accents'), style_class: 'system-status-icon' }));
        this.add_child(this._icon);
        this.connect('scroll-event', (_a, ev) => this._onScroll(ev));

        this._switch = new PopupMenu.PopupSwitchMenuItem('Bass isolation', false);
        this._switch.connect('toggled', (_i, on) => systemd(on ? 'StartUnit' : 'StopUnit'));
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
        this.menu.addMenuItem(this._bass);
        this.menu.addMenuItem(this._band);

        const snaps = new PopupMenu.PopupBaseMenuItem({ activate: false, reactive: false });
        const box = new St.BoxLayout({ x_expand: true, style: 'spacing: 6px;' });
        this._snapButtons = {};
        for (const [label, name] of [['Bass solo', 'solo'], ['Full mix', 'full'], ['No bass', 'nobass']]) {
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

        // Karaoke: the same controls, acting on the vocals. Switches to the vocals
        // model last used (or the first one), and back to the last bass model.
        this._karaoke = new PopupMenu.PopupSwitchMenuItem('Karaoke', false);
        this._karaoke.connect('toggled', (_i, on) => this._setMode(on ? 'vocals' : 'bass'));
        this.menu.addMenuItem(this._karaoke);
        this._lastModel = {};

        // Experiments: switch models live. The list comes from stemflow, which rescans
        // its models folder each time, so new exports appear without reloading anything.
        this._models = new PopupMenu.PopupSubMenuMenuItem('Model');
        this.menu.addMenuItem(this._models);
        this._exp = new PopupMenu.PopupSwitchMenuItem('Experiments', settings.get_boolean('experiments'));
        this._exp.connect('toggled', (_i, on) => settings.set_boolean('experiments', on));
        this.menu.addMenuItem(this._exp);
        this._expId = settings.connect('changed::experiments', () => this._sync());

        this._adv = new PopupMenu.PopupSwitchMenuItem('Separate sliders', settings.get_boolean('advanced'));
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
        const of = id => (id.endsWith('_vocals') ? 'vocals' : 'bass');
        const known = id => list.some(([m]) => m === id);
        const id = known(this._lastModel[target]) ? this._lastModel[target]
            : (list.find(([m]) => of(m) === target) || [])[0];
        if (id)
            this._setModel(id);
        else
            Main.notify('stemflow', `No ${target} model in the models folder`);
    }

    _fillModels(models, current) {
        this._modelList = models;
        if (current)
            this._lastModel[current.endsWith('_vocals') ? 'vocals' : 'bass'] = current;
        const key = JSON.stringify([models, current]);
        if (key === this._modelsKey)
            return; // rebuilding while the submenu is open would close it
        this._modelsKey = key;
        this._shownModel = current;
        this._models.menu.removeAll();
        for (const [id, label] of models) {
            const item = new PopupMenu.PopupMenuItem(label);
            item.setOrnament(id === current ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE);
            item.connect('activate', () => this._setModel(id));
            this._models.menu.addMenuItem(item);
        }
        const cur = models.find(([id]) => id === current);
        this._models.label.text = `Model: ${cur ? cur[1] : current}`;
    }

    _setModel(id) {
        Gio.DBus.session.call(BUS_NAME, OBJ_PATH, 'org.freedesktop.DBus.Properties', 'Set',
            new GLib.Variant('(ssv)', ['org.stemflow.Control', 'Model', new GLib.Variant('s', id)]),
            null, Gio.DBusCallFlags.NONE, -1, null, (conn, res) => {
                try {
                    conn.call_finish(res);
                } catch (e) {
                    Main.notify('stemflow', e.message);
                }
            });
    }

    _sync() {
        const on = this._proxy !== null && this._proxy.Description !== null;
        const adv = this._settings.get_boolean('advanced');
        this._switch.setToggleState(on);
        // Off: the clef takes the bar's own colour (white); on: water-blue.
        if (!on || this._clef.style !== CLEF_WARN)
            this._clef.style = on ? CLEF : '';
        for (const w of [this._desc, this._snaps, this._one, this._boost, this._bass, this._band, this._status, this._models, this._karaoke])
            w.visible = on;
        this._karaoke.visible = on && !!this._proxy.Model;
        // An older daemon has no Model property: no switcher.
        this._models.visible = on && this._settings.get_boolean('experiments') && !!this._proxy.Model;
        if (!on) {
            this._status.label.text = '';
            return;
        }
        this._one.visible = this._boost.visible = !adv;
        this._bass.visible = this._band.visible = adv;
        const p = this._proxy;
        // "Bass" is whichever stem stemflow targets: vocals in karaoke mode
        // (`stemflow live --target vocals`). Older daemons have no Target.
        const t = p.Target || 'bass';
        const T = t[0].toUpperCase() + t.slice(1);
        this._switch.label.text = t === 'bass' ? 'Bass isolation' : `${T} isolation (karaoke)`;
        this._karaoke.setToggleState(t === 'vocals');
        this._bass._label.text = T;
        this._snapButtons.solo.label = `${T} solo`;
        this._snapButtons.nobass.label = `No ${t}`;
        // "Loading …" or a load error, while there is one.
        this._desc.label.text = p.Status || p.Description;
        if (p.Model && this._modelsKey && p.Model !== this._shownModel)
            this._refreshStats(); // switched: move the dot
        this._one.set((p.Balance + 1) / 2, `No ${t}`);
        this._boost.set(Math.max(0, p.Bass - 1) / (MAX_BASS - 1),
            p.Bass > 1 ? `+${Math.round((p.Bass - 1) * 100)}%` : 'off');
        this._bass.set(p.Bass / MAX_BASS, `${Math.round(p.Bass * 100)}%`);
        this._band.set(p.Band, `${Math.round(p.Band * 100)}%`);
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        Gio.bus_unwatch_name(this._watch);
        this._settings.disconnect(this._advId);
        this._settings.disconnect(this._expId);
        this._vanished();
        super.destroy();
    }
});

let indicator = null;
let settings = null;
const KEYS = ['slide-left', 'slide-right', 'toggle-solo', 'toggle-nobass', 'toggle-running'];

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
        'toggle-running': () => systemd(indicator._proxy ? 'StopUnit' : 'StartUnit'),
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
