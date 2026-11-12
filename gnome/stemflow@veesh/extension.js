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

const BUS_NAME = 'org.stemflow.Control';
const OBJ_PATH = '/org/stemflow/Control';
const UNIT = 'stemflow.service';
const STEP = 0.1; // hotkey / scroll step on the -1..1 line

const IFACE = `
<node>
  <interface name="org.stemflow.Control">
    <property name="Balance" type="d" access="readwrite"/>
    <property name="Bass" type="d" access="readwrite"/>
    <property name="Band" type="d" access="readwrite"/>
    <property name="Description" type="s" access="read"/>
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

        this._icon = new St.Icon({ icon_name: 'audio-x-generic-symbolic', style_class: 'system-status-icon' });
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

        // Advanced mode: each level on its own.
        this._bass = new SliderRow('Bass', v => this._set('Bass', v));
        this._band = new SliderRow('Band', v => this._set('Band', v));
        this.menu.addMenuItem(this._bass);
        this.menu.addMenuItem(this._band);

        const snaps = new PopupMenu.PopupBaseMenuItem({ activate: false, reactive: false });
        const box = new St.BoxLayout({ x_expand: true, style: 'spacing: 6px;' });
        for (const [label, name] of [['Bass solo', 'solo'], ['Full mix', 'full'], ['No bass', 'nobass']]) {
            const b = new St.Button({ label, style_class: 'button', x_expand: true, can_focus: true });
            b.connect('clicked', () => this._snap(name));
            box.add_child(b);
        }
        snaps.add_child(box);
        this.menu.addMenuItem(snaps);
        this._snaps = snaps;

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._status = new PopupMenu.PopupMenuItem('', { reactive: false });
        this._status.label.style = 'font-size: smaller;';
        this.menu.addMenuItem(this._status);

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
        const icon = Gio.ThemedIcon.new('audio-x-generic-symbolic');
        // Description arrives with the property change; read it on the next tick.
        GLib.idle_add(GLib.PRIORITY_DEFAULT, () => {
            if (this._proxy)
                Main.osdWindowManager.show(-1, icon, this._proxy.Description, (b + 1) / 2, 1);
            return GLib.SOURCE_REMOVE;
        });
    }

    _osdOff() {
        Main.osdWindowManager.show(-1, Gio.ThemedIcon.new('audio-x-generic-symbolic'), 'stemflow is off', null);
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
                    this._status.label.text =
                        `${v('App')} · ${v('Latency').toFixed(1)} s behind · ` +
                        `GPU ${v('Compute').toFixed(2)} s of ${v('Hop').toFixed(1)} s` +
                        (over > 0 ? ` · ${over} dropouts` : '');
                    this._icon.style = over > 0 ? 'color: #f6d32d;' : '';
                } catch (e) {
                    this._status.label.text = '';
                }
            });
    }

    _sync() {
        const on = this._proxy !== null && this._proxy.Description !== null;
        const adv = this._settings.get_boolean('advanced');
        this._switch.setToggleState(on);
        this._icon.opacity = on ? 255 : 110;
        for (const w of [this._desc, this._snaps, this._one, this._bass, this._band, this._status])
            w.visible = on;
        if (!on) {
            this._status.label.text = '';
            return;
        }
        this._one.visible = !adv;
        this._bass.visible = this._band.visible = adv;
        const p = this._proxy;
        this._desc.label.text = p.Description;
        this._one.set((p.Balance + 1) / 2, 'No bass');
        this._bass.set(p.Bass, `${Math.round(p.Bass * 100)}%`);
        this._band.set(p.Band, `${Math.round(p.Band * 100)}%`);
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        Gio.bus_unwatch_name(this._watch);
        this._settings.disconnect(this._advId);
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
