/* The Loop section of the menu: the loop panel stemflow declares (D-Bus
 * `LoopPanel()`, src/loopdeck/panel.rs), drawn; and the loop view, drawn from
 * `LoopView(bins)`. Its controls send `Loop(cmd)`. */
const { GObject, St, GLib, Clutter } = imports.gi;
const Me = imports.misc.extensionUtils.getCurrentExtension();
const PopupMenu = imports.ui.popupMenu;
const Slider = imports.ui.slider;
const ModalDialog = imports.ui.modalDialog;

/* Ask for a name: `done(name)` unless cancelled. */
function askName(title, value, done) {
    const d = new ModalDialog.ModalDialog();
    d.contentLayout.add_child(new St.Label({ text: title, style: 'font-weight: bold; padding-bottom: 8px;' }));
    const entry = new St.Entry({ text: value, can_focus: true, style: 'min-width: 280px;' });
    d.contentLayout.add_child(entry);
    const ok = () => {
        const name = entry.get_text().trim();
        d.close();
        if (name)
            done(name);
    };
    entry.clutter_text.connect('activate', ok);
    d.setButtons([
        { label: 'Cancel', action: () => d.close(), key: Clutter.KEY_Escape },
        { label: 'Save', action: ok, default: true },
    ]);
    d.open();
    entry.grab_key_focus();
    entry.clutter_text.set_selection(0, value.length);
}

const GREEN = [0x79 / 255, 0xc2 / 255, 0x67 / 255];
const WAVE = [159 / 255, 182 / 255, 198 / 255];

/* A non-reactive menu row holding `child` (reactive children still get events). */
function holder(child) {
    const item = new PopupMenu.PopupBaseMenuItem({ activate: false, reactive: false, can_focus: false });
    item.add_child(child);
    return item;
}

function btn(label, act, toggle = false) {
    const b = new St.Button({ label, style_class: 'button', x_expand: true, can_focus: true, toggle_mode: toggle });
    b.connect('clicked', () => act(b));
    return b;
}

function box(children, style = 'spacing: 6px;', vertical = false) {
    const b = new St.BoxLayout({ x_expand: true, vertical, style });
    for (const c of children)
        b.add_child(c);
    return b;
}

function small(text, style = 'font-size: 9pt; color: #9aa;') {
    return new St.Label({ text, style, x_align: Clutter.ActorAlign.CENTER, x_expand: true });
}

/* The waveform of the loop with a bar either side: a tick on every beat, numbered
 * bar lines, the ends and the playhead. Drag an end to move it (it snaps to the
 * beats); right-click or long-press a beat to make it bar 1. */
var LoopView = GObject.registerClass(
class StemflowLoopView extends St.DrawingArea {
    _init(onEdges, onBarOne) {
        super._init({ reactive: true, x_expand: true, height: 96, style: 'min-height: 96px;' });
        this.pic = null;
        this._drag = 0; // 0 none, 1 start, 2 end
        this._dragBeat = -1;
        this._press = 0;
        this.connect('repaint', a => this._draw(a));
        this.connect('button-press-event', (_a, ev) => {
            const x = this._local(ev);
            if (ev.get_button() === 3) {
                this._barOne(x, onBarOne);
                return Clutter.EVENT_STOP;
            }
            const p = this.pic;
            if (!p)
                return Clutter.EVENT_PROPAGATE;
            const [s, e] = [this._x(p.start), this._x(p.end)];
            this._drag = Math.abs(x - s) < 18 && Math.abs(x - s) <= Math.abs(x - e) ? 1 : Math.abs(x - e) < 18 ? 2 : 0;
            this._dragBeat = -1;
            // long-press (not on an end): bar 1
            if (this._drag === 0) {
                this._press = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 600, () => {
                    this._press = 0;
                    this._barOne(x, onBarOne);
                    return GLib.SOURCE_REMOVE;
                });
            }
            return Clutter.EVENT_STOP;
        });
        this.connect('motion-event', (_a, ev) => {
            if (this._drag) {
                this._dragBeat = this._nearest(this._local(ev));
                this.queue_repaint();
            }
            return Clutter.EVENT_STOP;
        });
        this.connect('button-release-event', () => {
            this._cancelPress();
            const p = this.pic;
            if (this._drag && this._dragBeat >= 0 && p) {
                if (this._drag === 1 && this._dragBeat < p.b && this._dragBeat !== p.a)
                    onEdges(this._dragBeat, p.b);
                if (this._drag === 2 && this._dragBeat > p.a && this._dragBeat !== p.b)
                    onEdges(p.a, this._dragBeat);
            }
            this._drag = 0;
            this._dragBeat = -1;
            this.queue_repaint();
            return Clutter.EVENT_STOP;
        });
        this.connect('leave-event', () => this._cancelPress());
        this.connect('destroy', () => this._cancelPress());
    }

    _cancelPress() {
        if (this._press)
            GLib.source_remove(this._press);
        this._press = 0;
    }

    _barOne(x, onBarOne) {
        const i = this._nearest(x);
        if (i >= 0)
            onBarOne(i);
    }

    _local(ev) {
        const [sx, sy] = ev.get_coords();
        const [, x] = this.transform_stage_point(sx, sy);
        return x;
    }

    _x(f) {
        return f / (this.pic?.to || 1) * this.width;
    }

    _nearest(px) {
        const p = this.pic;
        if (!p || !p.beats.length)
            return -1;
        let best = p.beats[0];
        for (const b of p.beats) {
            if (Math.abs(this._x(b[0]) - px) < Math.abs(this._x(best[0]) - px))
                best = b;
        }
        return best[2];
    }

    set(pic) {
        this.pic = pic;
        this.queue_repaint();
    }

    _draw(area) {
        const cr = area.get_context();
        const [w, h] = area.get_surface_size();
        cr.setSourceRGBA(0, 0, 0, 0.25);
        cr.rectangle(0, 0, w, h);
        cr.fill();
        const p = this.pic;
        if (!p) {
            cr.$dispose();
            return;
        }
        const x = f => f / (p.to || 1) * w;
        const mid = h * 0.55;
        // waveform
        cr.setSourceRGBA(...WAVE, 0.6);
        const bw = w / Math.max(p.peaks.length, 1);
        p.peaks.forEach((v, k) => {
            const a = v * h * 0.4;
            cr.rectangle(k * bw, mid - a, Math.max(bw - 0.5, 0.5), 2 * a);
        });
        cr.fill();
        // the loop, with the end being dragged where it would land
        let [s, e] = [x(p.start), x(p.end)];
        const d = this._drag && p.beats.find(b => b[2] === this._dragBeat);
        if (d) {
            if (this._drag === 1)
                s = x(d[0]);
            else
                e = x(d[0]);
        }
        cr.setSourceRGBA(0.5, 0.5, 0.5, 0.3);
        cr.rectangle(0, 0, s, h);
        cr.rectangle(e, 0, w - e, h);
        cr.fill();
        cr.setSourceRGBA(108 / 255, 180 / 255, 232 / 255, 0.18);
        cr.rectangle(s, 0, e - s, h);
        cr.fill();
        // beats and bars, bars numbered from the loop's first bar line
        cr.setFontSize(10);
        let barNo = 0;
        for (const [f, bar, i] of p.beats) {
            const px = x(f);
            if (bar) {
                cr.setSourceRGB(...GREEN);
                cr.setLineWidth(2);
                cr.moveTo(px, 12);
                cr.lineTo(px, h);
                cr.stroke();
                if (i >= p.a)
                    barNo++;
                if (i >= p.a && i < p.b) {
                    cr.moveTo(px + 3, 10);
                    cr.showText(`${barNo}`);
                }
            } else {
                cr.setSourceRGBA(...WAVE, 0.7);
                cr.setLineWidth(1);
                cr.moveTo(px, h * 0.25);
                cr.lineTo(px, h);
                cr.stroke();
            }
        }
        // the ends, with a grip, and the playhead
        cr.setSourceRGB(0xea / 255, 0xf2 / 255, 0xf7 / 255);
        for (const px of [s, e]) {
            cr.setLineWidth(2.5);
            cr.moveTo(px, 0);
            cr.lineTo(px, h);
            cr.stroke();
            cr.arc(px, h - 8, 5, 0, 2 * Math.PI);
            cr.fill();
        }
        if (p.playing) {
            cr.setSourceRGB(229 / 255, 57 / 255, 53 / 255);
            cr.setLineWidth(1.5);
            cr.moveTo(x(p.playhead), 0);
            cr.lineTo(x(p.playhead), h);
            cr.stroke();
        }
        cr.$dispose();
    }
});

/* A labelled slider: [label  ───○───  value]. */
function sliderRow(label, onChange) {
    const s = new Slider.Slider(0);
    s.x_expand = true;
    const value = new St.Label({ y_align: Clutter.ActorAlign.CENTER, style: 'min-width: 3em; text-align: right;' });
    const name = new St.Label({ text: label, y_align: Clutter.ActorAlign.CENTER, style: 'min-width: 3.5em;' });
    const row = box([name, s, value], 'spacing: 8px;');
    row.label = name;
    let quiet = false;
    s.connect('notify::value', () => {
        if (!quiet)
            onChange(s.value);
    });
    row.set = (v, text) => {
        if (!s._dragging) {
            quiet = true;
            s.value = v;
            quiet = false;
        }
        value.text = text;
    };
    return row;
}

/* The loop panel as stemflow declares it (`LoopPanel()`, src/loopdeck/panel.rs):
 * rows of items, each of a kind in KINDS (panelKinds.js, generated from the Rust).
 * Each kind is drawn here once, in `widget`: the Rust tests check there's a case
 * for every kind. The texts, states and the lines the controls send all come from
 * stemflow. */
const KINDS = Me.imports.panelKinds.KINDS;

const lines = (j, k) => j[k] || [];

/* One item of the panel: `actor` for a row of controls, or `item` for one that is a
 * menu item itself; `update(j)` shows its JSON. */
function widget(kind, sec) {
    if (!KINDS.includes(kind))
        log(`stemflow: no such panel item: ${kind}`);
    switch (kind) {
    case 'text': {
        const item = new PopupMenu.PopupMenuItem('', { reactive: false });
        const actor = small('', 'font-size: 9pt;');
        const style = { title: 'font-weight: bold;', small: 'font-size: smaller;', normal: '' };
        return {
            item, actor, update: j => {
                item.label.text = actor.text = j.text;
                item.label.style = actor.style = style[j.style] ?? '';
            },
        };
    }
    case 'button': {
        let j = {};
        const b = btn('', () => sec.run(lines(j, 'send'), j.ask));
        return {
            actor: b, update: x => {
                j = x;
                b.label = j.label;
                b.reactive = j.enabled !== false;
                b.toggle_mode = j.on !== null && j.on !== undefined;
                if (b.toggle_mode)
                    b.set_checked(!!j.on);
            },
        };
    }
    case 'switch': {
        let j = {};
        const item = new PopupMenu.PopupSwitchMenuItem('', false);
        item.activate = () => sec.run(lines(j, 'send'));
        return {
            item, update: x => {
                j = x;
                item.label.text = j.label;
                item.setToggleState(!!j.on);
            },
        };
    }
    case 'choice': {
        let opts = [];
        const row = box([], 'spacing: 4px;');
        let buttons = [];
        return {
            actor: row, update: j => {
                opts = j.options || [];
                if (buttons.length !== opts.length) {
                    row.destroy_all_children();
                    buttons = opts.map((_o, i) => {
                        const b = btn('', () => sec.run(lines(opts[i], 'send')), true);
                        row.add_child(b);
                        return b;
                    });
                }
                opts.forEach((o, i) => {
                    buttons[i].label = o.label;
                    buttons[i].set_checked(i === j.selected);
                });
            },
        };
    }
    case 'nudge': {
        let j = {};
        const label = small('', 'font-size: 8pt; font-weight: bold;');
        const note = small('');
        const actor = box([
            label,
            box([btn('◀', () => sec.run(lines(j, 'back'))), btn('▶', () => sec.run(lines(j, 'forward')))], 'spacing: 4px;'),
            note,
        ], 'spacing: 2px;', true);
        return {
            actor, update: x => {
                j = x;
                label.text = j.label;
                note.text = j.note;
            },
        };
    }
    case 'slider': {
        let j = {};
        const row = sliderRow('', pos => {
            const step = Math.max(j.step || 1, 1e-6);
            let v = Math.round((j.min + pos * (j.max - j.min)) / step) * step;
            if (j.snap !== null && j.snap !== undefined && Math.abs(v - j.snap) < 2 * step)
                v = j.snap;
            sec.run([j.send.replace('{v}', step >= 1 ? `${Math.round(v)}` : v.toFixed(2))]);
        });
        return {
            actor: row, update: x => {
                j = x;
                row.label.text = j.label;
                const span = j.max - j.min;
                row.set(span > 0 ? (j.value - j.min) / span : 0, j.text);
            },
        };
    }
    case 'loop_view': {
        let j = {};
        sec.view.onEdges = (a, b) => sec.run([j.edges.replace('{a}', `${a}`).replace('{b}', `${b}`)]);
        sec.view.onBarOne = i => sec.run([j.bar1.replace('{beat}', `${i}`)]);
        return { actor: sec.view, update: x => (j = x) };
    }
    case 'bookmarks': {
        let j = null;
        const item = new PopupMenu.PopupSubMenuMenuItem('Bookmarks');
        const search = new St.Entry({ can_focus: true, x_expand: true });
        item.menu.addMenuItem(holder(search));
        let rows = [];
        const fill = () => {
            for (const r of rows)
                r.destroy();
            rows = [];
            const add = r => {
                item.menu.addMenuItem(r);
                rows.push(r);
            };
            item.label.text = j.title;
            search.hint_text = j.search_hint;
            const q = search.get_text().toLowerCase();
            let any = false;
            for (const g of j.groups) {
                const marks = g.marks.filter(m => !q || m.search.includes(q));
                if (!marks.length)
                    continue;
                any = true;
                const h = new PopupMenu.PopupMenuItem(g.title, { reactive: false });
                h.label.style = 'font-weight: bold; font-size: smaller;';
                add(h);
                for (const m of marks) {
                    const play = new St.Button({ label: m.label, style_class: 'button', x_expand: true, can_focus: true });
                    play.connect('clicked', () => sec.run(m.play));
                    const rename = btn('✎', () => sec.run(m.rename, m.rename_ask));
                    const del = btn('✕', () => sec.run(m.delete));
                    rename.x_expand = del.x_expand = false;
                    add(holder(box([play, rename, del], 'spacing: 4px;')));
                }
            }
            if (!any)
                add(new PopupMenu.PopupMenuItem(j.groups.length ? j.no_match : j.empty, { reactive: false }));
        };
        search.clutter_text.connect('text-changed', () => j && fill());
        return {
            item, update: x => {
                // only when it changed: keeps the search field's focus while typing
                if (JSON.stringify(x) === JSON.stringify(j))
                    return;
                j = x;
                fill();
            },
        };
    }
    }
    return null;
}

/* The section: `send(cmd)` runs a `loop …` command (without the `loop `),
 * `fetchView(bins, cb)` asks for the loop view, `fetchPanel(cb)` for the panel. */
var LoopSection = class {
    constructor(menu, send, fetchView, fetchPanel) {
        this._send = send;
        this._fetchView = fetchView;
        this._fetchPanel = fetchPanel;
        this.view = new LoopView((a, b) => this.view.onEdges(a, b), i => this.view.onBarOne(i));
        this.view.onEdges = () => {};
        this.view.onBarOne = () => {};
        this.hasLoop = false;
        this._rows = [];
        this._shape = '';
        this._timer = 0;
        this._busy = false;
        this.sep = new PopupMenu.PopupSeparatorMenuItem();
        this.section = new PopupMenu.PopupMenuSection();
        menu.addMenuItem(this.sep);
        menu.addMenuItem(this.section);
        menu.connect('open-state-changed', (_m, open) => {
            if (open)
                this.load();
            if (open && !this._timer) {
                this._timer = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 50, () => {
                    this._poll();
                    return GLib.SOURCE_CONTINUE;
                });
            } else if (!open && this._timer) {
                GLib.source_remove(this._timer);
                this._timer = 0;
            }
        });
    }

    /* Ask stemflow for the panel and show it. */
    load() {
        this._fetchPanel(p => this.render(p));
    }

    /* Send `lines` (`loop …`), asking for a name first (`{name}` in them) if `ask`. */
    run(ls, ask = null) {
        const go = name => {
            for (const l of ls)
                this._send(l.replace(/^loop /, '').replace('{name}', name ?? ''));
            this.load();
        };
        if (ask)
            askName(ask.title, ask.value, go);
        else
            go(null);
    }

    /* Show the panel (null: no looping): built once for its shape, then updated. */
    render(panel) {
        const rows = panel?.rows || [];
        this.sep.visible = this.section.actor.visible = rows.length > 0;
        const shape = rows.map(r => `${r.id}:${r.items.map(i => i.kind).join(',')}`).join('|');
        if (shape !== this._shape)
            this._build(rows, shape);
        this.hasLoop = false;
        rows.forEach((r, k) => {
            const built = this._rows[k];
            built.item.visible = !r.hidden;
            if (r.id === 'view')
                this.hasLoop = !r.hidden;
            r.items.forEach((i, n) => built.widgets[n]?.update(i));
        });
        if (!this.hasLoop)
            this.view.set(null);
    }

    _build(rows, shape) {
        if (this.view.get_parent())
            this.view.get_parent().remove_child(this.view);
        this.section.removeAll();
        this._rows = [];
        this._shape = shape;
        for (const r of rows) {
            const widgets = r.items.map(i => widget(i.kind, this)).filter(w => w);
            let item;
            if (widgets.length === 1 && widgets[0].item)
                item = widgets[0].item;
            else
                item = holder(box(widgets.map(w => w.actor).filter(a => a), r.id === 'edit' ? 'spacing: 10px;' : 'spacing: 6px;'));
            this.section.addMenuItem(item);
            this._rows.push({ item, widgets });
        }
    }

    _poll() {
        if (this._busy || !this.hasLoop)
            return;
        this._busy = true;
        this._fetchView(Math.max(40, Math.floor(this.view.width / 2)), pic => {
            this._busy = false;
            this.view.set(pic);
        });
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        this._timer = 0;
    }
};
