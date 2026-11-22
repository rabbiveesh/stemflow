/* The Loop section of the menu: the Android app's Loops screen (LoopView.kt,
 * MainActivity.loopsPage) over D-Bus: `Loop(cmd)`, `LoopState` and `LoopView(bins)`
 * (src/loopdeck.rs). */
const { GObject, St, GLib, Clutter } = imports.gi;
const PopupMenu = imports.ui.popupMenu;
const Slider = imports.ui.slider;

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

/* Buttons where one is picked: [labels], onPick(index). */
function choice(labels, onPick) {
    const buttons = labels.map((l, i) => btn(l, () => onPick(i), true));
    const row = box(buttons, 'spacing: 4px;');
    row.pick = i => buttons.forEach((b, k) => b.set_checked(k === i));
    row.buttons = buttons;
    return row;
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
    const row = box([new St.Label({ text: label, y_align: Clutter.ActorAlign.CENTER, style: 'min-width: 3.5em;' }), s, value], 'spacing: 8px;');
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

const METERS = [0, 2, 3, 4, 5, 6, 7];
const FEELS = [1, 0.5, 2];

/* The section: `send(cmd)` runs a loop command, `setBeat(prop, v)` sets Meter/Feel,
 * `fetchView(bins, cb)` asks for the loop view. */
var LoopSection = class {
    constructor(menu, send, setBeat, fetchView) {
        this._send = send;
        this._fetchView = fetchView;
        this.unit = 0; // what edits step by: 0 a beat, 1 a bar, 2 a phrase
        this.loop = null;
        this._timer = 0;
        this._busy = false;

        this.sep = new PopupMenu.PopupSeparatorMenuItem();
        this.info = new PopupMenu.PopupMenuItem('', { reactive: false });
        this.info.label.style = 'font-weight: bold;';
        this.hint = new PopupMenu.PopupMenuItem('', { reactive: false });
        this.hint.label.style = 'font-size: smaller;';

        this._grabs = {};
        const grabs = [1, 2, 4, 8].map(n => {
            const b = btn(n === 1 ? '1 bar' : `${n} bars`, () => send(`${n}`), true);
            this._grabs[n] = b;
            return b;
        });
        this._live = btn('Live', () => send('live'));
        this.grab = holder(box([...grabs, this._live]));

        this.view = new LoopView((a, b) => send(`edges ${a} ${b}`), i => send(`bar1 ${i}`));
        this.viewItem = holder(this.view);

        this._units = choice(['Beat', 'Bar', 'Phrase'], i => {
            this.unit = i;
            this._units.pick(i);
        });
        this._units.pick(0);
        this.units = holder(this._units);

        // Start, the whole loop, or the end: earlier or later by the unit.
        const pair = (name, note, da, db) => box([
            small(name.toUpperCase(), 'font-size: 8pt; font-weight: bold;'),
            box([btn('◀', () => this._edit(-da, -db)), btn('▶', () => this._edit(da, db))], 'spacing: 4px;'),
            small(note),
        ], 'spacing: 2px;', true);
        this.edit = holder(box([pair('Start', '◀ longer · shorter ▶', 1, 0), pair('Move', 'same length', 1, 1),
            pair('End', '◀ shorter · longer ▶', 0, 1)], 'spacing: 10px;'));

        // Meter and feel re-decode the loop's bars (and the click's).
        this._meters = choice(['Auto', '2', '3', '4', '5', '6', '7'], i => setBeat('Meter', METERS[i]));
        this._feels = choice(['As heard', '½×', '2×'], i => setBeat('Feel', FEELS[i]));
        this.meters = holder(this._meters);
        this.feels = holder(this._feels);

        // Practice speed (40-120%) and transposition (±12 semitones).
        this._speed = sliderRow('Speed', v => {
            let x = 0.4 + 0.8 * v;
            if (Math.abs(x - 1) < 0.02)
                x = 1;
            send(`speed ${x.toFixed(2)}`);
        });
        this._pitch = sliderRow('Pitch', v => send(`pitch ${Math.round(v * 24 - 12)}`));
        this.speed = holder(this._speed);
        this.pitch = holder(this._pitch);

        this._keep = btn('Keep pitch', () => send(`keep_pitch ${this.loop?.keep_pitch === false ? 'on' : 'off'}`), true);
        this._click = btn('Click', () => send(`click ${this.loop?.click ? 'off' : 'on'}`), true);
        this.chips = holder(box([
            btn('75%', () => send('speed 0.75')),
            btn('50%', () => send('speed 0.5')),
            this._keep, this._click,
            btn('Reset', () => {
                send('speed 1');
                send('pitch 0');
            }),
            btn('From the top', () => send('top')),
        ], 'spacing: 4px;'));

        this.items = [this.sep, this.info, this.hint, this.grab, this.viewItem, this.units, this.edit,
            this.meters, this.feels, this.speed, this.pitch, this.chips];
        for (const i of this.items)
            menu.addMenuItem(i);

        menu.connect('open-state-changed', (_m, open) => {
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

    _poll() {
        if (this._busy || !this.loop?.playing)
            return;
        this._busy = true;
        this._fetchView(Math.max(40, Math.floor(this.view.width / 2)), pic => {
            this._busy = false;
            this.view.set(pic);
        });
    }

    _step() {
        const m = Math.max(this.loop?.meter || 4, 1);
        return [1, m, Math.max(this.loop?.beats || m, 1)][this.unit];
    }

    _edit(da, db) {
        const u = this._step();
        this._send(`move ${da * u} ${db * u}`);
    }

    /* `l`: the parsed LoopState (null: no looping); `meter`, `feel`: the beat options. */
    sync(l, meter, feel) {
        this.loop = l;
        for (const i of this.items)
            i.visible = !!l;
        if (!l)
            return;
        const m = Math.max(l.meter || 4, 1);
        const beats = l.beats || 0;
        for (const i of [this.viewItem, this.units, this.edit, this.chips])
            i.visible = l.playing;
        for (const [n, b] of Object.entries(this._grabs))
            b.set_checked(l.playing && Number(n) === l.bars);
        this._live.reactive = l.playing || l.tail || l.finding;
        let info;
        if (l.playing) {
            const [whole, extra] = [Math.floor(beats / m), beats % m];
            info = `Looping ${whole} bar${whole === 1 ? '' : 's'}` +
                (extra ? ` + ${extra} beat${extra === 1 ? '' : 's'}` : '') +
                ` of ${m}/4 · ${Math.round(l.bpm)} bpm · ${l.length.toFixed(1)} s`;
        } else if (l.finding) {
            info = 'Finding the bars…';
        } else if (l.tail) {
            info = 'Catching up with the song';
        } else {
            info = 'Loop the last…';
        }
        this.info.label.text = info;
        const hint = l.status && !l.finding ? l.status
            : l.playing ? 'Drag an end · right-click or long-press a beat for bar 1'
                : l.finding ? '' : 'Press a length to loop what you just heard';
        this.hint.label.text = hint;
        this.hint.visible = !!hint;
        this._units.buttons[2].label = beats && beats % m === 0 ? `${beats / m}-bar phrase` : 'Phrase';
        this._meters.pick(Math.max(METERS.indexOf(meter), 0));
        this._feels.pick(Math.max(FEELS.indexOf(feel), 0));
        this._speed.set((l.speed - 0.4) / 0.8, `${Math.round(l.speed * 100)}%`);
        const st = Math.round(l.semis);
        this._pitch.set((l.semis + 12) / 24, st === 0 ? '0 st' : `${st > 0 ? '+' : ''}${st} st`);
        this._keep.set_checked(l.keep_pitch !== false);
        this._click.set_checked(!!l.click);
        if (!l.playing)
            this.view.set(null);
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        this._timer = 0;
    }
};
