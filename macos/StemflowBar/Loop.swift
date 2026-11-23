// The Loop section of the menu: the loop panel stemflow declares (`loop panel`,
// src/loopdeck/panel.rs), drawn; and the loop view, drawn from `loop view`.

import AppKit

/// The loop as `loop view <bins>` gives it: frames from the view's left edge.
struct LoopPic {
    var playing = false
    var bpm = 0.0, meter = 4, bars = 0, a = 0, b = 0
    var to = 1.0, start = 0.0, end = 0.0, playhead = 0.0
    var beats: [(frame: Double, bar: Bool, index: Int)] = []
    var peaks: [Double] = []

    init?(_ j: [String: Any]?) {
        guard let j = j, let beats = j["beats"] as? [[Double]] else { return nil }
        let d = { (k: String) in j[k] as? Double ?? 0 }
        playing = j["playing"] as? Bool ?? false
        (bpm, meter, bars, a, b) = (d("bpm"), Int(d("meter")), Int(d("bars")), Int(d("a")), Int(d("b")))
        (to, start, end, playhead) = (max(d("to"), 1), d("start"), d("end"), d("playhead"))
        self.beats = beats.filter { $0.count == 3 }.map { ($0[0], $0[1] > 0, Int($0[2])) }
        peaks = j["peaks"] as? [Double] ?? []
    }
}

/// The waveform of the loop with a bar either side: a tick on every beat, numbered
/// bar lines, the ends and the playhead. Drag an end to move it (it snaps to the
/// beats); hold, right-click (or ⌃/⌥-click) a beat to make it bar 1.
final class LoopView: NSView {
    var pic: LoopPic? { didSet { needsDisplay = true } }
    var onEdges: (Int, Int) -> Void = { _, _ in }
    var onBarOne: (Int) -> Void = { _ in }
    private var drag = 0 // 0 none, 1 start, 2 end
    private var dragBeat = -1

    override init(frame: NSRect) { super.init(frame: frame) }
    required init?(coder: NSCoder) { fatalError() }
    override var isFlipped: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    private func x(_ f: Double) -> CGFloat { CGFloat(f / (pic?.to ?? 1)) * bounds.width }

    private func nearest(_ px: CGFloat) -> Int? {
        pic?.beats.min(by: { abs(x($0.frame) - px) < abs(x($1.frame) - px) })?.index
    }

    override func mouseDown(with e: NSEvent) {
        guard let p = pic else { return }
        let px = convert(e.locationInWindow, from: nil).x
        if !e.modifierFlags.intersection([.control, .option]).isEmpty {
            if let i = nearest(px) { onBarOne(i) }
            return
        }
        let (s, en) = (x(p.start), x(p.end))
        let grab: CGFloat = 18
        drag = abs(px - s) < grab && abs(px - s) <= abs(px - en) ? 1 : abs(px - en) < grab ? 2 : 0
        dragBeat = -1
        // held still on a beat (as on the phone): that beat is bar 1
        hold?.invalidate()
        if drag == 0 {
            hold = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { [weak self] _ in
                guard let self else { return }
                self.hold = nil
                if let i = self.nearest(px) { self.onBarOne(i) }
            }
        }
    }

    private var hold: Timer?

    override func rightMouseDown(with e: NSEvent) {
        if let i = nearest(convert(e.locationInWindow, from: nil).x) { onBarOne(i) }
    }

    override func mouseDragged(with e: NSEvent) {
        hold?.invalidate()
        hold = nil
        guard drag != 0 else { return }
        dragBeat = nearest(convert(e.locationInWindow, from: nil).x) ?? -1
        needsDisplay = true
    }

    override func mouseUp(with e: NSEvent) {
        hold?.invalidate()
        hold = nil
        if drag != 0, dragBeat >= 0, let p = pic {
            if drag == 1 && dragBeat < p.b && dragBeat != p.a { onEdges(dragBeat, p.b) }
            if drag == 2 && dragBeat > p.a && dragBeat != p.b { onEdges(p.a, dragBeat) }
        }
        drag = 0
        dragBeat = -1
        needsDisplay = true
    }

    override func draw(_ dirty: NSRect) {
        NSColor.black.withAlphaComponent(0.25).setFill()
        NSBezierPath(roundedRect: bounds, xRadius: 6, yRadius: 6).fill()
        guard let p = pic else { return }
        let (w, h) = (bounds.width, bounds.height)
        let mid = h * 0.55
        // waveform
        NSColor(red: 159 / 255, green: 182 / 255, blue: 198 / 255, alpha: 0.6).setFill()
        let bw = w / CGFloat(max(p.peaks.count, 1))
        for (k, v) in p.peaks.enumerated() {
            let a = CGFloat(v) * h * 0.4
            NSRect(x: CGFloat(k) * bw, y: mid - a, width: max(bw - 0.5, 0.5), height: 2 * a).fill()
        }
        // the loop, with the end being dragged where it would land
        var (s, e) = (x(p.start), x(p.end))
        if drag != 0, let b = p.beats.first(where: { $0.index == dragBeat }) {
            if drag == 1 { s = x(b.frame) } else { e = x(b.frame) }
        }
        NSColor.gray.withAlphaComponent(0.3).setFill()
        NSRect(x: 0, y: 0, width: s, height: h).fill(using: .sourceOver)
        NSRect(x: e, y: 0, width: w - e, height: h).fill(using: .sourceOver)
        NSColor(red: 108 / 255, green: 180 / 255, blue: 232 / 255, alpha: 0.18).setFill()
        NSRect(x: s, y: 0, width: e - s, height: h).fill(using: .sourceOver)
        // beats and bars, bars numbered from the loop's first bar line
        let green = NSColor(red: 0x79 / 255, green: 0xc2 / 255, blue: 0x67 / 255, alpha: 1)
        let font = NSFont.systemFont(ofSize: 9, weight: .semibold)
        var barNo = 0
        for bt in p.beats {
            let px = x(bt.frame)
            let line = NSBezierPath()
            if bt.bar {
                green.setStroke()
                line.lineWidth = 2
                line.move(to: NSPoint(x: px, y: 11))
                if bt.index >= p.a { barNo += 1 }
                if bt.index >= p.a && bt.index < p.b {
                    ("\(barNo)" as NSString).draw(at: NSPoint(x: px + 2, y: 0),
                                                  withAttributes: [.font: font, .foregroundColor: green])
                }
            } else {
                NSColor(red: 159 / 255, green: 182 / 255, blue: 198 / 255, alpha: 0.7).setStroke()
                line.lineWidth = 1
                line.move(to: NSPoint(x: px, y: h * 0.25))
            }
            line.line(to: NSPoint(x: px, y: h))
            line.stroke()
        }
        // the ends, with a grip, and the playhead
        NSColor(red: 0xea / 255, green: 0xf2 / 255, blue: 0xf7 / 255, alpha: 1).set()
        for px in [s, e] {
            let l = NSBezierPath()
            l.lineWidth = 2.5
            l.move(to: NSPoint(x: px, y: 0))
            l.line(to: NSPoint(x: px, y: h))
            l.stroke()
            NSBezierPath(ovalIn: NSRect(x: px - 5, y: h - 13, width: 10, height: 10)).fill()
        }
        if p.playing {
            NSColor(red: 229 / 255, green: 57 / 255, blue: 53 / 255, alpha: 1).setStroke()
            let l = NSBezierPath()
            l.lineWidth = 1.5
            l.move(to: NSPoint(x: x(p.playhead), y: 0))
            l.line(to: NSPoint(x: x(p.playhead), y: h))
            l.stroke()
        }
    }
}

/// A row of the panel: `v` with the panel's side margins, as wide as the panel.
func inset(_ v: NSView, top: CGFloat = 3, bottom: CGFloat = 3) -> NSView {
    let box = NSView()
    box.translatesAutoresizingMaskIntoConstraints = false
    v.translatesAutoresizingMaskIntoConstraints = false
    box.addSubview(v)
    NSLayoutConstraint.activate([
        v.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 14),
        v.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -14),
        v.topAnchor.constraint(equalTo: box.topAnchor, constant: top),
        v.bottomAnchor.constraint(equalTo: box.bottomAnchor, constant: -bottom),
    ])
    return box
}

final class ClosureSegments: NSSegmentedControl {
    var handler: (Int) -> Void = { _ in }
    convenience init(_ labels: [String], handler: @escaping (Int) -> Void) {
        self.init(labels: labels, trackingMode: .selectOne, target: nil, action: nil)
        self.handler = handler
        target = self
        action = #selector(fire)
        controlSize = .small
        segmentDistribution = .fillProportionally
    }
    @objc func fire() { handler(selectedSegment) }
}

func smallButton(_ title: String, _ act: @escaping () -> Void) -> NSButton {
    let b = ClosureButton(title: title, handler: act)
    b.bezelStyle = .rounded
    b.controlSize = .small
    return b
}

func smallLabel(_ s: String, size: CGFloat = 10, color: NSColor = .secondaryLabelColor) -> NSTextField {
    let l = NSTextField(labelWithString: s)
    l.font = .systemFont(ofSize: size)
    l.textColor = color
    l.alignment = .center
    return l
}

/// A one-line label that truncates rather than grow: a text that changes every
/// second (the song's time) mustn't move or resize anything.
func lineLabel(_ s: String = "", font: NSFont = .systemFont(ofSize: NSFont.smallSystemFontSize),
               color: NSColor = .secondaryLabelColor) -> NSTextField {
    let l = NSTextField(labelWithString: s)
    l.font = font
    l.textColor = color
    l.lineBreakMode = .byTruncatingTail
    l.maximumNumberOfLines = 1
    l.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
    return l
}

// MARK: - The panel

/// The loop panel as stemflow declares it (`loop panel`, src/loopdeck/panel.rs):
/// rows of items, each of a kind in PanelKind (generated from the Rust). Each kind
/// is drawn here once; the switch in `makeWidget` is exhaustive, so a kind added in
/// Rust doesn't build until it's drawn. The rows' texts, states and the lines the
/// controls send all come from stemflow.
class Widget {
    var view: NSView { NSView() }
    func update(_ j: [String: Any]) {}
}

func str(_ j: [String: Any], _ k: String) -> String { j[k] as? String ?? "" }
func lines(_ j: [String: Any], _ k: String) -> [String] { j[k] as? [String] ?? [] }

func makeWidget(_ kind: PanelKind, _ sec: LoopSection) -> Widget {
    switch kind {
    case .text: return TextWidget()
    case .button: return ButtonWidget(sec)
    case .switch: return SwitchWidget(sec)
    case .choice: return ChoiceWidget(sec)
    case .nudge: return NudgeWidget(sec)
    case .slider: return SliderWidget(sec)
    case .loopView: return LoopViewWidget(sec)
    case .bookmarks: return BookmarksWidget(sec)
    }
}

/// A line of text (truncated, never wrapped: rows keep their height).
final class TextWidget: Widget {
    let label = lineLabel()
    override var view: NSView { label }
    override func update(_ j: [String: Any]) {
        let t = str(j, "text")
        let style = str(j, "style")
        label.font = style == "title" ? .boldSystemFont(ofSize: NSFont.systemFontSize)
            : style == "small" ? .systemFont(ofSize: NSFont.smallSystemFontSize) : .systemFont(ofSize: NSFont.systemFontSize)
        label.textColor = style == "small" ? .secondaryLabelColor : .labelColor
        label.stringValue = t
        label.toolTip = t
    }
}

final class ButtonWidget: Widget {
    let b: NSButton
    var j: [String: Any] = [:]
    init(_ sec: LoopSection) {
        b = smallButton("") {}
        super.init()
        (b as? ClosureButton)?.handler = { [unowned self, unowned sec] in
            sec.run(lines(j, "send"), ask: j["ask"] as? [String: Any])
        }
    }
    override var view: NSView { b }
    override func update(_ j: [String: Any]) {
        self.j = j
        b.title = str(j, "label")
        b.isEnabled = j["enabled"] as? Bool ?? true
        if let on = j["on"] as? Bool {
            b.setButtonType(.pushOnPushOff)
            b.state = on ? .on : .off
        }
    }
}

/// A checkbox.
final class SwitchWidget: Widget {
    let b = ClosureButton(title: "") {}
    var j: [String: Any] = [:]
    init(_ sec: LoopSection) {
        super.init()
        b.setButtonType(.switch)
        b.handler = { [unowned self, unowned sec] in sec.run(lines(j, "send")) }
    }
    override var view: NSView { b }
    override func update(_ j: [String: Any]) {
        self.j = j
        b.title = str(j, "label")
        b.state = (j["on"] as? Bool ?? false) ? .on : .off
        b.toolTip = str(j, "tip")
    }
}

/// Side by side when there are few; else a pop-up ("Beats per bar: 4").
final class ChoiceWidget: Widget {
    var seg: ClosureSegments!
    let pop = NSPopUpButton(frame: .zero, pullsDown: false)
    let box = NSStackView()
    var options: [[String: Any]] = []
    unowned let sec: LoopSection
    init(_ sec: LoopSection) {
        self.sec = sec
        super.init()
        seg = ClosureSegments([]) { [unowned self, unowned sec] i in
            if i >= 0 && i < options.count { sec.run(lines(options[i], "send")) }
        }
        pop.controlSize = .small
        pop.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        pop.target = self
        pop.action = #selector(picked)
    }
    @objc func picked() {
        let i = pop.indexOfSelectedItem
        if i >= 0 && i < options.count { sec.run(lines(options[i], "send")) }
    }
    override var view: NSView { box }
    override func update(_ j: [String: Any]) {
        options = j["options"] as? [[String: Any]] ?? []
        let sel = j["selected"] as? Int ?? -1
        let labels = options.map { str($0, "label") }
        let menu = options.count > 3
        let want: NSView = menu ? pop : seg
        if box.arrangedSubviews.first !== want {
            box.arrangedSubviews.forEach { $0.removeFromSuperview() }
            box.addArrangedSubview(want)
        }
        if menu {
            let t = str(j, "title")
            let titles = labels.map { t.isEmpty ? $0 : "\(t): \($0)" }
            if pop.itemTitles != titles {
                pop.removeAllItems()
                pop.addItems(withTitles: titles)
            }
            pop.selectItem(at: sel)
        } else {
            if seg.segmentCount != labels.count { seg.segmentCount = labels.count }
            for (i, l) in labels.enumerated() where seg.label(forSegment: i) != l { seg.setLabel(l, forSegment: i) }
            seg.selectedSegment = sel
        }
    }
}

/// LABEL, ◀ ▶, a note: the start, the whole loop or the end, earlier or later.
final class NudgeWidget: Widget {
    let col = NSStackView()
    let label = smallLabel("", size: 9, color: .labelColor)
    let note = smallLabel("", size: 9)
    var j: [String: Any] = [:]
    init(_ sec: LoopSection) {
        super.init()
        let l = smallButton("◀") { [unowned self, unowned sec] in sec.run(lines(j, "back")) }
        let r = smallButton("▶") { [unowned self, unowned sec] in sec.run(lines(j, "forward")) }
        let buttons = NSStackView(views: [l, r])
        buttons.distribution = .fillEqually
        buttons.spacing = 4
        for v in [label, buttons, note] { col.addArrangedSubview(v) }
        col.orientation = .vertical
        col.spacing = 2
        col.alignment = .centerX
        buttons.widthAnchor.constraint(equalTo: col.widthAnchor).isActive = true
    }
    override var view: NSView { col }
    override func update(_ j: [String: Any]) {
        self.j = j
        label.stringValue = str(j, "label")
        note.stringValue = str(j, "note")
    }
}

final class SliderWidget: Widget {
    var row: SliderRow!
    var j: [String: Any] = [:]
    init(_ sec: LoopSection) {
        super.init()
        row = SliderRow("", inset: 0) { [unowned self, unowned sec] pos in
            let d = { (k: String) in self.j[k] as? Double ?? 0 }
            let (lo, hi, step) = (d("min"), d("max"), max(d("step"), 1e-6))
            var v = ((lo + pos * (hi - lo)) / step).rounded() * step
            if let snap = j["snap"] as? Double, abs(v - snap) < 2 * step { v = snap }
            let text = step >= 1 ? String(Int(v.rounded())) : String(format: "%.2f", v)
            sec.run([str(j, "send").replacingOccurrences(of: "{v}", with: text)])
        }
    }
    override var view: NSView { row }
    override func update(_ j: [String: Any]) {
        self.j = j
        let d = { (k: String) in j[k] as? Double ?? 0 }
        row.label.stringValue = str(j, "label")
        let span = d("max") - d("min")
        row.set(span > 0 ? (d("value") - d("min")) / span : 0, str(j, "text"))
    }
}

/// The loop drawn (from `loop view`, polled while the panel shows it).
final class LoopViewWidget: Widget {
    var j: [String: Any] = [:]
    let v: LoopView
    init(_ sec: LoopSection) {
        v = sec.view
        super.init()
        v.translatesAutoresizingMaskIntoConstraints = false
        v.heightAnchor.constraint(equalToConstant: 96).isActive = true
        sec.view.onEdges = { [unowned self, unowned sec] a, b in
            sec.run([str(j, "edges").replacingOccurrences(of: "{a}", with: "\(a)")
                .replacingOccurrences(of: "{b}", with: "\(b)")])
        }
        sec.view.onBarOne = { [unowned self, unowned sec] i in
            sec.run([str(j, "bar1").replacingOccurrences(of: "{beat}", with: "\(i)")])
        }
    }
    override var view: NSView { v }
    override func update(_ j: [String: Any]) { self.j = j }
}

/// The saved loops: a button that drops a menu with a search field, by
/// recording; each plays, renames or deletes.
final class BookmarksWidget: Widget {
    let menu = NSMenu()
    var b: NSButton!
    let search = NSSearchField(frame: NSRect(x: 0, y: 0, width: 260, height: 22))
    var j: [String: Any] = [:]
    unowned let sec: LoopSection
    init(_ sec: LoopSection) {
        self.sec = sec
        super.init()
        b = smallButton("Bookmarks") { [unowned self] in
            fill()
            menu.popUp(positioning: nil, at: NSPoint(x: 0, y: b.bounds.height + 4), in: b)
        }
        search.sendsSearchStringImmediately = true
        search.target = self
        search.action = #selector(searched)
        let sf = NSView(frame: NSRect(x: 0, y: 0, width: 290, height: 30))
        search.frame = NSRect(x: 14, y: 4, width: 262, height: 22)
        sf.addSubview(search)
        menu.addItem(viewItem(sf))
    }
    override var view: NSView { b }
    @objc func searched() { fill() }
    override func update(_ j: [String: Any]) {
        // only when it changed: keeps the search field's focus while typing
        if NSDictionary(dictionary: j).isEqual(to: self.j) { return }
        self.j = j
        b.title = str(j, "title") + " ▾"
        fill()
    }
    func fill() {
        let m = menu
        while m.items.count > 1 { m.removeItem(at: m.items.count - 1) }
        search.placeholderString = str(j, "search_hint")
        let q = search.stringValue.lowercased()
        let groups = j["groups"] as? [[String: Any]] ?? []
        var any = false
        for g in groups {
            let marks = (g["marks"] as? [[String: Any]] ?? []).filter { q.isEmpty || str($0, "search").contains(q) }
            if marks.isEmpty { continue }
            any = true
            m.addItem(.separator())
            let h = textItem()
            h.title = str(g, "title")
            m.addItem(h)
            for b in marks {
                let i = NSMenuItem(title: str(b, "label"), action: nil, keyEquivalent: "")
                let sub = NSMenu()
                sub.addItem(ClosureItem("Play it") { [unowned self] in sec.run(lines(b, "play")) })
                sub.addItem(ClosureItem("Rename…") { [unowned self] in
                    sec.run(lines(b, "rename"), ask: b["rename_ask"] as? [String: Any])
                })
                sub.addItem(ClosureItem("Delete") { [unowned self] in sec.run(lines(b, "delete")) })
                i.submenu = sub
                m.addItem(i)
            }
        }
        if !any {
            let t = textItem()
            t.title = groups.isEmpty ? str(j, "empty") : str(j, "no_match")
            m.addItem(t)
        }
    }
}

/// The Loop part of the panel: a vertical stack of rows, built once for the
/// panel's shape (row ids and item kinds) and then only updated, so nothing is
/// rebuilt under the pointer.
final class LoopSection: NSObject {
    let send: (String) -> Void
    let view = LoopView(frame: NSRect(x: 0, y: 0, width: 352, height: 96))
    /// Where the rows go (Controller puts it in the panel).
    let box = NSStackView()
    var rows: [(view: NSView, widgets: [Widget])] = []
    var shape = ""
    var hasLoop = false
    /// After a change in which rows show: the panel fits itself to them.
    var onResize: () -> Void = {}

    init(send: @escaping (String) -> Void) {
        self.send = send
        super.init()
        box.orientation = .vertical
        box.alignment = .leading
        box.spacing = 2
        box.translatesAutoresizingMaskIntoConstraints = false
    }

    /// Send `lines`, asking for a name first (`{name}` in them) if `ask` says so.
    func run(_ lines: [String], ask a: [String: Any]? = nil) {
        guard let a = a else {
            for l in lines { send(l) }
            Controller.shared.loadPanel()
            return
        }
        DispatchQueue.main.async { [self] in
            guard let name = ask(str(a, "title"), str(a, "text"), value: str(a, "value"), ok: str(a, "ok")) else { return }
            for l in lines { send(l.replacingOccurrences(of: "{name}", with: name)) }
            Controller.shared.loadPanel()
        }
    }

    /// A name from the person (nil: cancelled).
    func ask(_ title: String, _ text: String, value: String, ok: String) -> String? {
        NSApp.activate(ignoringOtherApps: true)
        let a = NSAlert()
        a.messageText = title
        a.informativeText = text
        a.addButton(withTitle: ok.isEmpty ? "OK" : ok)
        a.addButton(withTitle: "Cancel")
        let f = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        f.stringValue = value
        a.accessoryView = f
        a.window.initialFirstResponder = f
        guard a.runModal() == .alertFirstButtonReturn else { return nil }
        let name = f.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? nil : name
    }

    /// Show the panel (nil: no looping): built once for its shape, then updated.
    func render(_ panel: [String: Any]?) {
        let rowsJ = panel?["rows"] as? [[String: Any]] ?? []
        let sig = rowsJ.map { r in
            str(r, "id") + ":" + (r["items"] as? [[String: Any]] ?? []).map { str($0, "kind") }.joined(separator: ",")
        }.joined(separator: "|")
        var changed = false
        if sig != shape { build(rowsJ, sig); changed = true }
        if box.isHidden != rowsJ.isEmpty { box.isHidden = rowsJ.isEmpty; changed = true }
        hasLoop = false
        for (r, j) in zip(rows, rowsJ) {
            let hidden = j["hidden"] as? Bool ?? false
            if r.view.isHidden != hidden { r.view.isHidden = hidden; changed = true }
            if str(j, "id") == "view" { hasLoop = !hidden }
            for (w, i) in zip(r.widgets, j["items"] as? [[String: Any]] ?? []) { w.update(i) }
        }
        if !hasLoop { view.pic = nil }
        if changed { onResize() }
    }

    private func build(_ rowsJ: [[String: Any]], _ sig: String) {
        for r in rows { r.view.removeFromSuperview() }
        rows = []
        shape = sig
        for j in rowsJ {
            let ws: [Widget] = (j["items"] as? [[String: Any]] ?? []).map { i in
                makeWidget(PanelKind(rawValue: str(i, "kind")) ?? .text, self)
            }
            let stack = NSStackView(views: ws.map { $0.view })
            stack.spacing = 6
            // buttons share the width evenly; a pop-up beside segments takes its
            // own width and the segments the rest
            stack.distribution = ws.contains { $0 is ChoiceWidget } ? .fill : .fillEqually
            let row = inset(stack)
            box.addArrangedSubview(row)
            row.widthAnchor.constraint(equalTo: box.widthAnchor).isActive = true
            rows.append((row, ws))
        }
    }
}

/// `m:ss`
func mmss(_ ms: Int) -> String { String(format: "%d:%02d", ms / 60000, ms / 1000 % 60) }
