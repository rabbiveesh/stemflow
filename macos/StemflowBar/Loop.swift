// The Loop section of the menu: the Android app's Loops screen (LoopView.kt,
// MainActivity.loopsPage) over the control socket's `loop …` requests
// (src/loopdeck.rs).

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
/// beats); right-click (or ⌃/⌥-click) a beat to make it bar 1.
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
    }

    override func rightMouseDown(with e: NSEvent) {
        if let i = nearest(convert(e.locationInWindow, from: nil).x) { onBarOne(i) }
    }

    override func mouseDragged(with e: NSEvent) {
        guard drag != 0 else { return }
        dragBeat = nearest(convert(e.locationInWindow, from: nil).x) ?? -1
        needsDisplay = true
    }

    override func mouseUp(with e: NSEvent) {
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

/// A view with its subviews in a stack, padded like the other menu rows.
func padded(_ v: NSView, height: CGFloat, width: CGFloat = 360) -> NSView {
    let box = NSView(frame: NSRect(x: 0, y: 0, width: width, height: height))
    v.translatesAutoresizingMaskIntoConstraints = false
    box.addSubview(v)
    NSLayoutConstraint.activate([
        v.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 14),
        v.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -14),
        v.topAnchor.constraint(equalTo: box.topAnchor, constant: 3),
        v.bottomAnchor.constraint(equalTo: box.bottomAnchor, constant: -3),
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
        segmentDistribution = .fillEqually
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

final class LoopSection: NSObject {
    let send: (String) -> Void
    let sep = NSMenuItem.separator()
    let info = textItem()
    let hint = textItem()
    var grabs: [Int: NSButton] = [:]
    var live: NSButton!
    var grabItem: NSMenuItem!
    let view = LoopView(frame: NSRect(x: 0, y: 0, width: 332, height: 96))
    var viewItem_: NSMenuItem!
    var units: ClosureSegments!
    var unitsItem: NSMenuItem!
    var editItem: NSMenuItem!
    var meters: ClosureSegments!, feels: ClosureSegments!
    var beatItem: NSMenuItem!
    var speed: SliderRow!, pitch: SliderRow!
    var speedItem: NSMenuItem!, pitchItem: NSMenuItem!
    var keep: NSButton!, click: NSButton!
    var chipsItem: NSMenuItem!
    var actionsItem: NSMenuItem!
    // the player: what plays, play/pause, save; bookmarks
    let nowItem = textItem()
    var playPause: NSButton!, save: NSButton!
    var stripItem: NSMenuItem!
    let bookmarksItem = NSMenuItem(title: "Bookmarks", action: nil, keyEquivalent: "")
    let stemsOnly = ClosureItem("Stems only when looping") {}
    var marks: [[String: Any]] = []
    var search = ""
    let searchField = NSSearchField(frame: NSRect(x: 0, y: 0, width: 260, height: 22))
    /// What the edits step by: 0 a beat, 1 a bar, 2 a phrase (the loop's length).
    var unit = 0
    var loop: [String: Any]? // the state's "loop"
    var meterValues = [0, 2, 3, 4, 5, 6, 7]
    var feelValues = [1.0, 0.5, 2.0]

    var items: [NSMenuItem] {
        [sep, nowItem, stripItem, bookmarksItem, info, hint, grabItem, viewItem_, unitsItem, editItem, beatItem,
         speedItem, pitchItem, chipsItem, actionsItem, stemsOnly]
    }

    init(send: @escaping (String) -> Void) {
        self.send = send
        super.init()
        let grabRow = NSStackView()
        grabRow.distribution = .fillEqually
        grabRow.spacing = 6
        for n in [1, 2, 4, 8] {
            let b = smallButton(n == 1 ? "1 bar" : "\(n) bars") { send("loop \(n)") }
            b.setButtonType(.pushOnPushOff)
            grabs[n] = b
            grabRow.addArrangedSubview(b)
        }
        live = smallButton("Live") { send("loop live") }
        grabRow.addArrangedSubview(live)
        grabItem = viewItem(padded(grabRow, height: 30))

        view.onEdges = { a, b in send("loop edges \(a) \(b)") }
        view.onBarOne = { i in send("loop bar1 \(i)") }
        viewItem_ = viewItem(padded(view, height: 102))

        units = ClosureSegments(["Beat", "Bar", "Phrase"]) { [unowned self] i in unit = i }
        units.selectedSegment = 0
        unitsItem = viewItem(padded(units, height: 28))

        // Start, the whole loop, or the end: earlier or later by the unit.
        let pair = { (name: String, note: String, da: Int, db: Int) -> NSView in
            let l = smallButton("◀") { self.edit(-da, -db) }
            let r = smallButton("▶") { self.edit(da, db) }
            let buttons = NSStackView(views: [l, r])
            buttons.distribution = .fillEqually
            buttons.spacing = 4
            let col = NSStackView(views: [smallLabel(name.uppercased(), size: 9, color: .labelColor), buttons,
                                          smallLabel(note, size: 9)])
            col.orientation = .vertical
            col.spacing = 2
            col.alignment = .centerX
            buttons.widthAnchor.constraint(equalTo: col.widthAnchor).isActive = true
            return col
        }
        let edits = NSStackView(views: [pair("Start", "◀ longer · shorter ▶", 1, 0),
                                        pair("Move", "same length", 1, 1),
                                        pair("End", "◀ shorter · longer ▶", 0, 1)])
        edits.distribution = .fillEqually
        edits.spacing = 10
        editItem = viewItem(padded(edits, height: 62))

        // Meter and feel re-decode the loop's bars (and the click's).
        meters = ClosureSegments(["Auto", "2", "3", "4", "5", "6", "7"]) { [unowned self] i in
            send("set meter \(meterValues[i])")
        }
        feels = ClosureSegments(["As heard", "½×", "2×"]) { [unowned self] i in
            send("set feel \(feelValues[i])")
        }
        let beatRow = NSStackView(views: [meters, feels])
        beatRow.spacing = 8
        meters.widthAnchor.constraint(equalTo: feels.widthAnchor, multiplier: 1.4).isActive = true
        beatItem = viewItem(padded(beatRow, height: 28))

        // Practice speed (40-120%) and transposition (±12 semitones).
        speed = SliderRow("Speed") { v in
            var x = 0.4 + 0.8 * v
            if abs(x - 1) < 0.02 { x = 1 }
            send(String(format: "loop speed %.2f", x))
        }
        pitch = SliderRow("Pitch") { v in send("loop pitch \(Int((v * 24 - 12).rounded()))") }
        (speedItem, pitchItem) = (viewItem(speed), viewItem(pitch))

        keep = smallButton("Keep pitch") { [unowned self] in
            send("loop keep_pitch \((loop?["keep_pitch"] as? Bool ?? true) ? "off" : "on")")
        }
        keep.setButtonType(.pushOnPushOff)
        click = smallButton("Click") { [unowned self] in
            send("loop click \((loop?["click"] as? Bool ?? false) ? "off" : "on")")
        }
        click.setButtonType(.pushOnPushOff)
        let chips = NSStackView(views: [
            smallButton("75%") { send("loop speed 0.75") },
            smallButton("50%") { send("loop speed 0.5") },
            keep, click,
            smallButton("Reset") { send("loop speed 1"); send("loop pitch 0") },
        ])
        chips.spacing = 4
        chipsItem = viewItem(padded(chips, height: 30))
        // From the top; Repair plays the loop's part of the song again and lays the
        // loop on that (for a glitch in it).
        let actions = NSStackView(views: [smallButton("From the top") { send("loop top") },
                                          smallButton("Repair") { send("loop repair") }])
        actions.distribution = .fillEqually
        actions.spacing = 6
        actionsItem = viewItem(padded(actions, height: 30))

        // The player: play/pause it, and save the loop as a bookmark.
        playPause = smallButton("Pause") { send("loop playpause") }
        save = smallButton("Save…") { [unowned self] in saveLoop() }
        let strip = NSStackView(views: [playPause, save])
        strip.distribution = .fillEqually
        strip.spacing = 6
        stripItem = viewItem(padded(strip, height: 30))
        bookmarksItem.submenu = NSMenu()
        searchField.placeholderString = "Search songs and loops"
        searchField.sendsSearchStringImmediately = true
        searchField.target = self
        searchField.action = #selector(searched)
        stemsOnly.handler = { [unowned self] in
            send("loop stems_only \((loop?["stems_only"] as? Bool ?? false) ? "off" : "on")")
        }
        stemsOnly.toolTip = "Hear the plain mix and save power; a loop is separated when you make it"
    }

    @objc func searched() {
        search = searchField.stringValue
        fillBookmarks(marks)
    }

    /// A name from the person (nil: cancelled).
    func ask(_ title: String, _ text: String, value: String) -> String? {
        NSApp.activate(ignoringOtherApps: true)
        let a = NSAlert()
        a.messageText = title
        a.informativeText = text
        a.addButton(withTitle: "Save")
        a.addButton(withTitle: "Cancel")
        let f = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        f.stringValue = value
        a.accessoryView = f
        a.window.initialFirstResponder = f
        guard a.runModal() == .alertFirstButtonReturn else { return nil }
        let name = f.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return name.isEmpty ? nil : name
    }

    func saveLoop() {
        let title = loop?["title"] as? String ?? ""
        let from = loop?["loop_from"] as? Int ?? -1
        let guess = title.isEmpty ? "Loop" : from >= 0 ? "\(title) \(mmss(from))" : title
        // after the menu has closed
        DispatchQueue.main.async { [self] in
            guard let name = ask("Save this loop", "Name it to come back to it later.", value: guess) else { return }
            send("loop save \(name)")
            Controller.shared.loadBookmarks()
        }
    }

    /// The saved loops, by recording; each one recalls, renames or deletes.
    func fillBookmarks(_ list: [[String: Any]]) {
        marks = list
        let m = bookmarksItem.submenu!
        // keep the search field (and its focus) while typing
        while m.items.count > 1 { m.removeItem(at: m.items.count - 1) }
        if m.items.isEmpty {
            let sf = NSView(frame: NSRect(x: 0, y: 0, width: 290, height: 30))
            searchField.frame = NSRect(x: 14, y: 4, width: 262, height: 22)
            sf.addSubview(searchField)
            m.addItem(viewItem(sf))
        }
        let q = search.lowercased()
        let shown = list.filter { b in
            q.isEmpty || ["name", "title", "artist"].contains { (b[$0] as? String ?? "").lowercased().contains(q) }
        }
        bookmarksItem.title = list.isEmpty ? "Bookmarks" : "Bookmarks (\(list.count))"
        if shown.isEmpty {
            let t = textItem()
            t.title = list.isEmpty ? "Save a loop (Save…) to come back to it later" : "Nothing matches"
            m.addItem(t)
            return
        }
        // by recording, the newest saved first
        var order: [String] = []
        var groups: [String: [[String: Any]]] = [:]
        for b in shown.sorted(by: { ($0["created"] as? Int ?? 0) > ($1["created"] as? Int ?? 0) }) {
            let key = "\(b["title"] as? String ?? "") — \(b["artist"] as? String ?? "")"
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(b)
        }
        for key in order {
            m.addItem(.separator())
            let h = textItem()
            h.title = key
            m.addItem(h)
            for b in groups[key]!.sorted(by: { ($0["start"] as? Int ?? 0) < ($1["start"] as? Int ?? 0) }) {
                let created = b["created"] as? Int ?? 0
                let name = b["name"] as? String ?? "Loop"
                let bars = b["bars"] as? Int ?? 0
                let i = NSMenuItem(title: "\(name)  ·  \(bars) bar\(bars == 1 ? "" : "s") · \(mmss(b["start"] as? Int ?? 0))",
                                   action: nil, keyEquivalent: "")
                let sub = NSMenu()
                sub.addItem(ClosureItem("Play it") { [unowned self] in send("loop recall \(created)") })
                sub.addItem(ClosureItem("Rename…") { [unowned self] in
                    DispatchQueue.main.async { [self] in
                        guard let n = ask("Rename the loop", "", value: name) else { return }
                        send("loop rename \(created) \(n)")
                        Controller.shared.loadBookmarks()
                    }
                })
                sub.addItem(ClosureItem("Delete") { [unowned self] in
                    send("loop delete \(created)")
                    Controller.shared.loadBookmarks()
                })
                i.submenu = sub
                m.addItem(i)
            }
        }
    }

    /// The step for an edit, in beats.
    func step() -> Int {
        let meter = loop?["meter"] as? Int ?? 4
        let beats = loop?["beats"] as? Int ?? meter
        return [1, max(meter, 1), max(beats, 1)][unit]
    }

    func edit(_ da: Int, _ db: Int) {
        let u = step()
        send("loop move \(da * u) \(db * u)")
    }

    var hasLoop: Bool { (loop?["playing"] as? Bool ?? false) }

    /// From the state: `loop` is the state's "loop" (nil: no looping), `meter`/`feel`
    /// the beat options.
    func sync(_ l: [String: Any]?, meter: Int, feel: Double) {
        loop = l
        guard let l = l else {
            for i in items { i.isHidden = true }
            return
        }
        for i in items { i.isHidden = false }
        let playing = l["playing"] as? Bool ?? false
        let finding = l["finding"] as? Bool ?? false
        let tail = l["tail"] as? Bool ?? false
        let bars = l["bars"] as? Int ?? 0
        let m = max(l["meter"] as? Int ?? 4, 1)
        let beats = l["beats"] as? Int ?? 0
        let status = l["status"] as? String ?? ""
        for r in [viewItem_, unitsItem, editItem, chipsItem, actionsItem] { r!.isHidden = !playing }
        // the player
        let title = l["title"] as? String ?? "", artist = l["artist"] as? String ?? ""
        let songPlaying = l["song_playing"] as? Bool ?? false
        let from = l["loop_from"] as? Int ?? -1, to = l["loop_to"] as? Int ?? -1
        if title.isEmpty {
            nowItem.title = "The player doesn't say what's playing"
        } else if playing && from >= 0 {
            nowItem.title = "\(title) — \(artist) · looping \(mmss(from))–\(mmss(to))"
        } else {
            nowItem.title = "\(title) — \(artist) · \(mmss(l["song_ms"] as? Int ?? 0))\(songPlaying ? "" : " · paused")"
        }
        playPause.title = songPlaying ? "Pause" : "Play"
        playPause.isEnabled = !title.isEmpty && !playing
        save.isEnabled = playing
        stemsOnly.state = (l["stems_only"] as? Bool ?? false) ? .on : .off
        for (n, b) in grabs { b.state = playing && n == bars ? .on : .off }
        live.isEnabled = playing || tail || finding
        if playing {
            let (whole, extra) = (beats / m, beats % m)
            info.title = "Looping \(whole) bar\(whole == 1 ? "" : "s")"
                + (extra > 0 ? " + \(extra) beat\(extra == 1 ? "" : "s")" : "") + " of \(m)/4"
                + String(format: " · %.0f bpm · %.1f s", l["bpm"] as? Double ?? 0, l["length"] as? Double ?? 0)
        } else if l["recalling"] as? Bool ?? false {
            info.title = "Going back to a bookmark…"
        } else if finding {
            info.title = "Finding the bars…"
        } else if tail {
            info.title = "Catching up with the song"
        } else {
            info.title = "Loop the last…"
        }
        hint.title = !status.isEmpty && !finding ? status
            : playing ? "Drag an end · right-click a beat for bar 1"
            : finding ? "" : "Press a length to loop what you just heard"
        hint.isHidden = hint.title.isEmpty
        units.setLabel(beats > 0 && beats % m == 0 ? "\(beats / m)-bar phrase" : "Phrase", forSegment: 2)
        meters.selectedSegment = meterValues.firstIndex(of: meter) ?? 0
        feels.selectedSegment = feelValues.firstIndex(of: feel) ?? 0
        let sp = l["speed"] as? Double ?? 1, st = l["semis"] as? Double ?? 0
        speed.set((sp - 0.4) / 0.8, "\(Int((sp * 100).rounded()))%")
        pitch.set((st + 12) / 24, st == 0 ? "0 st" : String(format: "%+d st", Int(st.rounded())))
        keep.state = (l["keep_pitch"] as? Bool ?? true) ? .on : .off
        click.state = (l["click"] as? Bool ?? false) ? .on : .off
        if !playing { view.pic = nil }
    }
}

/// `m:ss`
func mmss(_ ms: Int) -> String { String(format: "%d:%02d", ms / 60000, ms / 1000 % 60) }
