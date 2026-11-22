// stemflow — menu bar control for macOS, the counterpart of the GNOME extension
// (gnome/stemflow@veesh/extension.js).
//
// Talks to `stemflow live` over its control socket (src/socket.rs) and starts / stops
// it as a child process, the way the extension uses the systemd user unit. All the
// audio logic, the loudness scale and the remembered levels live in stemflow; this
// is a remote control.
//
// Built into Stemflow.app by macos/install.sh, which writes the checkout's path into
// Info.plist (StemflowRoot).

import AppKit
import Carbon

let maxBass = 2.0 // the daemon's limit: twice as loud as the original
let step = 0.1 // hotkey step on the -1..1 line (scrolling moves half of it)

// MARK: - Paths and files shared with stemflow

let home = FileManager.default.homeDirectoryForCurrentUser
let root: URL = {
    if let r = Bundle.main.object(forInfoDictionaryKey: "StemflowRoot") as? String {
        return URL(fileURLWithPath: r)
    }
    return home.appendingPathComponent("personal/stemflow")
}()

/// src/socket.rs's `path()` on macOS.
let socketPath: String = ProcessInfo.processInfo.environment["STEMFLOW_SOCKET"]
    ?? home.appendingPathComponent("Library/Application Support/stemflow/control.sock").path

/// stemflow's state dir (src/control.rs `state_file`): the model it will load on start.
func lastModel() -> String {
    let base = ProcessInfo.processInfo.environment["XDG_STATE_HOME"].map { URL(fileURLWithPath: $0) }
        ?? home.appendingPathComponent(".local/state")
    let f = base.appendingPathComponent("stemflow/model")
    return (try? String(contentsOf: f, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
}

/// Extra flags for `stemflow live`, as for the systemd unit: STEMFLOW_ARGS="..." in
/// ~/.config/stemflow/env.
func extraArgs() -> [String] {
    let f = home.appendingPathComponent(".config/stemflow/env")
    guard let text = try? String(contentsOf: f, encoding: .utf8) else { return [] }
    for line in text.split(separator: "\n") {
        let l = line.trimmingCharacters(in: .whitespaces)
        guard l.hasPrefix("STEMFLOW_ARGS=") else { continue }
        var v = String(l.dropFirst("STEMFLOW_ARGS=".count))
        if v.count >= 2, let q = v.first, q == "\"" || q == "'", v.last == q {
            v = String(v.dropFirst().dropLast())
        }
        return v.split(separator: " ").map(String.init)
    }
    return []
}

let logURL = home.appendingPathComponent("Library/Logs/stemflow.log")

// MARK: - Control socket

/// One request, one JSON line back (the whole state). nil when stemflow isn't there.
func request(_ line: String) -> [String: Any]? {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return nil }
    defer { close(fd) }
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let cap = MemoryLayout.size(ofValue: addr.sun_path)
    _ = withUnsafeMutablePointer(to: &addr.sun_path) {
        $0.withMemoryRebound(to: CChar.self, capacity: cap) { strncpy($0, socketPath, cap - 1) }
    }
    let ok = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard ok == 0 else { return nil }
    var tv = timeval(tv_sec: 2, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    let out = Array((line + "\n").utf8)
    guard write(fd, out, out.count) == out.count else { return nil }
    var data = Data()
    var buf = [UInt8](repeating: 0, count: 65536)
    while !data.contains(UInt8(ascii: "\n")) {
        let n = read(fd, &buf, buf.count)
        if n <= 0 { break }
        data.append(buf, count: n)
    }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

// MARK: - Stems (as src/models.rs knows them)

let stems = ["bass", "bass+vocals", "vocals", "other", "piano", "guitar", "drums"]
let stemLabels = [
    "bass": "Bass", "bass+vocals": "Bass + vocals", "vocals": "Vocals (karaoke)",
    "other": "Other (horns, strings…)", "piano": "Piano", "guitar": "Guitar", "drums": "Drums",
]

/// A model's id ends in `_<stem>`, or `_<stem>+<stem>`; anything else is a bass model.
func stemOf(_ id: String) -> String {
    if let r = id.range(of: "_[a-z]+\\+[a-z]+$", options: .regularExpression) {
        return String(id[r].dropFirst())
    }
    return stems.first { id.hasSuffix("_\($0)") } ?? "bass"
}

func cap(_ s: String) -> String { s.prefix(1).uppercased() + s.dropFirst() }

// MARK: - The icon: GNOME's clef, recoloured, with the leaf and drops on top

let clefImage = Bundle.main.url(forResource: "stemflow-clef-symbolic", withExtension: "svg")
    .flatMap { NSImage(contentsOf: $0) }
let accentsImage = Bundle.main.url(forResource: "stemflow-accents", withExtension: "svg")
    .flatMap { NSImage(contentsOf: $0) }
let clefOn = NSColor(red: 0x5a / 255.0, green: 0xa9 / 255.0, blue: 0xe0 / 255.0, alpha: 1) // water-blue
let clefWarn = NSColor(red: 0xf6 / 255.0, green: 0xd3 / 255.0, blue: 0x2d / 255.0, alpha: 1) // falling behind

/// `tint` nil: the menu bar's own text colour (off). `alpha` < 1: starting up.
func statusIcon(tint: NSColor?, alpha: CGFloat = 1) -> NSImage {
    guard let clef = clefImage, let accents = accentsImage else {
        let i = NSImage(systemSymbolName: "music.quarternote.3", accessibilityDescription: "stemflow")!
        i.isTemplate = tint == nil
        return i
    }
    let size = NSSize(width: 20, height: 20)
    let img = NSImage(size: size, flipped: false) { r in
        let colour = tint ?? .labelColor // resolved when drawn, so it follows the bar
        let tinted = NSImage(size: size, flipped: false) { rr in
            clef.draw(in: rr)
            colour.set()
            rr.fill(using: .sourceAtop)
            return true
        }
        tinted.draw(in: r, from: .zero, operation: .sourceOver, fraction: alpha)
        accents.draw(in: r, from: .zero, operation: .sourceOver, fraction: alpha)
        return true
    }
    img.accessibilityDescription = "stemflow"
    return img
}

// MARK: - Menu rows

/// [label  ───○───  value], live while dragging.
final class SliderRow: NSView {
    let label = NSTextField(labelWithString: "")
    let value = NSTextField(labelWithString: "")
    let slider = NSSlider(value: 0, minValue: 0, maxValue: 1, target: nil, action: nil)
    var onChange: (Double) -> Void = { _ in }
    var dragging = false

    init(_ text: String, onChange: @escaping (Double) -> Void) {
        super.init(frame: NSRect(x: 0, y: 0, width: 300, height: 26))
        self.onChange = onChange
        label.stringValue = text
        for l in [label, value] { l.font = .menuFont(ofSize: 0); l.textColor = .labelColor }
        value.alignment = .right
        slider.isContinuous = true
        slider.target = self
        slider.action = #selector(moved)
        let row = NSStackView(views: [label, slider, value])
        row.spacing = 8
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 14),
            row.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -14),
            row.centerYAnchor.constraint(equalTo: centerYAnchor),
            label.widthAnchor.constraint(equalToConstant: 52),
            value.widthAnchor.constraint(equalToConstant: 58),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }

    @objc func moved() {
        let ev = NSApp.currentEvent
        dragging = ev?.type == .leftMouseDragged || ev?.type == .leftMouseDown
        if ev?.type == .leftMouseUp { dragging = false }
        onChange(slider.doubleValue)
    }

    func set(_ v: Double, _ text: String) {
        if !dragging { slider.doubleValue = v }
        value.stringValue = text
    }
}

/// The snap buttons: Bass solo · Full mix · No bass · No vocals.
final class SnapRow: NSView {
    var buttons: [String: NSButton] = [:]
    init(onSnap: @escaping (String) -> Void) {
        super.init(frame: NSRect(x: 0, y: 0, width: 300, height: 30))
        let stack = NSStackView()
        stack.distribution = .fillEqually
        stack.spacing = 6
        for (label, name) in [("Bass solo", "solo"), ("Full mix", "full"), ("No bass", "nobass"), ("No vocals", "novocals")] {
            let b = ClosureButton(title: label) { onSnap(name) }
            b.bezelStyle = .rounded
            b.controlSize = .small
            stack.addArrangedSubview(b)
            buttons[name] = b
        }
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 14),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -14),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
    }
    required init?(coder: NSCoder) { fatalError() }
}

final class ClosureButton: NSButton {
    var handler: () -> Void = {}
    convenience init(title: String, handler: @escaping () -> Void) {
        self.init(frame: .zero)
        self.title = title
        self.handler = handler
        target = self
        action = #selector(fire)
    }
    @objc func fire() { handler() }
}

final class ClosureItem: NSMenuItem {
    var handler: () -> Void = {}
    convenience init(_ title: String, handler: @escaping () -> Void) {
        self.init(title: title, action: #selector(fire), keyEquivalent: "")
        self.target = self
        self.handler = handler
    }
    @objc func fire() { handler() }
}

func viewItem(_ v: NSView) -> NSMenuItem {
    let i = NSMenuItem()
    i.view = v
    return i
}

func textItem() -> NSMenuItem {
    let i = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    i.isEnabled = false
    return i
}

// MARK: - On-screen display (GNOME's OSD for hotkeys)

final class HUD {
    let panel: NSPanel
    let text = NSTextField(labelWithString: "")
    let bar = NSProgressIndicator()
    var hideWork: DispatchWorkItem?

    init() {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 240, height: 84),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.level = .statusBar
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .transient]
        let fx = NSVisualEffectView(frame: panel.contentRect(forFrameRect: panel.frame))
        fx.material = .hudWindow
        fx.state = .active
        fx.wantsLayer = true
        fx.layer?.cornerRadius = 14
        text.alignment = .center
        text.font = .systemFont(ofSize: 15, weight: .medium)
        bar.style = .bar
        bar.isIndeterminate = false
        bar.minValue = 0
        bar.maxValue = 1
        let stack = NSStackView(views: [text, bar])
        stack.orientation = .vertical
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        fx.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: fx.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: fx.trailingAnchor, constant: -20),
            stack.centerYAnchor.constraint(equalTo: fx.centerYAnchor),
        ])
        panel.contentView = fx
    }

    func show(_ message: String, level: Double?) {
        text.stringValue = message
        bar.isHidden = level == nil
        bar.doubleValue = level ?? 0
        if let s = NSScreen.main {
            let f = s.visibleFrame
            panel.setFrameOrigin(NSPoint(x: f.midX - panel.frame.width / 2, y: f.minY + 120))
        }
        panel.alphaValue = 1
        panel.orderFrontRegardless()
        hideWork?.cancel()
        let w = DispatchWorkItem { [weak self] in
            NSAnimationContext.runAnimationGroup({ $0.duration = 0.3; self?.panel.animator().alphaValue = 0 },
                                                completionHandler: { self?.panel.orderOut(nil) })
        }
        hideWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5, execute: w)
    }
}

// MARK: - Global hotkeys (Carbon: no Accessibility permission needed)

/// ⌃⌥ instead of GNOME's Super: ⌘[ / ⌘] are Back / Forward in most Mac apps.
let hotkeys: [(id: UInt32, key: Int, mods: Int, name: String)] = [
    (1, kVK_ANSI_LeftBracket, controlKey | optionKey, "slide-left"),
    (2, kVK_ANSI_RightBracket, controlKey | optionKey, "slide-right"),
    (3, kVK_ANSI_Backslash, controlKey | optionKey, "toggle-solo"),
    (4, kVK_ANSI_Backslash, controlKey | optionKey | shiftKey, "toggle-nobass"),
    (5, kVK_ANSI_B, controlKey | optionKey, "toggle-running"),
    (6, kVK_ANSI_L, controlKey | optionKey, "toggle-loop"),
]
var hotkeyRefs: [EventHotKeyRef?] = []

func registerHotkeys() {
    var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
    InstallEventHandler(GetApplicationEventTarget(), { _, event, _ -> OSStatus in
        var hk = EventHotKeyID()
        GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                          nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
        let id = hk.id
        DispatchQueue.main.async { Controller.shared.hotkey(id) }
        return noErr
    }, 1, &spec, nil, nil)
    for h in hotkeys {
        var ref: EventHotKeyRef?
        RegisterEventHotKey(UInt32(h.key), UInt32(h.mods), EventHotKeyID(signature: OSType(0x5354_464C), id: h.id),
                            GetApplicationEventTarget(), 0, &ref)
        hotkeyRefs.append(ref)
    }
}

// MARK: - The menu bar item

final class Controller: NSObject, NSMenuDelegate {
    static let shared = Controller()

    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    let menu = NSMenu()
    let hud = HUD()
    let defaults = UserDefaults.standard

    var state: [String: Any]? // nil: stemflow isn't running
    var child: Process?
    var starting: (since: Date, model: String)?
    var startError = ""
    var wantRunning = false
    var back = 0.0 // where toggle-solo / toggle-nobass return to
    var lastModelFor: [String: String] = [:]
    var pending: String? // the model asked for, until it is loaded (or fails)
    var wasLoading = false
    let queue = DispatchQueue(label: "stemflow.socket")

    // rows
    let switchItem = ClosureItem("Bass isolation") { Controller.shared.setRunning(!Controller.shared.isOnOrStarting) }
    let desc = textItem()
    var one: SliderRow!, boost: SliderRow!, bass: SliderRow!, band: SliderRow!, second: SliderRow!
    var oneItem: NSMenuItem!, boostItem: NSMenuItem!, bassItem: NSMenuItem!, bandItem: NSMenuItem!, secondItem: NSMenuItem!
    var snaps: SnapRow!, snapsItem: NSMenuItem!
    let sep1 = NSMenuItem.separator()
    let status = textItem()
    let clickItem = ClosureItem("Beat click") {
        let c = Controller.shared
        c.send("set click \((c.state?["click"] as? Bool ?? false) ? "off" : "on")")
    }
    let beatOpts = NSMenuItem(title: "Meter and feel", action: nil, keyEquivalent: "")
    var meterItems: [(Int, NSMenuItem)] = []
    var feelItems: [(Double, NSMenuItem)] = []
    let stemsItem = NSMenuItem(title: "Stem", action: nil, keyEquivalent: "")
    let modelsItem = NSMenuItem(title: "Model", action: nil, keyEquivalent: "")
    let expItem = ClosureItem("Experiments") { Controller.shared.flip("experiments") }
    let advItem = ClosureItem("Separate sliders") { Controller.shared.flip("advanced") }
    var modelsKey = ""
    // looping (state["loop"], src/loopdeck.rs; Loop.swift)
    let loopSec = LoopSection { Controller.shared.send($0) }
    var menuOpen = false
    let viewQueue = DispatchQueue(label: "stemflow.loopview")
    var viewBusy = false
    var loopState: [String: Any]? { state?["loop"] as? [String: Any] }

    var isOn: Bool { state != nil }
    var isOnOrStarting: Bool { state != nil || starting != nil }

    func setup() {
        // Keeps its place in the bar across relaunches; without it every reinstall
        // puts the icon at the far left, where the notch can hide it.
        item.autosaveName = "stemflow"
        item.button?.image = statusIcon(tint: nil)
        item.menu = menu
        menu.delegate = self
        menu.autoenablesItems = false

        one = SliderRow("Solo") { [unowned self] pos in
            var b = 2 * pos - 1
            if abs(b) < 0.04 { b = 0 } // detent at the original mix
            send("set balance \(b)")
        }
        boost = SliderRow("Boost") { [unowned self] v in send("set bass \(1 + v * (maxBass - 1))") }
        bass = SliderRow("Bass") { [unowned self] v in send("set bass \(v * maxBass)") }
        band = SliderRow("Band") { [unowned self] v in send("set band \(v)") }
        second = SliderRow("") { [unowned self] v in send("set second_level \(v)") }
        snaps = SnapRow { [unowned self] name in snap(name) }
        (oneItem, boostItem, bassItem, bandItem, secondItem, snapsItem) =
            (viewItem(one), viewItem(boost), viewItem(bass), viewItem(band), viewItem(second), viewItem(snaps))

        let meters = NSMenu()
        for m in [0, 2, 3, 4, 5, 6, 7] {
            let i = ClosureItem(m == 0 ? "Auto meter" : "\(m) beats per bar") { Controller.shared.send("set meter \(m)") }
            meters.addItem(i)
            meterItems.append((m, i))
        }
        meters.addItem(.separator())
        for (f, label) in [(1.0, "As heard"), (0.5, "Half time"), (2.0, "Double time")] {
            let i = ClosureItem(label) { Controller.shared.send("set feel \(f)") }
            meters.addItem(i)
            feelItems.append((f, i))
        }
        beatOpts.submenu = meters

        stemsItem.submenu = NSMenu()
        modelsItem.submenu = NSMenu()

        for i in [switchItem, desc, oneItem, boostItem, bassItem, secondItem, bandItem, snapsItem, sep1, status,
                  clickItem, beatOpts] {
            menu.addItem(i!)
        }
        for i in loopSec.items { menu.addItem(i) }
        menu.addItem(.separator())
        for i in [stemsItem, modelsItem, expItem, advItem] { menu.addItem(i) }
        menu.addItem(.separator())
        menu.addItem(ClosureItem("Open log") { NSWorkspace.shared.open(logURL) })
        menu.addItem(ClosureItem("Quit") { Controller.shared.quit() })

        // Scrolling on the icon moves the mix.
        NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { ev in
            guard ev.window == self.item.button?.window else { return ev }
            let d = ev.scrollingDeltaY + ev.scrollingDeltaX
            if abs(d) > 0.5 { self.nudge(d > 0 ? step / 2 : -step / 2) }
            return nil
        }

        let t = Timer(timeInterval: 1, repeats: true) { _ in self.poll() }
        RunLoop.main.add(t, forMode: .common) // keeps ticking while the menu is open
        // The loop view moves with the playhead while the menu shows it.
        let v = Timer(timeInterval: 0.05, repeats: true) { _ in self.pollView() }
        RunLoop.main.add(v, forMode: .common)
        registerHotkeys()
        poll()
    }

    func flip(_ key: String) {
        defaults.set(!defaults.bool(forKey: key), forKey: key)
        sync()
    }

    // MARK: talking to stemflow

    func poll() {
        queue.async {
            let s = request("state")
            DispatchQueue.main.async { self.got(s) }
        }
    }

    func pollView() {
        guard menuOpen, loopSec.hasLoop, !viewBusy else { return }
        viewBusy = true
        let bins = Int(loopSec.view.bounds.width / 2)
        viewQueue.async {
            let j = request("loop view \(bins)")
            DispatchQueue.main.async {
                self.viewBusy = false
                self.loopSec.view.pic = LoopPic(j)
            }
        }
    }

    func send(_ line: String, then: (() -> Void)? = nil) {
        queue.async {
            let s = request(line)
            DispatchQueue.main.async {
                if let e = s?["error"] as? String { self.hud.show(e, level: nil) }
                self.got(s)
                then?()
            }
        }
    }

    func got(_ s: [String: Any]?) {
        let was = state != nil
        state = s
        if s != nil && starting != nil { stopStarting("") }
        if s == nil && was { modelsKey = "" }
        sync()
    }

    func loop(_ cmd: String) { send("loop \(cmd)") }

    /// ⌃⌥L: loop the last 4 bars, or back to live if a loop plays (or is being found).
    func toggleLoop() {
        guard let l = loopState else { return hud.show(isOn ? "No looping: no beat model" : "stemflow is off", level: nil) }
        if l["playing"] as? Bool ?? false || l["finding"] as? Bool ?? false {
            loop("live")
            hud.show("Live", level: nil)
        } else {
            loop("4")
            hud.show("Looping the last 4 bars", level: nil)
        }
    }

    func snap(_ name: String) { send("snap \(name)") { self.osd() } }

    func nudge(_ d: Double) {
        guard isOn else { return osdOff() }
        send("nudge \(d)") { self.osd() }
    }

    /// Jump to a snap point, or back to where you were if already there.
    func toggle(_ name: String, at: Double) {
        guard let b = state?["balance"] as? Double else { return osdOff() }
        if abs(b - at) < 1e-3 {
            send("set balance \(back)") { self.osd() }
        } else {
            back = b
            snap(name)
        }
    }

    func hotkey(_ id: UInt32) {
        switch hotkeys.first(where: { $0.id == id })?.name {
        case "slide-left": nudge(-step)
        case "slide-right": nudge(step)
        case "toggle-solo": toggle("solo", at: -1)
        case "toggle-nobass": toggle("nobass", at: 1)
        case "toggle-running": setRunning(!isOnOrStarting)
        case "toggle-loop": toggleLoop()
        default: break
        }
    }

    func osd() {
        guard let s = state else { return }
        hud.show(s["description"] as? String ?? "", level: ((s["balance"] as? Double ?? 0) + 1) / 2)
    }

    func osdOff() { hud.show("stemflow is off", level: nil) }

    // MARK: starting and stopping (the systemd unit's job on Linux)

    func setRunning(_ on: Bool) {
        wantRunning = on
        if !on {
            stopStarting("")
            if let c = child, c.isRunning {
                c.interrupt() // SIGINT: stemflow hands the app back to the speakers
            } else if isOn {
                send("quit") // started elsewhere (a terminal)
            }
            return
        }
        if isOnOrStarting { return }
        startError = ""
        let exe = root.appendingPathComponent("target/release/stemflow")
        guard FileManager.default.isExecutableFile(atPath: exe.path) else {
            startError = "No \(exe.path) (cargo build --release)"
            return sync()
        }
        let p = Process()
        p.executableURL = exe
        p.currentDirectoryURL = root
        p.arguments = ["live", "--models", root.appendingPathComponent("models").path] + extraArgs()
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        if let log = try? FileHandle(forWritingTo: logURL) {
            p.standardOutput = log
            p.standardError = log
        }
        p.terminationHandler = { proc in
            DispatchQueue.main.async { self.exited(proc) }
        }
        do {
            try p.run()
        } catch {
            startError = "Couldn't start: \(error.localizedDescription)"
            return sync()
        }
        child = p
        starting = (Date(), lastModel())
        sync()
    }

    func exited(_ p: Process) {
        guard p === child else { return }
        child = nil
        let failed = p.terminationStatus != 0 && p.terminationReason != .uncaughtSignal
        if starting != nil {
            stopStarting("stemflow stopped while starting (\(lastLogLine()))")
        } else if wantRunning && failed {
            // Like the unit's Restart=on-failure.
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                if self.wantRunning && self.child == nil { self.setRunning(true) }
            }
        }
        poll()
    }

    func lastLogLine() -> String {
        let text = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
        return text.split(separator: "\n").last.map(String.init) ?? "see the log"
    }

    func stopStarting(_ err: String) {
        starting = nil
        startError = err
        sync()
    }

    func quit() {
        if let c = child, c.isRunning {
            c.interrupt()
            c.waitUntilExit()
        }
        NSApp.terminate(nil)
    }

    // MARK: menu contents

    func menuWillOpen(_ menu: NSMenu) {
        menuOpen = true
        poll()
    }

    func menuDidClose(_ menu: NSMenu) { menuOpen = false }

    func fillModels(_ models: [[String]], current: String) {
        if !current.isEmpty { lastModelFor[stemOf(current)] = current }
        let loading = (state?["status"] as? String ?? "").hasPrefix("Loading")
        let shown = loading && pending != nil ? pending! : current
        let key = "\(models)|\(shown)|\(loading)"
        if key == modelsKey { return }
        modelsKey = key
        let stem = stemOf(shown)
        let tail = loading ? " · loading…" : ""

        let sm = stemsItem.submenu!
        sm.removeAllItems()
        for s in stems where models.contains(where: { stemOf($0[0]) == s }) {
            let i = ClosureItem(stemLabels[s]!) { if s != stem { Controller.shared.setStem(s, models) } }
            i.state = s == stem ? .on : .off
            sm.addItem(i)
        }
        stemsItem.title = "Stem: \(stemLabels[stem] ?? stem)\(tail)"

        let mm = modelsItem.submenu!
        mm.removeAllItems()
        for m in models where stemOf(m[0]) == stem {
            let id = m[0]
            let i = ClosureItem(m[1]) { Controller.shared.setModel(id) }
            i.state = id == shown ? .on : .off
            mm.addItem(i)
        }
        let label = models.first { $0[0] == shown }?[1] ?? shown
        modelsItem.title = "Model: \(label)\(tail)"
    }

    func setStem(_ stem: String, _ models: [[String]]) {
        let known = { (id: String?) in id != nil && models.contains { $0[0] == id } }
        let id = known(lastModelFor[stem]) ? lastModelFor[stem]! : models.first { stemOf($0[0]) == stem }?[0]
        if let id { setModel(id) } else { hud.show("No \(stem) model in the models folder", level: nil) }
    }

    func setModel(_ id: String) {
        pending = id
        send("set model \(id)")
    }

    func sync() {
        let on = isOn
        let adv = defaults.bool(forKey: "advanced")
        let exp = defaults.bool(forKey: "experiments")
        switchItem.state = isOnOrStarting ? .on : .off
        expItem.state = exp ? .on : .off
        advItem.state = adv ? .on : .off
        let rows: [NSMenuItem] = [desc, oneItem, boostItem, bassItem, bandItem, secondItem, snapsItem, sep1, status,
                                  modelsItem, stemsItem, clickItem, beatOpts]
        for r in rows { r.isHidden = !on }
        if !on { loopSec.sync(nil, meter: 0, feel: 1) }

        let button = item.button
        guard let s = state else {
            button?.image = starting != nil ? statusIcon(tint: clefOn, alpha: 0.5) : statusIcon(tint: nil)
            switchItem.title = "Bass isolation"
            desc.isHidden = starting == nil && startError.isEmpty
            if let st = starting {
                let secs = Int(Date().timeIntervalSince(st.since))
                desc.title = "Loading \(st.model.isEmpty ? "the model" : st.model)… \(secs) s"
            } else {
                desc.title = startError
            }
            return
        }
        let over = s["overruns"] as? Int ?? 0
        button?.image = statusIcon(tint: over > 0 ? clefWarn : clefOn)

        let secondStem = s["second"] as? String ?? ""
        let three = !secondStem.isEmpty
        oneItem.isHidden = adv || three
        boostItem.isHidden = adv || three
        bassItem.isHidden = !(adv || three)
        bandItem.isHidden = !(adv || three)
        secondItem.isHidden = !three
        snaps.buttons["novocals"]?.isHidden = !three

        let t = s["target"] as? String ?? "bass"
        let T = cap(t)
        switchItem.title = three ? "\(T) + \(secondStem) isolation" : "\(T) isolation\(t == "vocals" ? " (karaoke)" : "")"
        bass.label.stringValue = T
        snaps.buttons["solo"]?.title = "\(T) solo"
        snaps.buttons["nobass"]?.title = "No \(t)"
        let statusText = s["status"] as? String ?? ""
        desc.title = statusText.isEmpty ? (s["description"] as? String ?? "") : statusText

        let balance = s["balance"] as? Double ?? 0
        let b = s["bass"] as? Double ?? 1
        let bd = s["band"] as? Double ?? 1
        one.set((balance + 1) / 2, "No \(t)")
        boost.set(max(0, b - 1) / (maxBass - 1), b > 1 ? "+\(Int(((b - 1) * 100).rounded()))%" : "off")
        bass.set(b / maxBass, "\(Int((b * 100).rounded()))%")
        band.set(bd, "\(Int((bd * 100).rounded()))%")
        if three {
            second.label.stringValue = cap(secondStem)
            let lv = s["second_level"] as? Double ?? 1
            second.set(lv, "\(Int((lv * 100).rounded()))%")
            snaps.buttons["novocals"]?.title = "No \(secondStem)"
        }

        let app = s["app"] as? String ?? ""
        let lat = s["latency"] as? Double ?? 0, comp = s["compute"] as? Double ?? 0, hop = s["hop"] as? Double ?? 0
        status.title = String(format: "%@ · %.1f s behind · model %.2f s of %.1f s", app, lat, comp, hop)
            + (over > 0 ? " · \(over) dropouts" : "")

        // No beat tracker model: no click (as the extension hides it for old daemons).
        let tracker = s["beat_tracker"] as? Bool ?? false
        clickItem.isHidden = !tracker
        beatOpts.isHidden = !tracker
        let click = s["click"] as? Bool ?? false
        let beats = s["beats"] as? String ?? ""
        clickItem.state = click ? .on : .off
        clickItem.title = click && !beats.isEmpty ? "Beat click · \(beats)" : "Beat click"
        let meter = s["meter"] as? Int ?? 0
        let feel = s["feel"] as? Double ?? 1
        for (m, i) in meterItems { i.state = m == meter ? .on : .off }
        for (f, i) in feelItems { i.state = f == feel ? .on : .off }
        beatOpts.title = "Meter and feel: \(meter > 0 ? "\(meter)/bar" : "auto meter")"
            + (feel == 0.5 ? " · half time" : feel == 2 ? " · double time" : "")

        loopSec.sync(loopState, meter: meter, feel: feel)

        let loading = statusText.hasPrefix("Loading")
        if wasLoading && !loading { pending = nil }
        wasLoading = loading
        let model = s["model"] as? String ?? ""
        let models = (s["models"] as? [[String]]) ?? []
        stemsItem.isHidden = model.isEmpty
        modelsItem.isHidden = !exp || model.isEmpty
        if !model.isEmpty { fillModels(models, current: model) }
    }
}

// MARK: - main

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
Controller.shared.setup()
signal(SIGINT) { _ in DispatchQueue.main.async { Controller.shared.quit() } }
app.run()
