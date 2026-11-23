// Generated from src/loopdeck/panel.rs by `cargo test`: don't edit.
// The kinds of item in the loop panel. Loop.swift draws each one; its switch
// over them is exhaustive, so a new kind doesn't build until it's drawn.

enum PanelKind: String, CaseIterable {
    case text
    case button
    case `switch`
    case choice
    case nudge
    case slider
    case loopView = "loop_view"
    case bookmarks
}
