// Generated from src/loopdeck/panel.rs by `cargo test`: don't edit.
// The kinds of item in the loop panel. LoopPanel.kt draws each one in a `when`
// over them with no `else`, so a new kind doesn't build until it's drawn.

package org.stemflow.app

enum class PanelKind(val wire: String) {
    TEXT("text"),
    BUTTON("button"),
    SWITCH("switch"),
    CHOICE("choice"),
    NUDGE("nudge"),
    SLIDER("slider"),
    LOOP_VIEW("loop_view"),
    BOOKMARKS("bookmarks");

    companion object {
        fun of(s: String): PanelKind? = entries.firstOrNull { it.wire == s }
    }
}
