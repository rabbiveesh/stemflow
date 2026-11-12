#!/usr/bin/env bash
# Install the stemflow systemd user service and GNOME Shell extension (GNOME 42).
# Re-run after moving the checkout. Needs a release build: cargo build --release --features cuda
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
units=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
exts=${XDG_DATA_HOME:-$HOME/.local/share}/gnome-shell/extensions

mkdir -p "$units" "$exts"
sed "s|@ROOT@|$root|g" "$here/stemflow.service.in" > "$units/stemflow.service"
systemctl --user daemon-reload

glib-compile-schemas "$here/stemflow@veesh/schemas"
ln -sfn "$here/stemflow@veesh" "$exts/stemflow@veesh"
gnome-extensions enable stemflow@veesh 2>/dev/null || true
echo "installed. On X11 reload the shell (Alt+F2, r); on Wayland log out and back in."
echo "then: gnome-extensions enable stemflow@veesh"
