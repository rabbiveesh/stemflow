#!/usr/bin/env bash
# Build the stemflow menu bar app (the macOS counterpart of gnome/install.sh) into
# ~/Applications/Stemflow.app. Needs the Xcode command line tools (swiftc) and a
# release build of stemflow: cargo build --release
#   macos/install.sh           build and open it
#   macos/install.sh --login   also start it at login
# Re-run after moving the checkout or changing macos/StemflowBar.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
app="${STEMFLOW_APP:-$HOME/Applications/Stemflow.app}"

pkill -x Stemflow 2>/dev/null || true
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
# The GNOME extension's icon: the clef (recoloured by the app) and its accents.
cp "$root/gnome/stemflow@veesh/icons/"stemflow-{clef-symbolic,accents}.svg "$app/Contents/Resources/"
swiftc -O -o "$app/Contents/MacOS/Stemflow" "$here"/StemflowBar/*.swift
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key><string>Stemflow</string>
	<key>CFBundleIdentifier</key><string>org.stemflow.menubar</string>
	<key>CFBundleName</key><string>Stemflow</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleShortVersionString</key><string>0.1</string>
	<key>LSMinimumSystemVersion</key><string>14.2</string>
	<key>LSUIElement</key><true/>
	<key>NSAudioCaptureUsageDescription</key>
	<string>stemflow listens to the app you pick (Spotify by default) to separate its stems as it plays.</string>
	<key>NSAppleEventsUsageDescription</key>
	<string>stemflow pauses Spotify or Music while a loop plays, so going live again carries on from the loop.</string>
	<key>StemflowRoot</key><string>$root</string>
</dict>
</plist>
EOF
codesign --force --sign - "$app"

if [[ "${1:-}" == --login ]]; then
  osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$app\", hidden:true}" >/dev/null
  echo "added to login items"
fi
open "$app"
echo "installed $app (the clef icon in the menu bar)"
