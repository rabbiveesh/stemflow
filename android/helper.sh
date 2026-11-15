#!/usr/bin/env bash
# Start stemflow's capture helper on the phone as the shell user (development stand-in
# for Shizuku). Switch "Bass isolation" on in the app first; this runs until you stop
# it (Ctrl-C) or switch isolation off.
#   android/helper.sh             all media playback except stemflow's own
#   android/helper.sh PACKAGE     only that app's playback, e.g. com.spotify.music
set -euo pipefail
uid_of() { adb shell pm list packages -U "$1" | sed -n "s/^package:$1 uid://p" | tr -d '\r'; }
apk="$(adb shell pm path org.stemflow.app | cut -d: -f2 | tr -d '\r')"
[[ -n "$apk" ]] || { echo "the stemflow app is not installed on the phone" >&2; exit 1; }
if [[ $# -gt 0 ]]; then mode=only pkg="$1"; else mode=except pkg=org.stemflow.app; fi
uid="$(uid_of "$pkg")"
[[ -n "$uid" ]] || { echo "$pkg is not installed on the phone" >&2; exit 1; }
exec adb shell "CLASSPATH=$apk app_process / org.stemflow.app.Helper 28711 $mode $uid"
