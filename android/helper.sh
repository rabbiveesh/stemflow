#!/usr/bin/env bash
# Start stemflow's capture helper on the phone as the shell user (development stand-in
# for Shizuku). Switch "Bass isolation" on in the app first; this runs until you stop
# it (Ctrl-C) or switch isolation off. Captures all media playback except stemflow's.
#   android/helper.sh
set -euo pipefail
pkg=org.stemflow.app
uid="$(adb shell pm list packages -U "$pkg" | sed -n "s/^package:$pkg uid://p" | tr -d '\r')"
apk="$(adb shell pm path "$pkg" | cut -d: -f2 | tr -d '\r')"
[[ -n "$uid" && -n "$apk" ]] || { echo "the stemflow app is not installed on the phone" >&2; exit 1; }
exec adb shell "CLASSPATH=$apk app_process / org.stemflow.app.Helper 28711 $uid"
