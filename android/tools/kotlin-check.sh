#!/usr/bin/env bash
# Compile-check the app's Kotlin without the Android SDK: kotlinc (from Maven Central)
# against Robolectric's android-all (the framework classes, API 34) and Shizuku's
# API, with a stub R. Catches what CI otherwise can't, such as LoopPanel.kt's
# `when` over the loop panel's kinds missing one. Jars are kept in $KT_CACHE.
#   android/tools/kotlin-check.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
src=$root/android/app/src/main
cache=${KT_CACHE:-$HOME/.cache/stemflow-kotlin-check}
mkdir -p "$cache"
m=https://repo1.maven.org/maven2
get() { # path in Maven Central
  local f=$cache/$(basename "$1")
  if [ ! -s "$f" ]; then
    curl -fsSL --retry 5 --retry-all-errors -o "$f.part" "$m/$1" ||
      { rm -f "$f.part"; echo "couldn't fetch $1" >&2; return 1; }
    mv "$f.part" "$f"
  fi
  echo "$f"
}
kc=$(get org/jetbrains/kotlin/kotlin-compiler-embeddable/2.0.21/kotlin-compiler-embeddable-2.0.21.jar)
std=$(get org/jetbrains/kotlin/kotlin-stdlib/2.0.21/kotlin-stdlib-2.0.21.jar)
tool=$kc:$std
for p in org/jetbrains/kotlin/kotlin-script-runtime/2.0.21/kotlin-script-runtime-2.0.21.jar \
  org/jetbrains/kotlin/kotlin-reflect/1.6.10/kotlin-reflect-1.6.10.jar \
  org/jetbrains/kotlin/kotlin-daemon-embeddable/2.0.21/kotlin-daemon-embeddable-2.0.21.jar \
  org/jetbrains/intellij/deps/trove4j/1.0.20200330/trove4j-1.0.20200330.jar \
  org/jetbrains/annotations/13.0/annotations-13.0.jar \
  org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.6.4/kotlinx-coroutines-core-jvm-1.6.4.jar; do
  tool=$tool:$(get $p)
done
cp=$(get org/robolectric/android-all/14-robolectric-10818077/android-all-14-robolectric-10818077.jar):$std
for n in api provider aidl; do
  jar=$cache/shizuku-$n.jar
  if [ ! -s "$jar" ]; then
    aar=$(get dev/rikka/shizuku/$n/13.1.5/$n-13.1.5.aar)
    (cd "$cache" && unzip -qo "$aar" classes.jar && mv classes.jar "$jar")
  fi
  cp=$cp:$jar
done
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
# R: the resource names the code uses, all 0
{
  echo "package org.stemflow.app"
  echo "object R {"
  for t in drawable mipmap; do
    echo "  object $t {"
    for f in $(cd "$src/res" && ls -d $t* | xargs -I{} ls {} | sed 's/\..*//' | sort -u); do echo "    const val $f = 0"; done
    echo "  }"
  done
  echo "}"
} > "$out/R.kt"
log=$out/log
java -Xmx2g -cp "$tool" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect \
  -jvm-target 17 -cp "$cp" -d "$out/classes" "$src/kotlin" "$src/java" "$out/R.kt" >"$log" 2>&1 || true
grep -v '^warning: \|^Picked up ' "$log" || true
! grep -q 'error:' "$log"
