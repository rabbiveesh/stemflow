#!/system/bin/sh
# Stress test on the phone: paced live-rate runs of the bench, with the GPU clock (5 Hz)
# and temperatures (every 20 s) logged, and optionally Settings scrolling in front as a
# screen load. Push it next to stemflow-bench (see ../README.md) and run it detached:
#   ./stress.sh phase NAME 'GPU OPTIONS' LOAD(0|1) REPEAT MODEL HOP LOOKAHEAD
#   e.g. (setsid nohup ./stress.sh phase busy 'priority = 3' 1 29 ft2_bass_344064.tflite 1 1 &)
# Writes bench_NAME.log, freq_NAME.log and thermal.log, then touches `done`.
# The screen must be on and unlocked for LOAD=1; nothing else should use the GPU.
cd /data/local/tmp/stemflow-speed
export LD_LIBRARY_PATH=.
M=/sys/devices/platform/1f000000.mali/cur_freq
rm -f done
phase() {
  name=$1; opts=$2; load=$3; rep=$4; model=$5; hop=$6; la=$7
  echo "== $name $(date +%s)" >> thermal.log
  ( while :; do echo "$(date +%s) $(cat $M)"; sleep 0.2; done ) > freq_$name.log &
  F=$!
  ( while :; do echo "$(date +%s) $(dumpsys thermalservice | grep -E 'mName=(VIRTUAL-SKIN|soc_therm|battery),' | sed 's/.*mValue=\([0-9.]*\).*mName=\([^,]*\), mStatus=\([0-9]\).*/\2=\1:\3/' | tr '\n' ' ')"; sleep 20; done ) >> thermal.log &
  T=$!
  L=
  if [ "$load" = 1 ]; then
    am start -a android.settings.SETTINGS >/dev/null 2>&1
    ( while :; do input swipe 540 1700 540 600 250; input swipe 540 600 540 1700 250; done ) &
    L=$!
  fi
  STEMFLOW_GPU_OPTIONS="$opts" ./stemflow-bench $model --gpu fp16 --wav clip.wav --hop $hop --lookahead $la --pace --repeat $rep > bench_$name.log 2>&1
  kill $F $T $L 2>/dev/null
  [ "$load" = 1 ] && input keyevent HOME
}
"$@"
touch done
