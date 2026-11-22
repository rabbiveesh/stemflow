#!/system/bin/sh
# Notification shade opened and closed continuously (its blur is GPU-heavy), while the
# bench runs paced; GPU priority alternates default / high in 3-minute blocks (D1 H1 D2
# H2), so both see the same heat. Push next to stemflow-bench and run detached, as
# stress.sh. Writes bench_shade_*.log, freq_shade_*.log, thermal_shade.log, then
# touches `shadedone`. It uses `cmd statusbar`, so nothing on screen is touched.
cd /data/local/tmp/stemflow-speed
export LD_LIBRARY_PATH=.
M=/sys/devices/platform/1f000000.mali/cur_freq
rm -f shadedone
( n=0; while :; do cmd statusbar expand-notifications; sleep 0.7; cmd statusbar collapse; sleep 0.7; n=$((n+1)); [ $((n % 6)) = 0 ] && input keyevent 59; done ) &
L=$!
( while :; do echo "$(date +%s) $(dumpsys thermalservice | grep -E 'mName=(VIRTUAL-SKIN|soc_therm),' | sed 's/.*mValue=\([0-9.]*\).*mName=\([^,]*\), mStatus=\([0-9]\).*/\2=\1:\3/' | tr '\n' ' ')"; sleep 20; done ) > thermal_shade.log &
T=$!
for seg in D1 H1 D2 H2; do
  case $seg in D*) opts='' ;; H*) opts='priority = 3' ;; esac
  ( while :; do echo "$(date +%s) $(cat $M)"; sleep 0.2; done ) > freq_shade_$seg.log &
  F=$!
  STEMFLOW_GPU_OPTIONS="$opts" ./stemflow-bench ft2_bass_344064.tflite --gpu fp16 --wav clip.wav --hop 1 --pace --repeat 6 > bench_shade_$seg.log 2>&1
  kill -9 $F 2>/dev/null
done
kill -9 $L $T 2>/dev/null
cmd statusbar collapse
touch shadedone
