#!/bin/bash
# poor man's profiler: SAMPLES snapshots of every thread's stack, every PERIOD seconds
PID=$1; SAMPLES=$2; PERIOD=$3; OUT=$4
: > $OUT
for i in $(seq 1 $SAMPLES); do
  gdb -p $PID -batch -nx -ex "set pagination 0" -ex "thread apply all bt 40" 2>/dev/null | grep -E "^#|^Thread" >> $OUT
  echo "=== sample $i" >> $OUT
  sleep $PERIOD
done
