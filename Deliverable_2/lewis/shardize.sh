#!/bin/bash
FILE=$1
NUM_SHARDS=$2
OUT_DIR=$3

mkdir -p $OUT_DIR

LINES_PER_SHARD=$(($(wc -l $FILE | cut -d " " -f1) / $NUM_SHARDS))

split -l $LINES_PER_SHARD -a 3 -d $FILE  $OUT_DIR/$(basename $FILE)_sharded.

for i in $OUT_DIR/*; do
    mv $i $(echo $i | sed 's/\.00/\./g'| sed 's/\.0/\./g') 2> /dev/null &
done
wait

mv $OUT_DIR/$(basename $FILE)_sharded. $OUT_DIR/$(basename $FILE)_sharded.0