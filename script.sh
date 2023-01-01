#!/usr/bin/env bash
echo "Preprocessing"
python3 src/preproc.py $1

echo "Optimizing trajectory"
python3 src/stabilize.py $2

echo "Generating output"
python3 src/generate.py $1 $2
