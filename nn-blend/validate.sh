#!/bin/bash
for file in nn-blend/blend_*.out; do
    ./bin/noise "$file"
    python upload.py "${file}_noisy.txt" 2>> nn-blend/notes.txt
done
