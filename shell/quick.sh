#!/bin/bash
set -euo pipefail

for N in {0..9}; do
    src=/pscratch/sd/v/vmikuni/datasets/qcd_dijet/train/train_$N/*.h5
    dest="/pscratch/sd/v/vmikuni/datasets/qcd_dijet_$N/train"

    mv $src "$dest"
done

echo "All folders moved."
