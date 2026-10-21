#!/bin/bash
# Run inside an existing Slurm allocation.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 config_file.yaml [training arguments...]" >&2
    exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$1
shift
# Accept a path relative to the current directory, or a filename in configs/.
if [[ ! -f "$CONFIG" ]]; then
    CONFIG="$ROOT/configs/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi
CONFIG=$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")

module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

srun -l -u --cpu-bind=cores --gpu-bind=none bash -c '
    export RANK=$SLURM_PROCID
    export WORLD_RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    exec "$@"
' bash omnilearned evaluate --config "$CONFIG" "$@"
