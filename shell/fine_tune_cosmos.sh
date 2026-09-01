#!/bin/bash
set -euo pipefail
set -x

module load pytorch/2.8.0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR="$(hostname)"

BATCH=2

# Arguments shared by every experiment
COMMON_ARGS=(
    -o /pscratch/sd/v/vmikuni/PET/checkpoints/
    --size small
    --num-classes 2
    --num-feat 3
    --batch "$BATCH"
    --fine-tune
    --interaction-type astro
    --local-interaction
    --num-coord 3
)

run_experiment() {
    local nevt="$1"
    local save_tag="$2"
    local dataset="$3"
    local pretrain_tag="$4"
    local mode="$5"
    local epochs="$6"
    local lr="$7"
    local wd="$8"
    local lr_factor="$9"
    local train_nevts="${10}"
    local iterations="${11}"

    shift 11
    local -a extra_args=("$@")

    local -a cmd=(
        omnilearned train
        "${COMMON_ARGS[@]}"
        --save-tag "${save_tag}_${nevt}"
        --dataset "$dataset"
        --mode "$mode"
        --epoch "$epochs"
        --lr "$lr"
        --wd "$wd"
        --lr-factor "$lr_factor"
    )

    [[ -n "$pretrain_tag" ]] &&
        cmd+=(--pretrain-tag "$pretrain_tag")

    [[ -n "$train_nevts" ]] &&
        cmd+=(--nevts "$train_nevts")

    [[ -n "$iterations" ]] &&
        cmd+=(--iterations "$iterations")

    cmd+=("${extra_args[@]}")

    srun -l -u bash -c '
        source export_ddp.sh
        exec "$@"
    ' bash "${cmd[@]}"
}


# Arguments:
# NEVT  SAVE_TAG  DATASET  PRETRAIN_TAG  MODE
# EPOCHS  LR  WD  LR_FACTOR  --nevts  --iterations  EXTRA_ARGS

# NEVT = 100
run_experiment 100 fine_tune_quijote_freeze_s   quijote pretrain_s regression   20 5e-6 0.1 2  6 60  --freeze
run_experiment 100 fine_tune_quijote_s          quijote pretrain_s regression   20 5e-6 0.1 2  6 60
run_experiment 100 fine_tune_v_quijote_freeze_s quijote pretrain_s segmentation 30 1e-5 1.0 10 6 100 --num-gen-classes 3 --freeze
run_experiment 100 fine_tune_v_quijote_s        quijote pretrain_s segmentation 30 1e-5 1.0 10 6 100 --num-gen-classes 3

# NEVT = 1000
run_experiment 1000 fine_tune_quijote_freeze_s   quijote pretrain_s regression   100 1e-5 0.0 10 62 "" --freeze
run_experiment 1000 fine_tune_quijote_s          quijote pretrain_s regression   100 1e-5 0.0 10 62 ""
run_experiment 1000 fine_tune_v_quijote_freeze_s quijote pretrain_s segmentation 300 1e-5 0.0 20 62 "" --num-gen-classes 3 --freeze
run_experiment 1000 fine_tune_v_quijote_s        quijote pretrain_s segmentation 300 1e-5 0.0 20 62 "" --num-gen-classes 3

# Full dataset: empty strings omit --nevts and --iterations
run_experiment 19651 fine_tune_quijote_freeze_s   quijote pretrain_s regression   100 1e-5 0.0 10 "" "" --freeze
run_experiment 19651 fine_tune_quijote_s          quijote pretrain_s regression   100 1e-5 0.0 10 "" ""
run_experiment 19651 fine_tune_v_quijote_freeze_s quijote pretrain_s segmentation 300 1e-5 0.0 20 "" "" --num-gen-classes 3 --freeze
run_experiment 19651 fine_tune_v_quijote_s        quijote pretrain_s segmentation 300 1e-5 0.0 20 "" "" --num-gen-classes 3

BATCH=4

# NEVT = 100
run_experiment 100 fine_tune_camels_gaussian_s   camels pretrain_s_gaussian regression   300 1e-5 1.0 20.0 6 100
run_experiment 100 fine_tune_camels_freeze_s     camels pretrain_s          regression   300 1e-5 1.0 20.0 6 100 --freeze
run_experiment 100 fine_tune_v_camels_gaussian_s camels pretrain_s_gaussian segmentation 20  1e-5 5.0 20.0 6 100 --num-gen-classes 3
run_experiment 100 fine_tune_v_camels_freeze_s   camels pretrain_s          segmentation 20  1e-5 5.0 20.0 6 100 --num-gen-classes 3 --freeze

# NEVT = 300
run_experiment 300 fine_tune_camels_gaussian_s   camels pretrain_s_gaussian regression   300 1e-5 1.0 20.0 19 100
run_experiment 300 fine_tune_camels_freeze_s     camels pretrain_s          regression   300 1e-5 1.0 20.0 19 100 --freeze
run_experiment 300 fine_tune_v_camels_gaussian_s camels pretrain_s_gaussian segmentation 20  1e-5 5.0 20.0 19 100 --num-gen-classes 3
run_experiment 300 fine_tune_v_camels_freeze_s   camels pretrain_s          segmentation 20  1e-5 5.0 20.0 19 100 --num-gen-classes 3 --freeze

# NEVT = 600
run_experiment 600 fine_tune_camels_gaussian_s   camels pretrain_s_gaussian regression   300 1e-5 1.0 20.0 -1 100
run_experiment 600 fine_tune_camels_freeze_s     camels pretrain_s          regression   300 1e-5 1.0 20.0 -1 100 --freeze
run_experiment 600 fine_tune_v_camels_gaussian_s camels pretrain_s_gaussian segmentation 20  1e-5 5.0 20.0 -1 100 --num-gen-classes 3
run_experiment 600 fine_tune_v_camels_freeze_s   camels pretrain_s          segmentation 20  1e-5 5.0 20.0 -1 100 --num-gen-classes 3 --freeze
