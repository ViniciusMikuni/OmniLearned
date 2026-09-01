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
    --wandb
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




# NEVT = 1000
# run_experiment 1000 fine_tune_quijote_baseline_s   quijote pretrain_s          regression   100  1e-5 0.1 10.0 62 -1 --k 20 
run_experiment 1000 fine_tune_quijote_gaussian_s   quijote pretrain_s_gaussian regression   200  1e-5 0.1 2.0 62 100  --freeze --k 20
#run_experiment 1000 fine_tune_quijote_freeze_s     quijote pretrain_s          regression   80  1e-5 1.0 10.0 62 -1 --freeze --k 20

# run_experiment 1000 fine_tune_v_quijote_baseline_s quijote pretrain_s          segmentation 200  1e-5 1.0 20.0 62 -1 --num-gen-classes 3 --k 20 --resuming
#run_experiment 1000 fine_tune_v_quijote_gaussian_s quijote pretrain_s_gaussian segmentation 200  1e-5 1.0 20.0 62 -1 --num-gen-classes 3 --freeze --k 20
#run_experiment 1000 fine_tune_v_quijote_freeze_s   quijote pretrain_s          segmentation 200  1e-5 1.0 20.0 62 -1 --num-gen-classes 3 --freeze --k 20

# # NEVT = 19651
# run_experiment 19651 fine_tune_quijote_baseline_s   quijote pretrain_s          regression   200  1e-5 0.1 10.0 -1 -1 --k 20
# run_experiment 19651 fine_tune_v_quijote_baseline_s quijote pretrain_s          segmentation 300  1e-5 1.0 20.0 -1 -1 --num-gen-classes 3 --k 20

# run_experiment 19651 fine_tune_quijote_gaussian_s   quijote pretrain_s_gaussian regression   200  1e-5 0.1 10.0 -1 -1 --k 20
# run_experiment 19651 fine_tune_quijote_freeze_s     quijote pretrain_s          regression   200  1e-5 0.1 10.0 -1 -1 --freeze
# run_experiment 19651 fine_tune_v_quijote_gaussian_s quijote pretrain_s_gaussian segmentation 300  1e-5 1.0 20.0 -1 -1 --num-gen-classes 3 --k 20
# run_experiment 19651 fine_tune_v_quijote_freeze_s   quijote pretrain_s          segmentation 300  1e-5 1.0 20.0 -1 -1 --num-gen-classes 3 --freeze
