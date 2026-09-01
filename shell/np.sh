# load libs
module load pytorch/2.8

# for DDP
export MASTER_ADDR=$(hostname)



#!/bin/bash

DATASETS=("jetht_dijet" "top_dijet" "top0_dijet" "top1_dijet" "top2_dijet" "stopw_dijet" "stop_dijet" "wjets_dijet" "zjets_dijet" "dihiggs_dijet" "ww_dijet" "wz_dijet" "zz_dijet")
DATASETS=("atlas_ad")
DATASETS=("jetht_dijet")
# "higgs_dijet" "topw_dijet" "topz_dijet"
#DATASETS=("qcd600_dijet")
#DATASETS=("aspen_small")
#DATASETS=("qcd_dijet_0" "qcd_dijet_1" "qcd_dijet_2" "qcd_dijet_3" "qcd_dijet_4" "qcd_dijet_5" "qcd_dijet_6" "qcd_dijet_7" "qcd_dijet_8" "qcd_dijet_9")
#"qcd_dijet_1" "qcd_dijet_2" "qcd_dijet_3" "qcd_dijet_4" "qcd_dijet_5" "qcd_dijet_6" "qcd_dijet_7" "qcd_dijet_8" "qcd_dijet_9")
#DATASETS=("jetht2017_dijet")
#DATASETS=("top0_dijet" "top1_dijet" "top2_dijet" "top3_dijet")
#DATASETS=("top_dijet")
#DATASETS=("singlemuon_cr" "stop_cr" "stopw_cr" "topw_cr" "wjets_cr" "top0_cr" "top1_cr" "top2_cr" "top3_cr")
#"top0_cr" "top1_cr" "top2_cr" "stop_cr" "stopw_cr" "topw_cr" "wjets_cr")
#DATASETS=("top0_cr" "top1_cr" "top2_cr" "top3_cr")
#DATASETS=("topw_cr")
FAILED=()

for ds in "${DATASETS[@]}"; do
    #omnilearned dataloader --dataset ${ds} --folder /pscratch/sd/v/vmikuni/datasets/

    cmd="omnilearned evaluate \
        -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
        --save-tag pretrain_l \
        --dataset ${ds} \
        --num-classes 210 \
        --size large \
        --use-event-loss \
        --interaction \
        --batch 32 \
        --local-interaction \
	--use-pid \
	--use-add \
        -o /pscratch/sd/v/vmikuni/Omnilearned/"

    echo "======================================"
    echo "Running dataset: ${ds}"
    echo "======================================"

    set -x
    srun -l -u \
        bash -c "
        source export_ddp.sh
        $cmd
        "
    EXIT_CODE=$?
    set +x

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Dataset ${ds} FAILED (exit code: $EXIT_CODE)"
        FAILED+=("${ds}")
    else
        echo "Dataset ${ds} succeeded"
    fi

done

echo
echo "================ SUMMARY ================"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All datasets succeeded"
else
    echo "Failed datasets:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi




