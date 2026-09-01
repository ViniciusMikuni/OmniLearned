module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)

export NEVT=19651

# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_s_$NEVT --dataset camels --epoch 200 --lr 5e-5 --size small --wd 1.0 --num-classes 2 --num-feat 3 --batch 4 --mode regression "

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "



# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_camels_s_$NEVT --dataset camels --epoch 300 --lr 5e-4 --size small --wd 1.0 --num-classes 2 --num-feat 3 --batch 4  --mode segmentation --num-gen-classes 3"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "




cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag quijote_s_$NEVT --dataset quijote --epoch 200 --lr 5e-5 --size small --wd 1.0 --num-classes 2 --num-feat 3 --batch 8 --mode regression"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_quijote_s_$NEVT --dataset quijote --epoch 300 --lr 5e-5 --size small --wd 1.0 --num-classes 2 --num-feat 3 --batch 8  --mode segmentation --num-gen-classes 3"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "





# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag v_camels_s --dataset camels --epoch 300 --lr 5e-4 --size small --wd 5.0 --num-classes 2 --num-feat 3 --batch 4  --mode segmentation --num-gen-classes 3 --wandb"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_camels_s --dataset camels --epoch 300 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 5.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 4 --mode segmentation --num-gen-classes 3   --wandb"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s --dataset quijote --epoch 80 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 8  --wandb --mode segmentation --num-gen-classes 3"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "
