#load libs
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_s --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-5 --base-dim 128 --num-transf 8  --num-head 8 --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --num-classes 5 --interaction"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_s --dataset jetclass --epoch 50 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.0 --num-transf-heads 2 --warmup-epoch 1 --use-add --use-pid --iterations 1000 --batch 128 --num-classes 10"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_s_test --dataset top --epoch 15 --lr 5e-4 --size small --wd 0.5 --interaction"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_2 --dataset top --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2 --warmup-epoch 1"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_3 --dataset top --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2 --warmup-epoch 1"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_4 --dataset top --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2 --warmup-epoch 1"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_5 --dataset top --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2 --warmup-epoch 1"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s --dataset qg --use-pid --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_2 --dataset qg --use-pid --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_3 --dataset qg --use-pid --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_4 --dataset qg --use-pid --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_5 --dataset qg --use-pid --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "
