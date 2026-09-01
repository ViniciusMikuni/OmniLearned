module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


#Medium PET Total params: 57.81M

#Pretrain
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset pretrain --use-pid --use-add --num-classes 210 --batch 32 --lr 5e-6 --iterations 1000 --mode pretrain  --epoch 500 --wd 0.1 --num-transf 12 --base-dim 512 --num-head 16 --feature-drop 0.1 --use-event-loss --num-workers 32 --wandb"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m_class --dataset pretrain --use-pid --use-add --num-classes 210 --batch 32  --iterations 1000 --mode classifier  --epoch 500 --wd 0.1 --num-transf 12 --base-dim 512 --num-head 16 --feature-drop 0.1 --use-event-loss --num-workers 32 --lr 5e-5 --wandb"

#Finetune Classification
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_m --dataset top  --epoch 10 --fine-tune --pretrain-tag pretrain_m_class --lr 3e-6 --lr-factor 10. --base-dim 512 --num-transf 12 --num-head 16 --wd 0.01 --warmup-epoch 0"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_qg_m --dataset qg --use-pid  --epoch 10 --fine-tune --pretrain-tag pretrain_m_class --lr 1e-6 --lr-factor 5. --base-dim 512 --num-transf 12 --num-head 16 --wd 1.0"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --dataset top  --epoch 10 --fine-tune --pretrain-tag pretrain_m --lr 3e-6 --lr-factor 10. --base-dim 512 --num-transf 12 --num-head 16 --wd 0.01 --warmup-epoch 0"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m --dataset qg --use-pid  --epoch 10 --fine-tune --pretrain-tag pretrain_m --lr 1e-6 --lr-factor 5. --base-dim 512 --num-transf 12 --num-head 16 --wd 10.0"


# #From scratch
# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5  --num-head 16  --num-transf-heads 2"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_2 --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5  --num-head 16  --num-transf-heads 2"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_3 --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5  --num-head 16  --num-transf-heads 2"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_4 --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5  --num-head 16  --num-transf-heads 2"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_5 --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5 --num-head 16  --num-transf-heads 2"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "




# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --num-head 16  --wd 0.1  --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_2 --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --num-head 16  --wd 0.1  --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_3 --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --num-head 16  --wd 0.1  --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_4 --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --num-head 16  --wd 0.1  --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_5 --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --num-head 16  --wd 0.1  --num-transf-heads 2"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "




# #Finetune

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_m --dataset jetclass --epoch 200  --lr 1e-6 --base-dim 512 --num-transf 12 --wd 10.  --num-head 16  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m --batch 32 --use-pid --use-add --num-classes 10 --iterations 1000"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_test --dataset top --epoch 5  --lr 1e-6 --size medium --wd 10. --fine-tune --pretrain-tag pretrain_m --lr-factor 1. --interaction"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_2 --dataset top --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --wd 10.  --num-head 16  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_3 --dataset top --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --wd 10.  --num-head 16  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_4 --dataset top --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --wd 10.  --num-head 16  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_5 --dataset top --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --wd 10. --num-head 16  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m"
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "




# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m --dataset qg --use-pid --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --num-head 16  --wd 1.0  --num-transf-heads 2  --fine-tune --pretrain-tag pretrain_m"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_2 --dataset qg --use-pid --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --num-head 16  --wd 1.0  --num-transf-heads 2 --fine-tune --pretrain-tag pretrain_m"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_3 --dataset qg --use-pid --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --num-head 16  --wd 1.0  --num-transf-heads 2 --fine-tune --pretrain-tag pretrain_m"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_4 --dataset qg --use-pid --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --num-head 16  --wd 1.0  --num-transf-heads 2 --fine-tune --pretrain-tag pretrain_m"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_5 --dataset qg --use-pid --epoch 5  --lr 1e-6 --base-dim 512 --num-transf 12 --num-head 16  --wd 1.0  --num-transf-heads 2 --fine-tune --pretrain-tag pretrain_m"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "



export NEVT=19651

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_camels_s_$NEVT --dataset camels --epoch 150 --lr 5e-6 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 4  --mode regression"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "




# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_camels_s_$NEVT --dataset camels --epoch 200 --lr 5e-6 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 4 --mode segmentation --num-gen-classes 3"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_R_quijote_s_$NEVT --dataset quijote --epoch 100 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 2  --mode regression"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "



# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_ftag_quijote_s_$NEVT --dataset quijote --epoch 200 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 20.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 8  --mode ftag --num-gen-classes 3"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_v_quijote_s_$NEVT --dataset quijote --epoch 300 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 10.0 --wd 1.0 --warmup-epoch 0  --num-classes 2 --num-feat 3 --batch 8 --mode segmentation --num-gen-classes 3"

# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "
