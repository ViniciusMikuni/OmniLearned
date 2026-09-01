module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


#Large PET Total params: 57.81M

#Finetune Classification

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l --dataset top --epoch 5  --lr 1e-6 --size large --wd 10. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 10."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_2 --dataset top --epoch 5  --lr 1e-6 --size large --wd 10. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 10."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "


# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_3 --dataset top --epoch 5  --lr 1e-6 --size large --wd 10. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 10."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_4 --dataset top --epoch 5  --lr 1e-6 --size large --wd 10. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 10."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_5 --dataset top --epoch 5  --lr 1e-6 --size large --wd 10. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 10."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "



# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_l --dataset qg --epoch 5  --lr 5e-6 --size large --wd 1. --fine-tune --pretrain-tag pretrain_l --interaction --batch 8 --lr-factor 5."
# set -x
# srun -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "
