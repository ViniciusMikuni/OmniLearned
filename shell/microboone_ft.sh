module load pytorch

# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB

# for DDP
#export MASTER_ADDR=$(hostname)



export BATCH=8

export LR=5e-5
export WD=3.0
export LRFACTOR=20.
export EPOCH=15


# for I in {0..9}
# do
#     cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_microboone_s_${I} --dataset microboone --epoch $EPOCH --lr $LR --size small --wd $WD --num-classes 2 --num-feat 4 --batch $BATCH --mode classifier --lr-factor $LRFACTOR --fine-tune --pretrain-tag pretrain_s_test --interaction-type ot --num-coord 3 --local-interaction --k 5"

#     set -x
#     srun -l -u \
#         bash -c "
#         source export_ddp.sh
#         $cmd
#         "
#     set +x

#     cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_microboone_full_s_${I} --dataset microboone --epoch $EPOCH --lr $LR --size small --wd $WD --num-classes 2 --num-feat 4 --batch $BATCH --mode classifier --lr-factor $LRFACTOR --fine-tune --pretrain-tag pretrain_s_test --interaction-type ot --num-coord 3 --local-interaction --interaction  --k 5"

#     set -x
#     srun -l -u \
#         bash -c "
#         source export_ddp.sh
#         $cmd
#         "
#     set +x
    
#     echo "Completed iteration $I"
# done



for I in {0..9}
do
    omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_microboone_full_s_${I} --dataset microboone  --num-classes 2 --size small --interaction -o /pscratch/sd/v/vmikuni/Omnilearned/ --num-feat 4 --interaction-type ot --num-coord 3  --local-interaction --batch 8 --k 5
    omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_microboone_s_${I} --dataset microboone  --num-classes 2 --size small -o /pscratch/sd/v/vmikuni/Omnilearned/ --num-feat 4 --interaction-type ot --num-coord 3  --local-interaction --batch 8 --k 5
    echo "Completed iteration $I"
done



