module load pytorch/2.8.0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)



export BATCH=128
export EPOCHS=500
export LR=5e-5
export WD=0.1
export NPART=30

#AE

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet${NPART}_ae_s --dataset jetnet$NPART  --batch $BATCH --mode encoder --epoch $EPOCHS --lr $LR --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 10.0  --num-latent 10 --num-classes 1 --conditional --num-cond 3 --local-interaction --interaction --num-part $NPART --wd $WD"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_jetnet${NPART}_ae_s --dataset jetnet${NPART}  --num-latent 10 --num-classes 1 --num-cond 3 --batch $BATCH --mode encoder --size small --conditional --local-interaction --interaction  --num-part $NPART  -o /pscratch/sd/v/vmikuni/datasets/jetnet${NPART}_ae/train"


#Generative Models

export BATCH=128
export EPOCHS=300
export LR=3e-5
export WD=0.0



#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet${NPART}_gen_s --dataset jetnet$NPART  --batch $BATCH --mode generator --epoch $EPOCHS --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 1.0 --wd 0.0 --num-classes 1"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet${NPART}_gen_cond_s --dataset jetnet$NPART  --batch $BATCH --mode generator --epoch $EPOCHS --lr $LR --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 10.0 --num-classes 1 --conditional --num-cond 3"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet${NPART}_ae_gen_cond_s --dataset jetnet${NPART}_ae  --batch $BATCH --mode generator --epoch $EPOCHS --lr $LR --size small  --num-classes 1 --conditional --num-cond 13 --fine-tune --pretrain-tag pretrain_s --lr-factor 10.0"




cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_jetnet${NPART}_gen_cond_s --dataset jetnet${NPART}  --num-classes 1 --batch $BATCH --mode generator --size small --conditional --sbi --num-cond 3 -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_jetnet${NPART}_ae_gen_cond_s --dataset jetnet${NPART}_ae  --num-classes 1 --num-cond 13 --batch $BATCH --mode generator --size small --conditional --sbi -o /pscratch/sd/v/vmikuni/Omnilearned/"


set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "

