module load pytorch/2.8.0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


export BATCH=512
export EPOCHS=30
export LR=5e-5
export WD=0.0


#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_s --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --conditional --num-cond 16 --pid-dim 8"

#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_int_s --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --conditional --num-cond 16 --pid-dim 8 --local-interaction --interaction"

export BATCH=512
export EPOCHS=15
export LR=5e-5
export WD=2.0
export LRFACTOR=5.


#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_ft_s --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --fine-tune --pretrain-tag pretrain_s --lr-factor $LRFACTOR  --conditional --num-cond 16 --pid-dim 8"

#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_ft_int_s --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --fine-tune --pretrain-tag pretrain_s --lr-factor $LRFACTOR  --pid-dim 8  --local-interaction --interaction --conditional --num-cond 16"

#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_ft_int_ot --interaction-type ot --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --fine-tune --pretrain-tag pretrain_s --lr-factor $LRFACTOR  --pid-dim 8  --local-interaction --interaction --conditional --num-cond 16 --k 5"


export BATCH=512
export EPOCHS=20
export LR=5e-6
export LRFACTOR=5.


#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva_ft_int_m --dataset minerva --epoch $EPOCHS --lr $LR --size medium --wd $WD --num-classes 5 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5  --fine-tune --pretrain-tag pretrain_m --lr-factor $LRFACTOR  --pid-dim 8  --conditional --num-cond 16"





#Eval

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag minerva_s --dataset minerva  --num-classes 5 --conditional --size small --mode classifier --batch 256 --num-feat 4  --use-pid --use-add --num-add 5 --num-cond 16 --pid-dim 8 -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag minerva_int_s --dataset minerva  --num-classes 5 --conditional --size small --mode classifier --batch 256 --num-feat 4  --use-pid --use-add --num-add 5 --num-cond 16 --pid-dim 8 -o /pscratch/sd/v/vmikuni/Omnilearned/  --interaction --local-interaction"

cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag minerva_ft_s --dataset minerva  --num-classes 5 --conditional --size small --mode classifier --batch 256 --num-feat 4  --use-pid --use-add --num-add 5 --num-cond 16 --pid-dim 8 -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag minerva_ft_int_s --dataset minerva  --num-classes 5 --conditional --size small --mode classifier --batch 256 --num-feat 4  --use-pid --use-add --num-add 5 --num-cond 16 --pid-dim 8 -o /pscratch/sd/v/vmikuni/Omnilearned/  --interaction --local-interaction"



set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "

