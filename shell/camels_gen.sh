module load pytorch/2.8.0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


export BATCH=8
export REPOCHS=150
export LR=5e-5
export WD=0.1
export LRFACTOR=1.

#AE

#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_camels_ae_s --dataset camels --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 1 --num-latent 20 --num-feat 6 --batch $BATCH  --mode encoder --k 10 --conditional --num-cond 2 --local-interaction --interaction-type astro   --fine-tune --pretrain-tag pretrain_lite  --lr-factor $LRFACTOR  --num-part 5000 --iterations 50"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_camels_ae_s --dataset camels  --num-latent 20 --num-classes 1 --num-cond 2 --num-feat 6 --batch $BATCH --mode encoder --size small --conditional --local-interaction  --num-part 5000   -o /pscratch/sd/v/vmikuni/datasets/camels_ae/val"

#Generative Models
export REPOCHS=150
export LRFACTOR=5.
export LR=1e-5


#cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_g --dataset camels --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 1 --num-feat 6 --batch $BATCH  --mode generator --k 10 --conditional --num-cond 2 --iterations 50"

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_g_learned --dataset camels_ae --epoch $REPOCHS --lr $LR --size small --wd $WD --num-classes 1 --num-feat 6 --batch $BATCH  --mode generator --k 10 --conditional --num-cond 22 --iterations 50"





set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
