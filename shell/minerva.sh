module load pytorch/2.8.0

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)


export BATCH=512
export EPOCHS=50
export LR=1e-4
export WD=0.0



cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag minerva --dataset minerva --epoch $EPOCHS --lr $LR --size small --wd $WD --num-classes 4 --num-feat 4 --batch $BATCH --mode classifier --use-pid --use-add --num-add 5 --conditional --num-cond 16 --pid-dim 8"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "

