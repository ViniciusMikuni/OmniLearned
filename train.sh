#load libs
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)

#Big PET  Total params: 460.84M

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_l --dataset pretrain --use-pid --use-add --num-classes 210 --batch 8 --iterations 1000 --mode pretrain --epoch 500 --wd 1.0 --lr 1e-5 --num-transf 28 --base-dim 1024 --num-head 32 --feature-drop 0.1 --use-event-loss --num-workers 32 --num-transf-heads 4--resuming"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_l_class --dataset pretrain --use-pid --use-add --num-classes 210 --batch 16 --iterations 1000 --mode classifier --epoch 1000 --wd 1.0 --lr 1e-6 --num-transf 28 --base-dim 1152 --num-head 16 --feature-drop 0.1 --use-event-loss --num-workers 32 --num-transf-heads 4 --wandb --use-amp"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_l --dataset top --batch 16 --epoch 10 --wd 10.0 --lr 1e-6 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4 --fine-tune --pretrain-tag pretrain_l_class"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_qg_l --dataset qg --use-pid --batch 16 --epoch 10 --wd 10.0 --lr 1e-6 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4 --fine-tune --pretrain-tag pretrain_l_class"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l --dataset top --batch 16 --epoch 10 --wd 10.0 --lr 1e-6 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4 --fine-tune --pretrain-tag pretrain_l"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_l --dataset qg --use-pid --batch 16 --epoch 10 --wd 10.0 --lr 1e-6 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4 --fine-tune --pretrain-tag pretrain_l"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_l --dataset top --batch 16 --epoch 15 --wd 0.5 --lr 1e-5 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_l --dataset qg --use-pid --batch 16 --epoch 15 --wd 0.5 --lr 1e-5 --num-transf 28 --base-dim 1152 --num-head 32  --num-workers 32 --num-transf-heads 4"

#Medium PET Total params: 57.81M

#Pretrain
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset pretrain --use-pid --use-add --num-classes 210 --batch 32 --lr 5e-6 --iterations 1000 --mode pretrain  --epoch 500 --wd 0.1 --num-transf 12 --base-dim 512 --num-head 16 --feature-drop 0.1 --use-event-loss --num-workers 32 --wandb"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m_class --dataset pretrain --use-pid --use-add --num-classes 210 --batch 32  --iterations 1000 --mode classifier  --epoch 500 --wd 0.1 --num-transf 12 --base-dim 512 --num-head 16 --feature-drop 0.1 --use-event-loss --num-workers 32 --lr 5e-5 --wandb"

#Finetune Classification
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_m --dataset top  --epoch 10 --fine-tune --pretrain-tag pretrain_m_class --lr 1e-6 --lr-factor 5. --base-dim 512 --num-transf 12 --num-head 16 --wd 5.0 --warmup-epoch 0 --num-head 16"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_qg_m --dataset qg --use-pid  --epoch 10 --fine-tune --pretrain-tag pretrain_m_class --lr 1e-6 --lr-factor 5. --base-dim 512 --num-transf 12 --num-head 16 --wd 1.0 --num-head 16"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --dataset top  --epoch 5 --fine-tune --pretrain-tag pretrain_m --lr 5e-6 --lr-factor 1. --base-dim 512 --num-transf 12 --num-head 16 --wd 10.0 --warmup-epoch 0"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m --dataset qg --use-pid  --epoch 10 --fine-tune --pretrain-tag pretrain_m --lr 1e-6 --lr-factor 5. --base-dim 512 --num-transf 12 --num-head 16 --wd 1.0"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet150_m --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-6 --base-dim 512 --num-transf 12  --num-head 16 --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --num-classes 5 --batch 32 --fine-tune --pretrain-tag pretrain_m --num-cond 3 --lr-factor 5."


#From scratch
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12 --wd 0.5  --num-transf-heads 2"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m --dataset qg --use-pid --epoch 15  --lr 5e-5 --base-dim 512 --num-transf 12  --wd 0.5  --num-transf-heads 2"

#Generative

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_m --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-5 --base-dim 512 --num-transf 12  --num-head 16 --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --num-classes 5 --batch 32 --num-cond 3"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag jetnet150_m --dataset jetnet150  --num-classes 5 --conditional --num-transf 12 --base-dim 512 --num-head 16 --num-transf-heads 2 --mode generator --batch 64"


#Small PET Total params: 2.8M

#Pretrain
#cmd="omnilearned train  -o  /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s --dataset pretrain --use-pid --use-add --num-classes 210 --batch 128 --iterations 1000 --mode pretrain  --epoch 500 --num-transf 8 --base-dim 128 --num-head 8 --num-workers 32 --feature-drop 0.3 --use-event-loss --num-transf-heads 2 --wandb"
#cmd="omnilearned train  -o  /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s_class --dataset pretrain --use-pid --use-add --num-classes 210 --batch 128 --iterations 1000 --mode classifier  --epoch 500 --num-transf 8 --base-dim 128 --num-head 8 --num-workers 32 --feature-drop 0.3 --use-event-loss --num-transf-heads 2 --wandb"

#Finetune Classification

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_s --dataset top --epoch 15 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s_class --lr-factor 10.0 --wd 0.5 --num-transf-heads 2 --warmup-epoch 0"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_qg_s --dataset qg --use-pid --epoch 15 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s_class --lr-factor 5.0 --wd 0.5 --num-transf-heads 2"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s --dataset top --epoch 10 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --num-transf-heads 2 --warmup-epoch 1"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s --dataset qg --use-pid --epoch 15 --lr 5e-6 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.5 --num-transf-heads 2"

#Finetune Generation
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet150_s --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-5 --base-dim 128 --num-transf 8  --num-head 8 --fine-tune --pretrain-tag pretrain_s --lr-factor 1.0 --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --num-classes 5"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_jetnet150_s --dataset jetnet150  --num-classes 5 --conditional --num-transf 8 --base-dim 128 --num-head 8 --num-transf-heads 2 --mode generator --batch 128"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_s --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-5 --base-dim 128 --num-transf 8  --num-head 8 --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --num-classes 5"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag jetnet150_s --dataset jetnet150  --num-classes 5 --conditional --num-transf 8 --base-dim 128 --num-head 8 --num-transf-heads 2 --mode generator --batch 128"


#From Scratch
# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_s --dataset top --epoch 15 --warmup-epoch 0 --lr 5e-4 --base-dim 128 --num-transf 8 --num-head 8 --wd 0.5 --num-transf-heads 2"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_s --dataset qg --use-pid --epoch 15 --lr 5e-4 --base-dim 128 --num-transf 8 --num-head 8 --wd 0.1 --num-transf-heads 2"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag sbi_s --dataset custom --epoch 15 --warmup-epoch 0 --lr 5e-5 --base-dim 128 --num-transf 8 --num-head 8 --wd 0.1 --num-transf-heads 2  --num-cond 1 --use-pid"

#Test
#cmd="omnilearned train  -o  /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s_test --dataset pretrain --use-pid --use-add --num-classes 210 --batch 128 --iterations 10 --mode pretrain  --epoch 1000 --num-transf 4 --base-dim 64 --num-head 8 --num-workers 32 --feature-drop 0.1 --use-event-loss --num-transf-heads 2"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
