#load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

#Examples for different model sizes. Uncomment the cmd you wan. You can run on a single GPU by running the cmd line relevant or within a slurm session with ./train.sh

#Large PET  Total params: 460.84M

cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_l --mode pretrain --dataset pretrain --num-classes 210  --epoch 500  --iterations 1000  --num-workers 32 --lr 1e-05 --optim lion --feature-drop 0.1 --wd 0.1 --batch 8 --size large  --use-pid --use-add --use-event-loss --interaction --wandb --resuming"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l --dataset top --batch 8 --epoch 5 --wd 10.0 --lr 1e-6 --size large --num-workers 4  --fine-tune --pretrain-tag pretrain_l --interaction"

#From scratch

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_l --dataset top --batch 8 --epoch 10 --wd 10.0 --lr 5e-6 --size large --interaction"


#Medium PET Total params: 57.81M

#Pretrain

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset pretrain --use-pid --use-add --num-classes 210 --batch 32 --lr 5e-6 --iterations 1000 --mode pretrain  --epoch 500 --wd 0.1 --size medium --feature-drop 0.1 --use-event-loss --num-workers 32 --interaction --wandb"

#Fine-tune

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --dataset top  --epoch 5 --fine-tune --pretrain-tag pretrain_m --lr 1e-6 --lr-factor 1. --size medium  --wd 10.0 --interaction"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_atlas_flav_m --dataset atlas_flav --epoch 30 --lr 5e-5 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 --wd 0.1  --use-add --num-add 17 --num-classes 4  --batch 256 --iterations 2000 --num-gen-classes 8 --mode ftag --conditional --num-cond 4 --interaction"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet150_m --dataset jetnet150 --conditional --mode generator --epoch 100 --lr 5e-6 --size medium  --wd 0.0 --warmup-epoch 0 --num-classes 5 --batch 32 --fine-tune --pretrain-tag pretrain_m --lr-factor 5."


#From scratch

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top --epoch 15  --lr 5e-5 --size medium --wd 0.5 --interaction"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_m --dataset jetnet150 --conditional --mode generator --epoch 100 --lr 5e-5 --size medium --wd 0.0 --warmup-epoch 0 --num-classes 5 --batch 32"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag atlas_flav_m --dataset atlas_flav --epoch 40 --lr 7e-5 --size medium --wd 0.1 --interaction --use-add --num-add 17 --num-classes 4 --iterations 2000 --batch 256 --num-gen-classes 8 --mode ftag --conditional --num-cond 4"


#Small PET Total params: 2.8M

#Pretrain
#cmd="omnilearned train  -o  /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s_test --dataset pretrain --use-pid --use-add --num-classes 210 --batch 128 --iterations 1000 --mode pretrain  --epoch 500 --size small --num-workers 32 --feature-drop 0.1 --use-event-loss --interaction"

#Finetune 

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s --dataset top --epoch 10 --lr 5e-6 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --warmup-epoch 1 --interaction"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_atlas_flav_s --dataset atlas_flav --epoch 30 --lr 5e-4 --size small --wd 0.1 --use-add --num-add 17 --num-classes 4 --iterations 2000 --batch 512  --fine-tune --pretrain-tag pretrain_s --lr-factor 1.0 --interaction --num-gen-classes 8 --mode ftag --conditional --num-cond 4"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet150_s --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 1e-5 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 1.0 --wd 0.0 --num-classes 5"

#From Scratch
# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_s --dataset top --epoch 15 --lr 5e-4 --size small --wd 0.5 --interaction"
# cmd="omnilearned train -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_s --dataset qg --epoch 15 --lr 5e-4 --size small --wd 0.5 --interaction --use-pid"
#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag atlas_flav_s --dataset atlas_flav --epoch 30 --lr 5e-5 --size small --wd 0.1 --interaction --use-add --num-add 17 --num-classes 4 --iterations 2000 --batch 512 --wandb --num-gen-classes 8 --mode ftag --conditional --num-cond 4"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_s --dataset jetnet150 --conditional --mode generator --epoch 200 --lr 5e-5 --size small --wd 0.0 --num-classes 5"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
