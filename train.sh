# load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

#Big PET  Total params: 463.68M

# cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_l --dataset pretrain --use-pid --use-add --use-clip --num-classes 201 --batch 8 --iterations 1000 --mode pretrain --epoch 1000 --wd 0.01 --num-transf 28 --base-dim 1152 --num-head 16 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0"

#Medium PET Total params: 62.61M

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset pretrain --use-pid --use-add --use-clip --num-classes 201 --batch 16  --iterations 1000 --mode pretrain  --epoch 250 --wd 0.01 --num-transf 12 --base-dim 512 --num-head 8 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0 --wandb"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top   --epoch 15 --fine-tune --pretrain-tag pretrain_m --lr 5e-6 --lr-factor 0.1 --base-dim 512 --num-transf 12  --attn-drop 0.1 --mlp-drop 0.1 --wd 0.3"

#Small PET Total params: 0.9M

#cmd="omnilearned train  -o ./ --save-tag pretrain_s --dataset pretrain --use-pid --use-add --use-clip --num-classes 201 --batch 64 --iterations 1000 --mode pretrain  --epoch 1000 --wd 0.1 --num-transf 6 --base-dim 96 --num-head 8 --num-workers 32 --feature-drop 0.1 --wandb --resuming"
#cmd="omnilearned train  -o ./ --save-tag pretrain_s_gen --dataset pretrain --use-pid --use-add --num-classes 201 --batch 64 --iterations 1000 --mode pretrain  --epoch 1000 --wd 0.1 --num-transf 6 --base-dim 96 --num-head 8 --num-workers 32 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0 --wandb --resuming"
#cmd="omnilearned train  -o ./ --save-tag pretrain_s_class --dataset pretrain --use-pid --use-add --num-classes 201 --batch 128 --iterations 1000 --mode classifier  --epoch 500 --wd 0.01 --num-transf 6 --base-dim 64 --num-head 8 --num-workers 32 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0 --wandb"

#cmd="omnilearned train  -o ./ --save-tag top_s --dataset top   --epoch 15 --fine-tune --pretrain-tag pretrain_s --lr 5e-4 --lr-factor 1.0 --base-dim 96  --attn-drop 0.1 --mlp-drop 0.1 --wd 0.3"
#cmd="omnilearned train  -o ./ --save-tag top_s_gen --dataset top   --epoch 15 --fine-tune --pretrain-tag pretrain_s_gen --lr 5e-4 --lr-factor 1.0 --base-dim 96  --attn-drop 0.1 --mlp-drop 0.1  --wd 0.3"
#cmd="omnilearned train  -o ./ --save-tag top_s --dataset top   --epoch 15  --lr 5e-4 --base-dim 64"


#cmd="omnilearned train  -o ./ --save-tag top_s --dataset top   --epoch 15 --lr 5e-5  --base-dim 64 --num-transf 6  --attn-drop 0.1 --mlp-drop 0.1 --wd 0.3"
#cmd="omnilearned train  -o ./ --save-tag top_s --dataset top   --epoch 15 --lr 5e-5  --base-dim 1152 --num-transf 28  --attn-drop 0.1 --mlp-drop 0.1 --wd 0.3 --num-head 16 --batch 32"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
