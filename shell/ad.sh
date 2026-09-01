# load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)


# High level feature model

#cmd="omnilearned train-hl  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sb_hl --dataset aspen_top_ad_sb --epoch 100 --lr 5e-5 --conditional --batch 128 --num-cond 1"

#cmd="omnilearned evaluate-hl -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sb_hl --dataset aspen_top_ad_sb --conditional --batch 2048 --num-cond 1  --num-feat 4 --path /pscratch/sd/v/vmikuni/Omnilearned/"


#cmd="omnilearned train-hl  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_bsm_ad_sb_hl --dataset aspen_bsm_ad_sb --epoch 100 --lr 5e-5 --conditional --batch 128 --num-cond 1  --num-feat 4"

#cmd="omnilearned evaluate-hl -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_bsm_ad_sb_hl --dataset aspen_bsm_ad_sb --conditional --batch 2048 --num-cond 1 --path /pscratch/sd/v/vmikuni/Omnilearned/ -o /pscratch/sd/v/vmikuni/datasets/aspen_bsm_ad_sr_hl --num-feat 4"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_bsm_ad_sb_m --dataset aspen_bsm_ad_sb --mode generator --epoch 35 --lr 1e-5 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 --wd 0.0 --conditional --num-classes 1 --batch 32 --num-cond 5 --interaction --local-interaction"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_bsm_ad_sb_m --dataset aspen_bsm_ad_sr_hl  --num-classes 1 --conditional --size medium --mode generator --batch 256 --num-cond 5 --interaction -o /pscratch/sd/v/vmikuni/datasets/aspen_bsm_ad_sr  --interaction --local-interaction"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_bsm_ad_sr_m --dataset aspen_bsm_ad_sr --epoch 10 --lr 5e-6 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 10.0 --wd 1.0 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 5"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_bsm_ad_sr_s --dataset aspen_bsm_ad_sr --epoch 10 --lr 5e-6 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 5.0 --wd 0.1 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 5"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_bsm_ad_sr_m --dataset aspen_bsm_ad_sr --epoch 5 --lr 5e-6 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 5.0 --wd 0.1 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 4 --local-interaction"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_bsm_ad_sr_s --dataset aspen_bsm_ad_sr  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 5 --batch 64 -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_bsm_ad_sr_s --dataset aspen_bsm_ad_sb  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 5 --batch 64 -o /pscratch/sd/v/vmikuni/Omnilearned/"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_bsm_ad_sr_m --dataset aspen_bsm_ad_sr  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 5 --batch 64 -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_bsm_ad_sr_m --dataset aspen_bsm_ad_sb  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 5 --batch 64 -o /pscratch/sd/v/vmikuni/Omnilearned/"



#Train the generative model

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_top_ad_sb_s --dataset aspen_top_ad_sb --mode generator --epoch 30 --lr 1e-5 --size small --wd 0.0 --num-transf-heads 2 --warmup-epoch 0 --conditional --num-classes 1 --batch 64 --fine-tune --pretrain-tag pretrain_s  --lr-factor 1.0"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_top_ad_sb_m --dataset aspen_top_ad_sb --mode generator --epoch 55 --lr 1e-5 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 --wd 0.0 --conditional --num-classes 1 --batch 32 --num-cond 4 --interaction --warmup-epoch 1"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sb_s --dataset aspen_top_ad_sb --mode generator --epoch 10 --lr 5e-5 --size small --wd 0.0 --warmup-epoch 0 --conditional --num-classes 1 --batch 128"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sb_m --dataset aspen_top_ad_sb --mode generator --epoch 30 --lr 5e-5 --size medium --wd 0.0 --warmup-epoch 0 --conditional --num-classes 1 --batch 32"





#Generate events

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sb_s --dataset aspen_top_ad_sr_hl  --num-classes 1 --conditional  --size small --mode generator --batch 128  --num-cond 4"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sb_m --dataset aspen_top_ad_sr_hl  --num-classes 1 --conditional --size medium --mode generator --batch 256 --num-cond 4 --interaction"



#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sb_s --dataset aspen_top_ad_sr_hl  --num-classes 1 --conditional --size small --mode generator --batch 128 --num-cond 4"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sb_m --dataset aspen_top_ad_sr_hl  --num-classes 1 --conditional --size medium  --mode generator --batch 128  --num-cond 4"





#Train the classifier

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_top_ad_sr_s --dataset aspen_top_ad_sr --epoch 5 --lr 5e-6 --size small --fine-tune --pretrain-tag pretrain_s --lr-factor 1.0 --wd 0.1 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 4"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_top_ad_sr_m --dataset aspen_top_ad_sr --epoch 5 --lr 5e-6 --size medium --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 --wd 1.0 --warmup-epoch 0 --num-classes 2  --clip-inputs --conditional --num-cond 4"


#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sr_s --dataset aspen_top_ad_sr --epoch 10 --lr 5e-5 --size small --wd 1. --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 4"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sr_m --dataset aspen_top_ad_sr --epoch 5 --lr 5e-6 --size medium --wd 1.0 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 4"

#cmd="omnilearned train  -o /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sr_l --dataset aspen_top_ad_sr --epoch 5 --lr 5e-6 --size large --wd 1.0 --warmup-epoch 0 --num-classes 2 --clip-inputs  --conditional --num-cond 4 --batch 4"



# Evaluate the classifier

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sr_s --dataset aspen_top_ad_sr  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 4 --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sr_s --dataset aspen_top_ad_sb  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 4 --batch 64"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sr_s --dataset aspen_top_ad_sr  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 4  --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sr_s --dataset aspen_top_ad_sb  --num-classes 2 --size small --clip-inputs  --conditional --num-cond 4  --batch 64"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sr_m --dataset aspen_top_ad_sr  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 4 --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag aspen_top_ad_sr_m --dataset aspen_top_ad_sb  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 4 --batch 64"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sr_m --dataset aspen_top_ad_sr  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 4  --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_aspen_top_ad_sr_m --dataset aspen_top_ad_sb  --num-classes 2 --size medium --clip-inputs  --conditional --num-cond 4  --batch 64"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset aspen_top_ad_sr  --num-classes 210 --size small --use-event-loss --interaction --batch 64 --use-pid -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset aspen_top_ad_sb  --num-classes 210 --size small --use-event-loss --interaction --batch 64 --use-pid -o /pscratch/sd/v/vmikuni/Omnilearned/"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset aspen_top_ad_sr  --num-classes 210 --size large --use-event-loss --interaction --batch 16 -o /pscratch/sd/v/vmikuni/Omnilearned/ --use-pid"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset aspen_top_ad_sb  --num-classes 210 --size large --use-event-loss --interaction --batch 16 -o /pscratch/sd/v/vmikuni/Omnilearned/ --use-pid"



#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_top_ad_sr  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --use-pid  -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_top_ad_sb  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --use-pid  -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset qcd_dijet  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset cms_muon  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset qcd_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_bsm_ad_sr  --num-classes 210 --size medium --use-event-loss --interaction --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_bsm_ad_sb  --num-classes 210 --size medium --use-event-loss --interaction --batch 64"


# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/ -o /pscratch/sd/v/vmikuni/Omnilearned/  --save-tag pretrain_s --dataset aspen_bsm  --num-classes 210 --size small --use-event-loss --interaction --batch 64 --use-pid --use-add"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/ -o /pscratch/sd/v/vmikuni/Omnilearned/  --save-tag pretrain_m --dataset cms_top  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --use-pid --use-add"


# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset cms_muon  --num-classes 210 --size large --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset cms_top  --num-classes 210 --size large --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset jetht_dijet  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset top_dijet  --num-classes 210 --size medium --use-event-loss --interaction --batch 64 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"


# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset singlemuon_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 128 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset qcd_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 128 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset jetht_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 128 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset top_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 128 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset ggf_dijet  --num-classes 210 --size small --use-event-loss --interaction --batch 128 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"


# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset singlemuon_dijet  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset jetht_dijet  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset ggf_dijet  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset top_dijet  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset 4top_dijet  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset topv_cr  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

#CR Validation
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset singlemuon_cr  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset top_cr  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"

# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_0  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
# cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_1  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_2  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_3  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_4  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_5  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_6  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_7  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_8  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_l --dataset qcd_dijet_9  --num-classes 210 --size large --use-event-loss --interaction --batch 32 --local-interaction --use-pid --use-add -o /pscratch/sd/v/vmikuni/Omnilearned/"



#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_bsm_ad_sr  --num-classes 210 --size medium --use-event-loss --interaction --batch 64"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen_bsm_ad_sb  --num-classes 210 --size medium --use-event-loss --interaction --batch 64"



#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen  --num-classes 210 --size medium --use-event-loss --use-pid --use-add --interaction --batch 128"


set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
