# Historical experiment commands

Reference only. Copy a command into a standalone shell script when needed. Paths and parameters below are the recorded historical values, not automatically selected settings.

## legacy/4top_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset 4top_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/aspen/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset aspen --size medium --interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 128
```

## legacy/aspen_bsm/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_s --dataset aspen_bsm --size small --interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 64
```

## legacy/aspen_bsm_ad_sb/classifier/evaluate_fine_tune_aspen_bsm_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sb --batch 64 --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_aspen_bsm_ad_sr_m --size medium --conditional --num-cond 5 --num-classes 2 --clip-inputs
```

## legacy/aspen_bsm_ad_sb/classifier/evaluate_fine_tune_aspen_bsm_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sb --batch 64 --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_aspen_bsm_ad_sr_s --size small --conditional --num-cond 5 --num-classes 2 --clip-inputs
```

## legacy/aspen_bsm_ad_sb/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sb --batch 64 --save-tag pretrain_m --size medium --interaction --use-event-loss --num-classes 210
```

## legacy/aspen_bsm_ad_sb/generator/train_fine_tune_aspen_bsm_ad_sb_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_bsm_ad_sb_m --pretrain-tag pretrain_m --dataset aspen_bsm_ad_sb --fine-tune --size medium --interaction --local-interaction --conditional --num-cond 5 --num-classes 1 --mode generator --batch 32 --epoch 35 --lr 1e-05 --lr-factor 1.0 --wd 0.0
```

## legacy/aspen_bsm_ad_sb/high_level/evaluate-hl_aspen_bsm_ad_sb_hl

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/datasets/aspen_bsm_ad_sr_hl --save-tag aspen_bsm_ad_sb_hl --dataset aspen_bsm_ad_sb --path /pscratch/sd/v/vmikuni/Omnilearned/ --num-feat 4 --conditional --num-cond 1 --batch 2048
```

## legacy/aspen_bsm_ad_sb/high_level/train-hl_aspen_bsm_ad_sb_hl

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_bsm_ad_sb_hl --dataset aspen_bsm_ad_sb --num-feat 4 --conditional --num-cond 1 --batch 128 --epoch 100 --lr 5e-05
```

## legacy/aspen_bsm_ad_sr/classifier/evaluate_fine_tune_aspen_bsm_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --batch 64 --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_aspen_bsm_ad_sr_m --size medium --conditional --num-cond 5 --num-classes 2 --clip-inputs
```

## legacy/aspen_bsm_ad_sr/classifier/evaluate_fine_tune_aspen_bsm_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --batch 64 --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_aspen_bsm_ad_sr_s --size small --conditional --num-cond 5 --num-classes 2 --clip-inputs
```

## legacy/aspen_bsm_ad_sr/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --batch 64 --save-tag pretrain_m --size medium --interaction --use-event-loss --num-classes 210
```

## legacy/aspen_bsm_ad_sr/classifier/train_fine_tune_aspen_bsm_ad_sr_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --fine-tune --conditional --num-classes 2 --warmup-epoch 0 --clip-inputs --lr 5e-06 --save-tag fine_tune_aspen_bsm_ad_sr_m --pretrain-tag pretrain_m --size medium --num-cond 5 --epoch 10 --lr-factor 10.0 --wd 1.0
```

## legacy/aspen_bsm_ad_sr/classifier/train_fine_tune_aspen_bsm_ad_sr_m_ad_28

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --fine-tune --conditional --num-classes 2 --warmup-epoch 0 --clip-inputs --lr 5e-06 --save-tag fine_tune_aspen_bsm_ad_sr_m --pretrain-tag pretrain_m --size medium --local-interaction --num-cond 4 --epoch 5 --lr-factor 5.0 --wd 0.1
```

## legacy/aspen_bsm_ad_sr/classifier/train_fine_tune_aspen_bsm_ad_sr_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_bsm_ad_sr --fine-tune --conditional --num-classes 2 --warmup-epoch 0 --clip-inputs --lr 5e-06 --save-tag fine_tune_aspen_bsm_ad_sr_s --pretrain-tag pretrain_s --size small --num-cond 5 --epoch 10 --lr-factor 5.0 --wd 0.1
```

## legacy/aspen_bsm_ad_sr_hl/generator/evaluate_fine_tune_aspen_bsm_ad_sb_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/datasets/aspen_bsm_ad_sr --save-tag fine_tune_aspen_bsm_ad_sb_m --dataset aspen_bsm_ad_sr_hl --size medium --interaction --local-interaction --conditional --num-cond 5 --num-classes 1 --mode generator --batch 256
```

## legacy/aspen_small/classifier/dataloader_np

```bash
omnilearned dataloader --dataset aspen_small --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/aspen_small/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset aspen_small --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/aspen_top_ad_sb/classifier/evaluate_aspen_top_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --save-tag aspen_top_ad_sr_m --size medium --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sb/classifier/evaluate_aspen_top_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --save-tag aspen_top_ad_sr_s --size small --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sb/classifier/evaluate_fine_tune_aspen_top_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --save-tag fine_tune_aspen_top_ad_sr_m --size medium --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sb/classifier/evaluate_fine_tune_aspen_top_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --save-tag fine_tune_aspen_top_ad_sr_s --size small --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sb/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --size large --interaction --use-pid --use-event-loss --num-classes 210 --batch 16
```

## legacy/aspen_top_ad_sb/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_m --size medium --interaction --use-pid --use-event-loss --num-classes 210 --batch 64
```

## legacy/aspen_top_ad_sb/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_s --size small --interaction --use-pid --use-event-loss --num-classes 210 --batch 64
```

## legacy/aspen_top_ad_sb/generator/train_aspen_top_ad_sb_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --conditional --num-classes 1 --mode generator --wd 0.0 --save-tag aspen_top_ad_sb_m --size medium --batch 32 --epoch 30 --warmup-epoch 0 --lr 5e-05
```

## legacy/aspen_top_ad_sb/generator/train_aspen_top_ad_sb_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --conditional --num-classes 1 --mode generator --wd 0.0 --save-tag aspen_top_ad_sb_s --size small --batch 128 --epoch 10 --warmup-epoch 0 --lr 5e-05
```

## legacy/aspen_top_ad_sb/generator/train_fine_tune_aspen_top_ad_sb_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sb --conditional --num-classes 1 --mode generator --wd 0.0 --save-tag fine_tune_aspen_top_ad_sb_m --pretrain-tag pretrain_m --fine-tune --size medium --interaction --num-cond 4 --batch 32 --epoch 55 --warmup-epoch 1 --lr 1e-05 --lr-factor 1.0
```

## legacy/aspen_top_ad_sb/high_level/evaluate-hl_aspen_top_ad_sb_hl

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sb_hl --dataset aspen_top_ad_sb --path /pscratch/sd/v/vmikuni/Omnilearned/ --num-feat 4 --conditional --num-cond 1 --batch 2048
```

## legacy/aspen_top_ad_sb/high_level/train-hl_aspen_top_ad_sb_hl

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag aspen_top_ad_sb_hl --dataset aspen_top_ad_sb --conditional --num-cond 1 --batch 128 --epoch 100 --lr 5e-05
```

## legacy/aspen_top_ad_sr/classifier/evaluate_aspen_top_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --save-tag aspen_top_ad_sr_m --size medium --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sr/classifier/evaluate_aspen_top_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --save-tag aspen_top_ad_sr_s --size small --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sr/classifier/evaluate_fine_tune_aspen_top_ad_sr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --save-tag fine_tune_aspen_top_ad_sr_m --size medium --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sr/classifier/evaluate_fine_tune_aspen_top_ad_sr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --save-tag fine_tune_aspen_top_ad_sr_s --size small --conditional --num-cond 4 --num-classes 2 --batch 64 --clip-inputs
```

## legacy/aspen_top_ad_sr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --size large --interaction --use-pid --use-event-loss --num-classes 210 --batch 16
```

## legacy/aspen_top_ad_sr/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_m --size medium --interaction --use-pid --use-event-loss --num-classes 210 --batch 64
```

## legacy/aspen_top_ad_sr/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_s --size small --interaction --use-pid --use-event-loss --num-classes 210 --batch 64
```

## legacy/aspen_top_ad_sr/classifier/train_aspen_top_ad_sr_l

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --conditional --num-cond 4 --num-classes 2 --warmup-epoch 0 --clip-inputs --save-tag aspen_top_ad_sr_l --size large --batch 4 --epoch 5 --lr 5e-06 --wd 1.0
```

## legacy/aspen_top_ad_sr/classifier/train_aspen_top_ad_sr_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --conditional --num-cond 4 --num-classes 2 --warmup-epoch 0 --clip-inputs --save-tag aspen_top_ad_sr_m --size medium --epoch 5 --lr 5e-06 --wd 1.0
```

## legacy/aspen_top_ad_sr/classifier/train_aspen_top_ad_sr_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --conditional --num-cond 4 --num-classes 2 --warmup-epoch 0 --clip-inputs --save-tag aspen_top_ad_sr_s --size small --epoch 10 --lr 5e-05 --wd 1.0
```

## legacy/aspen_top_ad_sr/classifier/train_fine_tune_aspen_top_ad_sr_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --conditional --num-cond 4 --num-classes 2 --warmup-epoch 0 --clip-inputs --save-tag fine_tune_aspen_top_ad_sr_m --pretrain-tag pretrain_m --fine-tune --size medium --epoch 5 --lr 5e-06 --lr-factor 1.0 --wd 1.0
```

## legacy/aspen_top_ad_sr/classifier/train_fine_tune_aspen_top_ad_sr_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr --conditional --num-cond 4 --num-classes 2 --warmup-epoch 0 --clip-inputs --save-tag fine_tune_aspen_top_ad_sr_s --pretrain-tag pretrain_s --fine-tune --size small --epoch 5 --lr 5e-06 --lr-factor 1.0 --wd 0.1
```

## legacy/aspen_top_ad_sr_hl/generator/evaluate_aspen_top_ad_sb_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr_hl --conditional --num-cond 4 --num-classes 1 --mode generator --save-tag aspen_top_ad_sb_m --size medium --batch 128
```

## legacy/aspen_top_ad_sr_hl/generator/evaluate_aspen_top_ad_sb_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr_hl --conditional --num-cond 4 --num-classes 1 --mode generator --save-tag aspen_top_ad_sb_s --size small --batch 128
```

## legacy/aspen_top_ad_sr_hl/generator/evaluate_fine_tune_aspen_top_ad_sb_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr_hl --conditional --num-cond 4 --num-classes 1 --mode generator --save-tag fine_tune_aspen_top_ad_sb_m --size medium --interaction --batch 256
```

## legacy/aspen_top_ad_sr_hl/generator/evaluate_fine_tune_aspen_top_ad_sb_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset aspen_top_ad_sr_hl --conditional --num-cond 4 --num-classes 1 --mode generator --save-tag fine_tune_aspen_top_ad_sb_s --size small --batch 128
```

## legacy/astro/classifier/evaluate_astro_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag astro_s --dataset astro --num-feat 3 --size small --use-add --num-classes 2 --batch 256
```

## legacy/atlas_ad/classifier/dataloader_np

```bash
omnilearned dataloader --dataset atlas_ad --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/atlas_ad/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset atlas_ad --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/atlas_flav/classifier/evaluate_atlas_flav_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --use-add --num-classes 4 --save-tag atlas_flav_m --size medium --num-add 18 --batch 1024
```

## legacy/atlas_flav/classifier/evaluate_atlas_flav_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --use-add --num-classes 4 --save-tag atlas_flav_s --size small --conditional --num-cond 4 --num-add 17 --batch 256
```

## legacy/atlas_flav/classifier/evaluate_fine_tune_atlas_flav_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --use-add --num-classes 4 --save-tag fine_tune_atlas_flav_s --size small --num-add 18 --batch 1024
```

## legacy/atlas_flav/ftag/train_atlas_flav_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --conditional --num-cond 4 --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 --mode ftag --iterations 2000 --wd 0.1 --save-tag atlas_flav_m --size medium --batch 256 --epoch 40 --lr 7e-05
```

## legacy/atlas_flav/ftag/train_atlas_flav_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --conditional --num-cond 4 --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 --mode ftag --iterations 2000 --wd 0.1 --save-tag atlas_flav_s --wandb --size small --batch 512 --epoch 30 --lr 5e-05
```

## legacy/atlas_flav/ftag/train_fine_tune_atlas_flav_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --conditional --num-cond 4 --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 --mode ftag --iterations 2000 --wd 0.1 --save-tag fine_tune_atlas_flav_m --pretrain-tag pretrain_m --fine-tune --size medium --batch 256 --epoch 30 --lr 5e-05 --lr-factor 1.0
```

## legacy/atlas_flav/ftag/train_fine_tune_atlas_flav_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset atlas_flav --interaction --conditional --num-cond 4 --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 --mode ftag --iterations 2000 --wd 0.1 --save-tag fine_tune_atlas_flav_s --pretrain-tag pretrain_s --fine-tune --size small --batch 512 --epoch 30 --lr 0.0005 --lr-factor 1.0
```

## legacy/camels/encoder/evaluate_quijote_encoder_local_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/datasets/camels_ae/val --save-tag quijote_encoder_local_s --dataset camels --num-feat 3 --num-part 5000 --num-latent 10 --size small --local-interaction --interaction-type astro --num-classes 1 --mode encoder --batch 4
```

## legacy/camels/encoder/train_camels_encoder_local_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_encoder_local_s --dataset camels --num-feat 3 --num-part 5000 --num-latent 10 --size small --local-interaction --interaction-type astro --num-classes 1 --mode encoder --batch 4 --iterations 60 --epoch 250 --lr 3e-05
```

## legacy/camels/generator/evaluate_camels_generation_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag camels_generation_s
```

## legacy/camels/generator/evaluate_fine_tune_camels_generation_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag fine_tune_camels_generation_s
```

## legacy/camels/generator/train_camels_gen_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --conditional --num-cond 2 --mode generator --batch 4 --save-tag camels_gen_s_600 --wandb --num-coord 3 --num-classes 1 --iterations 60 --epoch 200 --lr 5e-05 --nevts -1
```

## legacy/camels/generator/train_camels_generation_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --conditional --num-cond 2 --mode generator --batch 4 --save-tag camels_generation_s --num-coord 3 --num-classes 1 --iterations 60 --epoch 200 --lr 5e-05
```

## legacy/camels/generator/train_fine_tune_camels_generation_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --conditional --num-cond 2 --mode generator --batch 4 --save-tag fine_tune_camels_generation_s --pretrain-tag quijote_generation_local_s --wandb --fine-tune --num-coord 3 --num-classes 1 --iterations 60 --epoch 200 --lr 3e-05 --lr-factor 10.0
```

## legacy/camels/generator/train_quijote_gen_s_19651_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --conditional --num-cond 2 --mode generator --batch 4 --save-tag quijote_gen_s_19651_600 --pretrain-tag fine_tune_quijote_gen_s_19651 --wandb --fine-tune --interaction-type astro --num-classes 2 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 10.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/evaluate_camels_regression_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag camels_regression_s --size lite --local-interaction
```

## legacy/camels/regression/evaluate_camels_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag camels_s_100 --size small
```

## legacy/camels/regression/evaluate_camels_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag camels_s_600 --size small
```

## legacy/camels/regression/evaluate_fine_tune_camels_baseline_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_camels_baseline_s_600 --size small --local-interaction
```

## legacy/camels/regression/evaluate_fine_tune_camels_freeze_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_camels_freeze_s_600 --size small --local-interaction
```

## legacy/camels/regression/evaluate_fine_tune_camels_gaussian_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_camels_gaussian_s_600 --size small --local-interaction
```

## legacy/camels/regression/evaluate_fine_tune_camels_regression_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_camels_regression_s --size lite --local-interaction
```

## legacy/camels/regression/train_camels_regression_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag camels_regression_test_s --no-wandb --local-interaction --num-coord 3 --interaction-type astro --iterations 60 --epoch 200 --lr 5e-05 --wd 1.0 --persistent-workers --num-workers 16
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --interaction-type astro --batch 4 --iterations -1 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_100_fine_tune_camels_80

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_300 --pretrain-tag pretrain_s --wandb --fine-tune --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 30.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_300_fine_tune_camels_89

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_300 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 30.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_600 --pretrain-tag pretrain_s --wandb --fine-tune --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_baseline_s_600_fine_tune_camels_98

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_baseline_s_600 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations -1 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_100_fine_tune_camels_84

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_100_fine_tune_cosmos_98

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_100 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_300 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_300_fine_tune_camels_93

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_300 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_300_fine_tune_cosmos_104

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_300 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_600 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 10.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_600_fine_tune_camels_102

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_600 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 10.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_freeze_s_600_fine_tune_cosmos_110

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_freeze_s_600 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_100_fine_tune_camels_83

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_100_fine_tune_cosmos_97

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_300_fine_tune_camels_92

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_300_fine_tune_cosmos_103

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 19
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_600_fine_tune_camels_101

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 60 --epoch 100 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_600_fine_tune_camels_600_101

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_gaussian_s_600_fine_tune_cosmos_109

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 100 --epoch 300 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_quijote_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_quijote_s_100 --pretrain-tag fine_tune_quijote_freeze_s_19651 --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/camels/regression/train_fine_tune_camels_quijote_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_quijote_s_600 --pretrain-tag fine_tune_quijote_freeze_s_19651 --wandb --fine-tune --freeze --interaction-type astro --batch 4 --iterations -1 --epoch 150 --lr 1e-05 --lr-factor 10.0 --wd 1.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_regression_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_regression_s --pretrain-tag quijote_regression_s --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations -1 --epoch 200 --lr 2e-05 --lr-factor 5.0 --wd 0.0 --nevts -1
```

## legacy/camels/regression/train_fine_tune_camels_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_camels_s_19651 --pretrain-tag pretrain_s --fine-tune --batch 4 --epoch 150 --warmup-epoch 0 --lr 5e-06 --lr-factor 20.0 --wd 1.0
```

## legacy/camels/segmentation/evaluate_fine_tune_v_camels_baseline_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_camels_baseline_s_600
```

## legacy/camels/segmentation/evaluate_fine_tune_v_camels_freeze_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_camels_freeze_s_600
```

## legacy/camels/segmentation/evaluate_fine_tune_v_camels_gaussian_s_600

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_camels_gaussian_s_600
```

## legacy/camels/segmentation/evaluate_v_camels_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset camels --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag v_camels_s_100
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_100 --pretrain-tag pretrain_s --wandb --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_100_fine_tune_camels_81

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_100 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_300 --pretrain-tag pretrain_s --wandb --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_300_fine_tune_camels_90

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_300 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_600 --pretrain-tag pretrain_s --wandb --interaction-type astro --iterations 100 --epoch 30 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_baseline_s_600_fine_tune_camels_99

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_baseline_s_600 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 30 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_100 --pretrain-tag pretrain_s --wandb --freeze --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_100_fine_tune_camels_86

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_100 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_100_fine_tune_cosmos_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_100 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_300 --pretrain-tag pretrain_s --wandb --freeze --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_300_fine_tune_camels_95

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_300 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_300_fine_tune_cosmos_106

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_300 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_600 --pretrain-tag pretrain_s --wandb --freeze --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_600_fine_tune_camels_104

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_600 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_freeze_s_600_fine_tune_cosmos_112

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_freeze_s_600 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --freeze --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 1.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_100_fine_tune_camels_85

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 1.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_100_fine_tune_cosmos_99

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 6
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_300

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --wandb --freeze --interaction-type astro --iterations 100 --epoch 50 --lr 1e-05 --wd 1.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_300_fine_tune_camels_94

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 50 --lr 1e-05 --wd 1.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_300_fine_tune_cosmos_105

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_300 --pretrain-tag pretrain_s_gaussian --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts 19
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_600

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --freeze --interaction-type astro --iterations 100 --epoch 50 --lr 1e-05 --wd 0.1 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_600_fine_tune_camels_103

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 50 --lr 1e-05 --wd 0.1 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_600_fine_tune_camels_600_103

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 30 --lr 1e-05 --wd 1.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_gaussian_s_600_fine_tune_cosmos_111

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_gaussian_s_600 --pretrain-tag pretrain_s_gaussian --local-interaction --num-coord 3 --interaction-type astro --iterations 100 --epoch 20 --lr 1e-05 --wd 5.0 --nevts -1
```

## legacy/camels/segmentation/train_fine_tune_v_camels_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset camels --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --lr-factor 20.0 --save-tag fine_tune_v_camels_s_19651 --pretrain-tag pretrain_s --epoch 200 --warmup-epoch 0 --lr 5e-06 --wd 1.0
```

## legacy/camels_ae/generator/evaluate_camels_generation_ae_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag camels_generation_ae_s --dataset camels_ae --num-feat 3 --size small --num-coord 3 --conditional --num-cond 12 --num-classes 1 --mode generator --batch 4
```

## legacy/camels_ae/generator/train_camels_generation_ae_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag camels_generation_ae_s --dataset camels_ae --num-feat 3 --size small --num-coord 3 --conditional --num-cond 12 --num-classes 1 --mode generator --batch 4 --iterations 60 --epoch 200 --lr 5e-05
```

## legacy/cms_muon/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset cms_muon --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 64 --save-tag pretrain_l --size large
```

## legacy/cms_muon/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset cms_muon --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 64 --save-tag pretrain_m --size medium
```

## legacy/cms_top/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset cms_top --interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 64 --save-tag pretrain_l --size large --local-interaction
```

## legacy/cms_top/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset cms_top --interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 64 --save-tag pretrain_m --size medium
```

## legacy/dctr/classifier/evaluate_dctr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset dctr --interaction --conditional --num-cond 4 --use-pid --num-classes 2 --save-tag dctr_m --size medium
```

## legacy/dctr/classifier/evaluate_dctr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset dctr --interaction --conditional --num-cond 4 --use-pid --num-classes 2 --save-tag dctr_s --size small
```

## legacy/dctr/classifier/evaluate_fine_tune_dctr_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset dctr --interaction --conditional --num-cond 4 --use-pid --num-classes 2 --save-tag fine_tune_dctr_m --size medium
```

## legacy/dctr/classifier/evaluate_fine_tune_dctr_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset dctr --interaction --conditional --num-cond 4 --use-pid --num-classes 2 --save-tag fine_tune_dctr_s --size small
```

## legacy/dihiggs_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset dihiggs_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/dihiggs_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset dihiggs_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/gaussian/pretrain/train_pretrain_s_gaussian

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s_gaussian --dataset gaussian --size small --num-classes 10 --mode pretrain --batch 256 --iterations 1000 --epoch 10 --num-workers 32
```

## legacy/ggf_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset ggf_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_l --size large --batch 32
```

## legacy/ggf_dijet/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset ggf_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 128
```

## legacy/jetclass/classifier/train_fine_tune_jetclass_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_s --pretrain-tag pretrain_s --dataset jetclass --fine-tune --size small --interaction --local-interaction --use-pid --use-add --num-classes 10 --batch 256 --iterations 1000 --epoch 30 --lr 1e-05 --lr-factor 1.0 --wd 0.0
```

## legacy/jetht2017_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset jetht2017_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/jetht2017_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset jetht2017_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/jetht_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset jetht_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/jetht_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset jetht_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_l --size large --batch 32
```

## legacy/jetht_dijet/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset jetht_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_m --size medium --batch 64
```

## legacy/jetht_dijet/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset jetht_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 128
```

## legacy/jetnet150/generator/train_fine_tune_jetnet150_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset jetnet150 --conditional --num-classes 5 --mode generator --wd 0.0 --save-tag fine_tune_jetnet150_m --pretrain-tag pretrain_m --fine-tune --size medium --batch 32 --epoch 100 --warmup-epoch 0 --lr 5e-06 --lr-factor 5.0
```

## legacy/jetnet150/generator/train_fine_tune_jetnet150_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset jetnet150 --conditional --num-classes 5 --mode generator --wd 0.0 --save-tag fine_tune_jetnet150_s --pretrain-tag pretrain_s --fine-tune --size small --epoch 200 --lr 1e-05 --lr-factor 1.0
```

## legacy/jetnet150/generator/train_jetnet150_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset jetnet150 --conditional --num-classes 5 --mode generator --wd 0.0 --save-tag jetnet150_m --size medium --batch 32 --epoch 100 --warmup-epoch 0 --lr 5e-05
```

## legacy/jetnet150/generator/train_jetnet150_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset jetnet150 --conditional --num-classes 5 --mode generator --wd 0.0 --save-tag jetnet150_s --size small --epoch 200 --lr 5e-05
```

## legacy/jetnet30/encoder/evaluate_fine_tune_jetnet30_ae_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/datasets/jetnet30_ae/val --save-tag fine_tune_jetnet30_ae_s --dataset jetnet30 --num-part 30 --num-latent 10 --size small --interaction --local-interaction --num-classes 1 --mode encoder --batch 128
```

## legacy/jetnet30/encoder/train_fine_tune_jetnet30_ae_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet30_ae_s --pretrain-tag pretrain_s --dataset jetnet30 --fine-tune --num-part 30 --num-latent 10 --size small --interaction --local-interaction --num-classes 1 --mode encoder --batch 128 --epoch 1000 --lr 5e-05 --lr-factor 10.0 --wd 0.1
```

## legacy/jetnet30/generator/evaluate_fine_tune_jetnet30_gen_cond_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_jetnet30_gen_cond_s --dataset jetnet30 --size small --conditional --num-cond 3 --num-classes 1 --mode generator --batch 128
```

## legacy/jetnet30/generator/train_fine_tune_jetnet30_gen_cond_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --pretrain-tag pretrain_s --dataset jetnet30 --fine-tune --size small --num-classes 1 --mode generator --batch 128 --epoch 300 --save-tag fine_tune_jetnet30_gen_cond_s --conditional --num-cond 3 --lr 3e-05 --lr-factor 10.0
```

## legacy/jetnet30/generator/train_fine_tune_jetnet30_gen_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --pretrain-tag pretrain_s --dataset jetnet30 --fine-tune --size small --num-classes 1 --mode generator --batch 128 --epoch 300 --save-tag fine_tune_jetnet30_gen_s --lr 1e-05 --lr-factor 1.0 --wd 0.0
```

## legacy/jetnet30_ae/generator/evaluate_fine_tune_jetnet30_ae_gen_cond_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_jetnet30_ae_gen_cond_s --dataset jetnet30_ae --size small --conditional --num-cond 13 --num-classes 1 --mode generator --batch 128
```

## legacy/jetnet30_ae/generator/train_fine_tune_jetnet30_ae_gen_cond_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetnet30_ae_gen_cond_s --pretrain-tag pretrain_s --dataset jetnet30_ae --fine-tune --size small --conditional --num-cond 13 --num-classes 1 --mode generator --batch 128 --epoch 300 --lr 3e-05 --lr-factor 5.0
```

## legacy/lhco_ad/classifier/train_lhco_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag lhco_s --pretrain-tag pretrain_s --dataset lhco_ad --fine-tune --num-feat 6 --size small --interaction --local-interaction --conditional --num-cond 11 --num-classes 2 --batch 32 --epoch 20 --lr 1e-05 --lr-factor 10.0 --wd 0.0
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_0

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_0 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_1

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_1 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_2

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_2 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_3

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_3 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_4

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_4 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_5 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_6

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_6 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_7

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_7 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_8

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_8 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_full_s_9

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_full_s_9 --interaction
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_0

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_0
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_1

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_1
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_2

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_2
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_3

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_3
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_4

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_4
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_5
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_6

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_6
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_7

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_7
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_8

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_8
```

## legacy/microboone/classifier/evaluate_fine_tune_microboone_s_9

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag fine_tune_microboone_s_9
```

## legacy/microboone/classifier/evaluate_microboone_full_s_0

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_0 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_1

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_1 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_2

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_2 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_3

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_3 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_4

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_4 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_5 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_6

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_6 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_7

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_7 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_8

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_8 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_full_s_9

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_full_s_9 --interaction
```

## legacy/microboone/classifier/evaluate_microboone_s_0

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_0
```

## legacy/microboone/classifier/evaluate_microboone_s_1

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_1
```

## legacy/microboone/classifier/evaluate_microboone_s_2

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_2
```

## legacy/microboone/classifier/evaluate_microboone_s_3

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_3
```

## legacy/microboone/classifier/evaluate_microboone_s_4

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_4
```

## legacy/microboone/classifier/evaluate_microboone_s_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_5
```

## legacy/microboone/classifier/evaluate_microboone_s_6

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_6
```

## legacy/microboone/classifier/evaluate_microboone_s_7

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_7
```

## legacy/microboone/classifier/evaluate_microboone_s_8

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_8
```

## legacy/microboone/classifier/evaluate_microboone_s_9

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --batch 8 --save-tag microboone_s_9
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_0

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_0 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_1

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_1 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_2

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_2 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_3

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_3 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_4

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_4 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_5

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_5 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_6

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_6 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_7

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_7 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_8

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_8 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_full_s_9

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_full_s_9 --pretrain-tag pretrain_s_test --fine-tune --interaction --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_0

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_0 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_1

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_1 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_2

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_2 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_3

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_3 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_4

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_4 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_5

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_5 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_6

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_6 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_7

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_7 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_8

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_8 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_fine_tune_microboone_s_9

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag fine_tune_microboone_s_9 --pretrain-tag pretrain_s_test --fine-tune --epoch 15 --lr 5e-05 --lr-factor 20.0 --wd 3.0
```

## legacy/microboone/classifier/train_microboone_full_s_0

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_0 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_1

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_1 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_2

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_2 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_3

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_3 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_4

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_4 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_5

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_5 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_6

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_6 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_7

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_7 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_8

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_8 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_full_s_9

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_full_s_9 --interaction --epoch 25 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_0

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_0 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_1

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_1 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_2

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_2 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_3

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_3 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_4

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_4 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_5

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_5 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_6

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_6 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_7

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_7 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_8

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_8 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/microboone/classifier/train_microboone_s_9

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset microboone --num-feat 4 --size small --local-interaction --num-coord 3 --k 5 --interaction-type ot --num-classes 2 --mode classifier --batch 8 --save-tag microboone_s_9 --epoch 35 --lr 0.0001 --wd 1.0
```

## legacy/minerva/classifier/evaluate_minerva_ft_int_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset minerva --num-feat 4 --size small --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --num-classes 5 --mode classifier --batch 256 --save-tag minerva_ft_int_s --interaction --local-interaction
```

## legacy/minerva/classifier/evaluate_minerva_ft_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset minerva --num-feat 4 --size small --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --num-classes 5 --mode classifier --batch 256 --save-tag minerva_ft_s
```

## legacy/minerva/classifier/evaluate_minerva_int_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset minerva --num-feat 4 --size small --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --num-classes 5 --mode classifier --batch 256 --save-tag minerva_int_s --interaction --local-interaction
```

## legacy/minerva/classifier/evaluate_minerva_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset minerva --num-feat 4 --size small --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --num-classes 5 --mode classifier --batch 256 --save-tag minerva_s
```

## legacy/minerva/classifier/train_minerva

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva --size small --num-classes 4 --epoch 50 --lr 0.0001 --wd 0.0
```

## legacy/minerva/classifier/train_minerva_ft_int_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_ft_int_m --pretrain-tag pretrain_m --fine-tune --size medium --num-classes 5 --epoch 20 --lr 5e-06 --lr-factor 5.0 --wd 2.0
```

## legacy/minerva/classifier/train_minerva_ft_int_ot

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_ft_int_ot --pretrain-tag pretrain_s --fine-tune --size small --interaction --local-interaction --k 5 --interaction-type ot --num-classes 5 --epoch 15 --lr 5e-05 --lr-factor 5.0 --wd 2.0
```

## legacy/minerva/classifier/train_minerva_ft_int_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_ft_int_s --pretrain-tag pretrain_s --fine-tune --size small --interaction --local-interaction --num-classes 5 --epoch 15 --lr 5e-05 --lr-factor 5.0 --wd 2.0
```

## legacy/minerva/classifier/train_minerva_ft_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_ft_s --pretrain-tag pretrain_s --fine-tune --size small --num-classes 5 --epoch 15 --lr 5e-05 --lr-factor 5.0 --wd 2.0
```

## legacy/minerva/classifier/train_minerva_int_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_int_s --size small --interaction --local-interaction --num-classes 5 --epoch 30 --lr 5e-05 --wd 0.0
```

## legacy/minerva/classifier/train_minerva_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset minerva --num-feat 4 --conditional --num-cond 16 --use-pid --pid-dim 8 --use-add --num-add 5 --mode classifier --batch 512 --save-tag minerva_s --size small --num-classes 5 --epoch 30 --lr 5e-05 --wd 0.0
```

## legacy/pretrain/pretrain/train_pretrain_l

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset pretrain --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --mode pretrain --iterations 1000 --feature-drop 0.1 --save-tag pretrainv2_l --no-wandb --no-resuming --size large --batch 16 --val-iterations 1000 --epoch 750 --optim lion --lr 3e-05 --wd 0.1 --persistent-workers --num-workers 16 --prefetch-factor 2 --max-open-files 128
```

## legacy/pretrain/pretrain/train_pretrain_lite

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset pretrain --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --mode pretrain --iterations 1000 --feature-drop 0.1 --save-tag pretrain_lite --size small --batch 128 --epoch 750 --num-workers 32
```

## legacy/pretrain/pretrain/train_pretrain_m_test

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset pretrain --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --mode pretrain --iterations 1000 --feature-drop 0.1 --save-tag pretrain_m_test --size medium --batch 32 --epoch 250 --lr 5e-06 --wd 0.1 --num-workers 32
```

## legacy/pretrain/pretrain/train_pretrain_s_test

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset pretrain --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --mode pretrain --iterations 1000 --feature-drop 0.1 --save-tag pretrainv2_s --size small --batch 128 --val-iterations 500 --epoch 500 --lr 5e-05 --persistent-workers --num-workers 16 --prefetch-factor 2 --max-open-files 128 --resuming
```

## legacy/qcd600_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd600_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd600_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd600_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset qcd_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_m --size medium --batch 64
```

## legacy/qcd_dijet/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset qcd_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 64
```

## legacy/qcd_dijet/classifier/evaluate_pretrain_s_ad_140

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset qcd_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 128
```

## legacy/qcd_dijet_0/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_0 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_0/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_0 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_1/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_1 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_1/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_1 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_2/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_2 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_2/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_2 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_3/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_3 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_3/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_3 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_4/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_4 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_4/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_4 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_5/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_5 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_5/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_5 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_6/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_6 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_6/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_6 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_7/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_7 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_7/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_7 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_8/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_8 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_8/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_8 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qcd_dijet_9/classifier/dataloader_np

```bash
omnilearned dataloader --dataset qcd_dijet_9 --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/qcd_dijet_9/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset qcd_dijet_9 --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/qg/classifier/evaluate_fine_tune_qg_s_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_5 --dataset qg --size small --interaction --use-pid --num-classes 2
```

## legacy/qg/classifier/train_fine_tune_qg_l

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset qg --interaction --save-tag fine_tune_qg_l --pretrain-tag pretrain_l --fine-tune --size large --batch 8 --epoch 5 --lr 5e-06 --lr-factor 5.0 --wd 1.0
```

## legacy/qg/classifier/train_qg_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset qg --interaction --save-tag qg_s --size small --local-interaction --use-pid --epoch 15 --lr 0.0005 --wd 0.5
```

## legacy/quijote/encoder/train_quijote_encoder_local_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag quijote_encoder_local_s --dataset quijote --num-feat 3 --num-part 5000 --num-latent 10 --size small --local-interaction --interaction-type astro --num-classes 1 --mode encoder --batch 4 --iterations 60 --epoch 300 --lr 3e-05
```

## legacy/quijote/ftag/train_fine_tune_ftag_quijote_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_ftag_quijote_s_19651 --pretrain-tag pretrain_s --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode ftag --batch 8 --epoch 200 --warmup-epoch 0 --lr 1e-05 --lr-factor 20.0 --wd 1.0
```

## legacy/quijote/generator/evaluate_fine_tune_camels_quijote_s_19651

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag fine_tune_camels_quijote_s_19651 --dataset quijote --num-feat 3 --size small --conditional --num-cond 2 --num-classes 1 --mode generator --batch 8
```

## legacy/quijote/generator/train_quijote_gen_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_100 --k 20 --iterations 60 --epoch 200 --lr 5e-05 --nevts 6
```

## legacy/quijote/generator/train_quijote_gen_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_1000 --k 20 --iterations -1 --epoch 200 --lr 5e-05 --nevts 62
```

## legacy/quijote/generator/train_quijote_gen_s_1000_quijote_gen_70

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_1000 --iterations 60 --epoch 200 --lr 5e-05 --nevts 62
```

## legacy/quijote/generator/train_quijote_gen_s_100_quijote_gen_68

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_100 --iterations 60 --epoch 200 --lr 5e-05 --nevts 6
```

## legacy/quijote/generator/train_quijote_gen_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_19651 --k 20 --iterations -1 --epoch 200 --lr 5e-05 --nevts -1
```

## legacy/quijote/generator/train_quijote_gen_s_19651_quijote_gen_72

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_gen_s_19651 --iterations 60 --epoch 200 --lr 5e-05 --nevts -1
```

## legacy/quijote/generator/train_quijote_generation_local_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --wandb --num-feat 3 --size small --num-coord 3 --conditional --num-cond 2 --num-classes 1 --mode generator --batch 4 --save-tag quijote_generation_local_s --resuming --iterations 60 --epoch 500 --lr 3e-05
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_baseline_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_baseline_s_100 --k 20
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_freeze_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_freeze_s_100
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_freeze_s_100_eval_quijote_53

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_freeze_s_100 --k 20
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_gaussian_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_gaussian_s_100
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_gaussian_s_100_eval_quijote_69

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_gaussian_s_100 --k 20
```

## legacy/quijote/regression/evaluate_fine_tune_quijote_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag fine_tune_quijote_s_100 --k 20
```

## legacy/quijote/regression/evaluate_quijote_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --mode regression --batch 4 --save-tag quijote_s_100 --k 20
```

## legacy/quijote/regression/train_fine_tune_R_quijote_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_R_quijote_s_19651 --pretrain-tag pretrain_s --fine-tune --batch 2 --epoch 100 --warmup-epoch 0 --lr 1e-05 --lr-factor 20.0 --wd 1.0
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 1e-05 --lr-factor 2.0 --wd 0.1 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_1000 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_1000_fine_tune_quijote_large_89

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_1000 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_100_fine_tune_quijote_small_80

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 5e-06 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_19651 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_baseline_s_19651_fine_tune_quijote_large_98

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_baseline_s_19651 --pretrain-tag pretrain_s --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_100 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 5e-06 --lr-factor 2.0 --wd 0.1 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_1000 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_1000_fine_tune_quijote_83

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_1000 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 80 --lr 1e-05 --lr-factor 10.0 --wd 1.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_1000_fine_tune_quijote_large_91

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_1000 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_100_fine_tune_quijote_large_84

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 1e-05 --lr-factor 2.0 --wd 0.1 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_100_fine_tune_quijote_small_81

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_100 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 5e-06 --lr-factor 2.0 --wd 0.1 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_19651 --pretrain-tag pretrain_s --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.0
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_19651_fine_tune_quijote_94

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_19651 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_freeze_s_19651_fine_tune_quijote_large_102

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_freeze_s_19651 --pretrain-tag pretrain_s --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 20 --lr 1e-05 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_1000 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 200 --lr 1e-05 --lr-factor 2.0 --wd 0.1 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_1000_fine_tune_quijote_large_90

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_1000 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_100_fine_tune_quijote_small_82

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 60 --epoch 30 --lr 1e-05 --lr-factor 2.0 --wd 1.0 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_19651 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_gaussian_s_19651_fine_tune_quijote_large_101

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_gaussian_s_19651 --pretrain-tag pretrain_s_gaussian --wandb --fine-tune --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.1 --nevts -1
```

## legacy/quijote/regression/train_fine_tune_quijote_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_s_100 --pretrain-tag pretrain_s --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 60 --epoch 20 --lr 5e-06 --lr-factor 2.0 --wd 0.1 --nevts 6
```

## legacy/quijote/regression/train_fine_tune_quijote_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_s_1000 --pretrain-tag pretrain_s --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.0 --nevts 62
```

## legacy/quijote/regression/train_fine_tune_quijote_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag fine_tune_quijote_s_19651 --pretrain-tag pretrain_s --fine-tune --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 100 --lr 1e-05 --lr-factor 10.0 --wd 0.0
```

## legacy/quijote/regression/train_quijote_regression_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --num-feat 3 --size small --num-classes 2 --mode regression --save-tag quijote_regression_s --local-interaction --num-coord 3 --interaction-type astro --batch 4 --iterations 60 --epoch 300 --lr 3e-05 --wd 0.0 --nevts -1
```

## legacy/quijote/segmentation/evaluate_fine_tune_v_quijote_baseline_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_quijote_baseline_s_100 --k 20
```

## legacy/quijote/segmentation/evaluate_fine_tune_v_quijote_freeze_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_quijote_freeze_s_100
```

## legacy/quijote/segmentation/evaluate_fine_tune_v_quijote_freeze_s_100_eval_quijote_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_quijote_freeze_s_100 --k 20
```

## legacy/quijote/segmentation/evaluate_fine_tune_v_quijote_gaussian_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_quijote_gaussian_s_100 --k 20
```

## legacy/quijote/segmentation/evaluate_fine_tune_v_quijote_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag fine_tune_v_quijote_s_100 --k 20
```

## legacy/quijote/segmentation/evaluate_v_quijote_s_100

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset quijote --num-feat 3 --size small --local-interaction --num-coord 3 --interaction-type astro --num-classes 2 --num-gen-classes 3 --mode segmentation --batch 4 --save-tag v_quijote_s_100 --k 20
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_baseline_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_baseline_s_100 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 30 --lr-factor 10.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_baseline_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_baseline_s_1000 --pretrain-tag pretrain_s --wandb --resuming --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_baseline_s_1000_fine_tune_quijote_large_93

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_baseline_s_1000 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 0.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_baseline_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_baseline_s_19651 --pretrain-tag pretrain_s --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_baseline_s_19651_fine_tune_quijote_large_99

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_baseline_s_19651 --pretrain-tag pretrain_s --wandb --resuming --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_100 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 100 --epoch 30 --lr-factor 10.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_1000 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 300 --lr-factor 20.0 --wd 0.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_1000_fine_tune_quijote_87

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_1000 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_1000_fine_tune_quijote_large_95

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_1000 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 0.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_100_fine_tune_quijote_large_86

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_100 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 100 --epoch 20 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_100_fine_tune_quijote_small_86

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_100 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 40 --lr-factor 10.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_19651 --pretrain-tag pretrain_s --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 300 --lr-factor 20.0 --wd 0.0
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_19651_fine_tune_quijote_96

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_19651 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_freeze_s_19651_fine_tune_quijote_large_103

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_freeze_s_19651 --pretrain-tag pretrain_s --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 30 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_1000 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_1000_fine_tune_quijote_large_94

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_1000 --pretrain-tag pretrain_s_gaussian --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 0.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_100_fine_tune_quijote_small_85

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_100 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations 100 --epoch 30 --lr-factor 20.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_19651 --pretrain-tag pretrain_s_gaussian --wandb --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 300 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_gaussian_s_19651_fine_tune_quijote_large_104

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_gaussian_s_19651 --pretrain-tag pretrain_s_gaussian --wandb --freeze --local-interaction --num-coord 3 --k 20 --interaction-type astro --batch 2 --iterations -1 --epoch 200 --lr-factor 20.0 --wd 1.0 --nevts -1
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_s_100

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_s_100 --pretrain-tag pretrain_s --local-interaction --num-coord 3 --interaction-type astro --batch 2 --iterations 100 --epoch 30 --lr-factor 10.0 --wd 1.0 --nevts 6
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_s_1000

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_s_1000 --pretrain-tag pretrain_s --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 300 --lr-factor 20.0 --wd 0.0 --nevts 62
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_s_19651

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_s_19651 --pretrain-tag pretrain_s --local-interaction --num-coord 3 --interaction-type astro --batch 2 --epoch 300 --lr-factor 20.0 --wd 0.0
```

## legacy/quijote/segmentation/train_fine_tune_v_quijote_s_19651_train_m_263

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset quijote --fine-tune --num-feat 3 --size small --num-classes 2 --num-gen-classes 3 --mode segmentation --lr 1e-05 --save-tag fine_tune_v_quijote_s_19651 --pretrain-tag pretrain_s --batch 8 --epoch 300 --warmup-epoch 0 --lr-factor 10.0 --wd 1.0
```

## legacy/singlemuon_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset singlemuon_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/singlemuon_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset singlemuon_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/singlemuon_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset singlemuon_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_l --size large --batch 32
```

## legacy/singlemuon_dijet/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset singlemuon_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 128
```

## legacy/stop_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset stop_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/stop_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset stop_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/stop_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset stop_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/stop_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset stop_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/stopw_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset stopw_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/stopw_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset stopw_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/stopw_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset stopw_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/stopw_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset stopw_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top/classifier/evaluate_fine_tune_top_l_5

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset top --interaction --num-classes 2 --save-tag fine_tune_top_l_5 --size large --local-interaction --batch 8
```

## legacy/top/classifier/evaluate_fine_tune_top_m_test

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset top --interaction --num-classes 2 --save-tag fine_tune_top_m_test --size medium
```

## legacy/top/classifier/evaluate_fine_tune_top_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset top --interaction --num-classes 2 --save-tag fine_tune_top_s --size small
```

## legacy/top/classifier/evaluate_top_s_test

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --dataset top --interaction --num-classes 2 --save-tag top_s_test --size small
```

## legacy/top/classifier/train_camels

```bash
omnilearned train
```

## legacy/top/classifier/train_fine_tune_top_dense

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_dense --pretrain-tag pretrain_s --dataset top --fine-tune --size dense --epoch 10 --lr 1e-05 --lr-factor 5.0 --wd 0.0
```

## legacy/top/classifier/train_fine_tune_top_l

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --local-interaction --batch 8 --epoch 5 --lr 1e-06 --wd 10.0 --num-workers 4
```

## legacy/top/classifier/train_fine_tune_top_l_2

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_2 --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --batch 8 --epoch 5 --lr 1e-06 --lr-factor 10.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_l_3

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_3 --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --batch 8 --epoch 5 --lr 1e-06 --lr-factor 10.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_l_4

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_4 --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --batch 8 --epoch 5 --lr 1e-06 --lr-factor 10.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_l_5

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l_5 --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --batch 8 --epoch 5 --lr 1e-06 --lr-factor 10.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_l_train_l_14

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_l --pretrain-tag pretrain_l --dataset top --fine-tune --size large --interaction --batch 8 --epoch 5 --lr 1e-06 --lr-factor 10.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_lite

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_lite --pretrain-tag pretrainv2_s --dataset top --fine-tune --size small --interaction --local-interaction --epoch 10 --lr 1e-05 --lr-factor 5.0 --wd 0.0
```

## legacy/top/classifier/train_fine_tune_top_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --interaction --local-interaction --epoch 5 --lr 1e-06 --lr-factor 1.0 --wd 10.0
```

## legacy/top/classifier/train_fine_tune_top_m_test

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_test --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --interaction --epoch 5 --lr 1e-06 --lr-factor 1.0 --wd 10.0
```

## legacy/top/classifier/train_top_dense

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_dense --dataset top --size dense --epoch 10 --lr 1e-05 --lr-factor 5.0 --wd 0.0
```

## legacy/top/classifier/train_top_l

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_l --dataset top --size large --interaction --local-interaction --batch 8 --epoch 10 --lr 5e-06 --wd 10.0
```

## legacy/top/classifier/train_top_lite

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_lite --dataset top --size lite --epoch 15 --lr 0.0005 --wd 0.5
```

## legacy/top/classifier/train_top_m

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top --size medium --interaction --local-interaction --epoch 15 --lr 5e-05 --wd 0.5
```

## legacy/top/classifier/train_top_s

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_s --dataset top --size small --interaction --local-interaction --epoch 15 --lr 0.0005 --wd 0.5
```

## legacy/top/classifier/train_top_s_test

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_s_test --dataset top --size small --interaction --local-interaction --epoch 15 --lr 0.0005 --wd 0.5
```

## legacy/top0_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top0_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top0_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top0_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top0_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top0_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top0_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top0_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top1_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top1_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top1_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top1_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top1_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top1_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top1_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top1_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top2_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top2_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top2_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top2_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top2_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top2_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top2_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top2_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top3_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top3_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top3_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top3_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top3_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top3_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top3_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top3_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset top_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/top_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset top_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/top_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset top_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_l --size large --batch 32
```

## legacy/top_dijet/classifier/evaluate_pretrain_m

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset top_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_m --size medium --batch 64
```

## legacy/top_dijet/classifier/evaluate_pretrain_s

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --dataset top_dijet --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --save-tag pretrain_s --size small --batch 128
```

## legacy/topv_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset topv_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/topw_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset topw_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/topw_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset topw_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/toy/classifier/smoke

```bash
omnilearned train --dataset toy --path ./datasets --output_dir ./checkpoints --save-tag toy --mode classifier --size small --num-feat 4 --num-classes 2 --batch 2 --epoch 2 --iterations 2 --val-iterations 2 --num-workers 0 --no-persistent-workers --prefetch-factor 1 --precision fp32 --attention-backend auto --no-qk-norm --seed 42
```

## legacy/wjets_cr/classifier/dataloader_np

```bash
omnilearned dataloader --dataset wjets_cr --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/wjets_cr/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset wjets_cr --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/wjets_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset wjets_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/wjets_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset wjets_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/ww_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset ww_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/ww_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset ww_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/wz_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset wz_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/wz_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset wz_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/zjets_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset zjets_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/zjets_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset zjets_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## legacy/zz_dijet/classifier/dataloader_np

```bash
omnilearned dataloader --dataset zz_dijet --folder /pscratch/sd/v/vmikuni/datasets/
```

## legacy/zz_dijet/classifier/evaluate_pretrain_l

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --output_dir /pscratch/sd/v/vmikuni/Omnilearned/ --save-tag pretrain_l --dataset zz_dijet --size large --interaction --local-interaction --use-pid --use-add --use-event-loss --num-classes 210 --batch 32
```

## recovered/eval/pretrain_m_12

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset aspen --size medium --use-pid --use-add --use-event-loss --num-classes 210
```

## recovered/eval/fine_tune_jetclass_s_16

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_s --dataset jetclass --size small --use-pid --use-add --num-classes 10
```

## recovered/eval/fine_tune_class_top_s_21

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_s --dataset top --size small --num-classes 2
```

## recovered/eval/fine_tune_qg_m_5_24

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_5 --dataset qg --size medium --use-pid --num-classes 2
```

## recovered/eval/fine_tune_top_m_27

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --dataset jetclass --size medium --use-pid --use-add --num-classes 10
```

## recovered/eval/fine_tune_class_top_m_29

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_m --dataset top --size medium --num-classes 2
```

## recovered/eval/qcd_s_35

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qcd_s --dataset top --size small --use-pid --num-classes 2
```

## recovered/eval/top_m_5_37

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_5 --dataset top --size medium --num-classes 2
```

## recovered/eval/qg_m_5_49

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_5 --dataset qg --size medium --use-pid --num-classes 2
```

## recovered/train_m/pretrain_m_13

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m --dataset pretrain --wandb --size medium --use-pid --use-add --use-event-loss --num-classes 210 --mode pretrain --batch 32 --iterations 1000 --epoch 500 --lr 5e-06 --wd 0.1 --feature-drop 0.1 --num-workers 32
```

## recovered/train_m/pretrain_m_class_14

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_m_class --dataset pretrain --wandb --size medium --use-pid --use-add --use-event-loss --num-classes 210 --mode classifier --batch 32 --iterations 1000 --epoch 500 --lr 5e-05 --wd 0.1 --feature-drop 0.1 --num-workers 32
```

## recovered/train_m/fine_tune_class_top_m_17

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_top_m --pretrain-tag pretrain_m_class --dataset top --fine-tune --size medium --epoch 10 --warmup-epoch 0 --lr 3e-06 --lr-factor 10.0 --wd 0.01
```

## recovered/train_m/fine_tune_class_qg_m_18

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_class_qg_m --pretrain-tag pretrain_m_class --dataset qg --fine-tune --size medium --use-pid --epoch 10 --lr 1e-06 --lr-factor 5.0 --wd 1.0
```

## recovered/train_m/fine_tune_top_m_20

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --epoch 10 --warmup-epoch 0 --lr 3e-06 --lr-factor 10.0 --wd 0.01
```

## recovered/train_m/fine_tune_qg_m_21

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 10 --lr 1e-06 --lr-factor 5.0 --wd 10.0
```

## recovered/train_m/top_m_25

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m --dataset top --size medium --epoch 15 --lr 5e-05 --wd 0.5
```

## recovered/train_m/top_m_2_34

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_2 --dataset top --size medium --epoch 15 --lr 5e-05 --wd 0.5
```

## recovered/train_m/top_m_3_42

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_3 --dataset top --size medium --epoch 15 --lr 5e-05 --wd 0.5
```

## recovered/train_m/top_m_4_50

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_4 --dataset top --size medium --epoch 15 --lr 5e-05 --wd 0.5
```

## recovered/train_m/top_m_5_58

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag top_m_5 --dataset top --size medium --epoch 15 --lr 5e-05 --wd 0.5
```

## recovered/train_m/qg_m_69

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m --dataset qg --size medium --use-pid --epoch 15 --lr 5e-05 --wd 0.1
```

## recovered/train_m/qg_m_2_78

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_2 --dataset qg --size medium --use-pid --epoch 15 --lr 5e-05 --wd 0.1
```

## recovered/train_m/qg_m_3_87

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_3 --dataset qg --size medium --use-pid --epoch 15 --lr 5e-05 --wd 0.1
```

## recovered/train_m/qg_m_4_96

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_4 --dataset qg --size medium --use-pid --epoch 15 --lr 5e-05 --wd 0.1
```

## recovered/train_m/qg_m_5_105

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag qg_m_5 --dataset qg --size medium --use-pid --epoch 15 --lr 5e-05 --wd 0.1
```

## recovered/train_m/fine_tune_jetclass_m_119

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_m --pretrain-tag pretrain_m --dataset jetclass --fine-tune --size medium --use-pid --use-add --num-classes 10 --batch 32 --iterations 1000 --epoch 200 --lr 1e-06 --wd 10.0
```

## recovered/train_m/fine_tune_top_m_2_137

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_2 --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --epoch 5 --lr 1e-06 --wd 10.0
```

## recovered/train_m/fine_tune_top_m_3_145

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_3 --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --epoch 5 --lr 1e-06 --wd 10.0
```

## recovered/train_m/fine_tune_top_m_4_153

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_4 --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --epoch 5 --lr 1e-06 --wd 10.0
```

## recovered/train_m/fine_tune_top_m_5_161

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_m_5 --pretrain-tag pretrain_m --dataset top --fine-tune --size medium --epoch 5 --lr 1e-06 --wd 10.0
```

## recovered/train_m/fine_tune_qg_m_172

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 5 --lr 1e-06 --wd 1.0
```

## recovered/train_m/fine_tune_qg_m_2_181

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_2 --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 5 --lr 1e-06 --wd 1.0
```

## recovered/train_m/fine_tune_qg_m_3_190

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_3 --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 5 --lr 1e-06 --wd 1.0
```

## recovered/train_m/fine_tune_qg_m_4_199

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_4 --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 5 --lr 1e-06 --wd 1.0
```

## recovered/train_m/fine_tune_qg_m_5_208

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_m_5 --pretrain-tag pretrain_m --dataset qg --fine-tune --size medium --use-pid --epoch 5 --lr 1e-06 --wd 1.0
```

## recovered/ad/fine_tune_aspen_top_ad_sb_s_41

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_aspen_top_ad_sb_s --pretrain-tag pretrain_s --dataset aspen_top_ad_sb --fine-tune --size small --conditional --num-classes 1 --mode generator --batch 64 --epoch 30 --warmup-epoch 0 --lr 1e-05 --lr-factor 1.0 --wd 0.0
```

## recovered/eval/pretrain_s_class_8

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s_class --dataset aspen_top_ad_sr --size small --use-pid --use-add --zero-add --use-event-loss --num-classes 210
```

## recovered/eval/pretrain_s_10

```bash
omnilearned evaluate --input_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag pretrain_s --dataset cms_qcd --size small --use-pid --use-add --zero-add --use-event-loss --num-classes 210
```

## recovered/train_s/jetnet150_s_13

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag jetnet150_s --dataset jetnet150 --size small --interaction --conditional --num-classes 5 --mode generator --epoch 200 --warmup-epoch 0 --lr 5e-05 --wd 0.0
```

## recovered/train_s/fine_tune_jetclass_s_23

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_jetclass_s --pretrain-tag pretrain_s --dataset jetclass --fine-tune --size small --use-pid --use-add --num-classes 10 --batch 128 --iterations 1000 --epoch 50 --warmup-epoch 1 --lr 5e-06 --lr-factor 5.0 --wd 0.0
```

## recovered/train_s/fine_tune_top_s_2_48

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_2 --pretrain-tag pretrain_s --dataset top --fine-tune --size small --epoch 10 --warmup-epoch 1 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_top_s_3_57

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_3 --pretrain-tag pretrain_s --dataset top --fine-tune --size small --epoch 10 --warmup-epoch 1 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_top_s_4_66

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_4 --pretrain-tag pretrain_s --dataset top --fine-tune --size small --epoch 10 --warmup-epoch 1 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_top_s_5_75

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_top_s_5 --pretrain-tag pretrain_s --dataset top --fine-tune --size small --epoch 10 --warmup-epoch 1 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_qg_s_85

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s --pretrain-tag pretrain_s --dataset qg --fine-tune --size small --use-pid --epoch 10 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_qg_s_2_94

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_2 --pretrain-tag pretrain_s --dataset qg --fine-tune --size small --use-pid --epoch 10 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_qg_s_3_103

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_3 --pretrain-tag pretrain_s --dataset qg --fine-tune --size small --use-pid --epoch 10 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_qg_s_4_112

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_4 --pretrain-tag pretrain_s --dataset qg --fine-tune --size small --use-pid --epoch 10 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```

## recovered/train_s/fine_tune_qg_s_5_121

```bash
omnilearned train --output_dir /pscratch/sd/v/vmikuni/PET/checkpoints/ --save-tag fine_tune_qg_s_5 --pretrain-tag pretrain_s --dataset qg --fine-tune --size small --use-pid --epoch 10 --lr 5e-06 --lr-factor 5.0 --wd 0.1
```
