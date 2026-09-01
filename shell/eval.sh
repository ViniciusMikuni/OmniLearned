# load libs
module load pytorch/2.8.0

# for DDP
#export MASTER_ADDR=$(hostname)


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s_class --dataset aspen_top_ad_sr  --num-classes 210  --use-pid --use-add --num-transf 8 --base-dim 128 --use-event-loss --num-transf-heads 2 --zero-add"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_s --dataset cms_qcd  --num-classes 210  --use-pid --use-add --num-transf 8 --base-dim 128 --use-event-loss --num-transf-heads 2 --zero-add"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag pretrain_m --dataset aspen  --num-classes 210 --num-transf 12 --base-dim 512 --num-head 16 --use-event-loss --num-transf-heads 2 --use-pid --use-add"



#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_jetclass_s --dataset jetclass  --num-classes 10 --num-transf 8 --base-dim 128 --num-transf-heads 2 --use-pid --use-add"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_top_s --dataset top  --num-classes 2 --size small --interaction"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_qg_s_5 --dataset qg --use-pid --num-classes 2 --size small --interaction"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_class_top_s --dataset top  --num-classes 2 --num-transf 8 --base-dim 128 --num-transf-heads 2"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_top_m_test --dataset top  --num-classes 2  --size medium --interaction"
#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_qg_m_5 --dataset qg --use-pid  --num-classes 2  --num-transf 12 --base-dim 512 --num-head 16  --num-transf-heads 2"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_top_m --dataset jetclass  --num-classes 10 --num-transf 12 --base-dim 512 --num-head 16 --num-transf-heads 2 --use-pid --use-add"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_class_top_m --dataset top  --num-classes 2  --num-transf 12 --base-dim 512 --num-head 16  --num-transf-heads 2"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag top_s_test --dataset top  --num-classes 2 --size small --interaction"

#omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_top_l_5 --dataset top  --num-classes 2  --size large --interaction --local-interaction --batch 8

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag qcd_s --use-pid --dataset top  --num-classes 2 --num-transf 8 --base-dim 128 --num-transf-heads 2"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag top_m_5 --dataset top  --num-classes 2  --num-transf 12 --base-dim 512 --num-head 16  --num-transf-heads 2"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_dctr_m --dataset dctr  --num-classes 2  --size medium  --use-pid --interaction --conditional --num-cond 4"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_dctr_s --dataset dctr  --num-classes 2  --size small  --use-pid --interaction --conditional --num-cond 4"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag dctr_m --dataset dctr  --num-classes 2  --size medium  --use-pid --interaction --conditional --num-cond 4"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag dctr_s --dataset dctr  --num-classes 2  --size small  --use-pid --interaction --conditional --num-cond 4"


#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag qg_m_5 --dataset qg  --use-pid --num-classes 2  --num-transf 12 --base-dim 512 --num-head 16  --num-transf-heads 2"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag atlas_flav_m --dataset atlas_flav  --num-classes 4  --size medium  --interaction --use-add --num-add 18 --batch 1024"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag atlas_flav_s --dataset atlas_flav  --num-classes 4  --size small  --interaction --use-add --num-add 17 --batch 256 --conditional --num-cond 4"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag fine_tune_atlas_flav_s --dataset atlas_flav  --num-classes 4  --size small  --interaction --use-add --num-add 18 --batch 1024"

#cmd="omnilearned evaluate -i /pscratch/sd/v/vmikuni/PET/checkpoints/  --save-tag astro_s --dataset astro  --num-classes 2  --size small --use-add --batch 256 --num-feat 3"

dataset=quijote

for NEVT in 100; do
    # omnilearned evaluate \
    #     -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    #     --save-tag camels_s_${NEVT} \
    #     --dataset camels \
    #     --num-classes 2 \
    #     --size small \
    #     --batch 4 \
    #     --mode regression \
    #     -o /pscratch/sd/v/vmikuni/Omnilearned/ \
    #     --num-feat 3 \
    # 	--interaction-type astro \
    # 	--num-coord 3

    omnilearned evaluate \
		-i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
		--save-tag fine_tune_${dataset}_baseline_s_${NEVT} \
		--dataset ${dataset} \
		--num-classes 2 \
		--size small \
		--batch 4 \
		--mode regression \
		-o /pscratch/sd/v/vmikuni/Omnilearned/ \
		--num-feat 3 \
		--interaction-type astro \
		--local-interaction \
		--num-coord 3 \
                --k 20



    omnilearned evaluate \
		-i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
		--save-tag fine_tune_${dataset}_freeze_s_${NEVT} \
		--dataset ${dataset} \
		--num-classes 2 \
		--size small \
		--batch 4 \
		--mode regression \
		-o /pscratch/sd/v/vmikuni/Omnilearned/ \
		--num-feat 3 \
		--interaction-type astro \
		--local-interaction \
		--num-coord 3 \


    omnilearned evaluate \
		-i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
		--save-tag fine_tune_${dataset}_gaussian_s_${NEVT} \
		--dataset ${dataset} \
		--num-classes 2 \
		--size small \
		--batch 4 \
		--mode regression \
		-o /pscratch/sd/v/vmikuni/Omnilearned/ \
		--num-feat 3 \
		--interaction-type astro \
		--local-interaction \
		--num-coord 3 \

    
#     omnilearned evaluate \
#         -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
#         --save-tag v_camels_s_${NEVT} \
#         --dataset camels \
#         --num-classes 2 \
#         --size small \
#         --batch 4 \
#         --mode segmentation \
#         -o /pscratch/sd/v/vmikuni/Omnilearned/ \
#         --num-feat 3 \
#         --num-gen-classes 3 \
# 	--interaction-type astro \
# 	--local-interaction \
# 	--num-coord 3

    omnilearned evaluate \
        -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
        --save-tag fine_tune_v_${dataset}_freeze_s_${NEVT} \
        --dataset ${dataset} \
        --num-classes 2 \
        --size small \
        --batch 4 \
        --mode segmentation \
        -o /pscratch/sd/v/vmikuni/Omnilearned/ \
        --num-feat 3 \
        --num-gen-classes 3 \
	--interaction-type astro \
	--local-interaction \
	--num-coord 3

    omnilearned evaluate \
        -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
        --save-tag fine_tune_v_${dataset}_gaussian_s_${NEVT} \
        --dataset ${dataset} \
        --num-classes 2 \
        --size small \
        --batch 4 \
        --mode segmentation \
        -o /pscratch/sd/v/vmikuni/Omnilearned/ \
        --num-feat 3 \
        --num-gen-classes 3 \
	--interaction-type astro \
	--local-interaction \
	--num-coord 3 \
	--k 20

    omnilearned evaluate \
        -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
        --save-tag fine_tune_v_${dataset}_baseline_s_${NEVT} \
        --dataset ${dataset} \
        --num-classes 2 \
        --size small \
        --batch 4 \
        --mode segmentation \
        -o /pscratch/sd/v/vmikuni/Omnilearned/ \
        --num-feat 3 \
        --num-gen-classes 3 \
	--interaction-type astro \
	--local-interaction \
	--num-coord 3 \
	--k 20

    
done

# for NEVT in 100; do
#     omnilearned evaluate \
#         -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
#         --save-tag quijote_s_${NEVT} \
#         --dataset quijote \
#         --num-classes 2 \
#         --size small \
#         --batch 4 \
#         --mode regression \
#         -o /pscratch/sd/v/vmikuni/Omnilearned/ \
#         --num-feat 3 \
# 	--interaction-type astro \
# 	--local-interaction \
# 	--num-coord 3 \
# 	--k 20


#     omnilearned evaluate \
#         -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
#         --save-tag fine_tune_quijote_s_${NEVT} \
#         --dataset quijote \
#         --num-classes 2 \
#         --size small \
#         --batch 4 \
#         --mode regression \
#         -o /pscratch/sd/v/vmikuni/Omnilearned/ \
#         --num-feat 3 \
# 	--interaction-type astro \
# 	--local-interaction \
# 	--num-coord 3 \
# 	--k 20

#     omnilearned evaluate \
#         -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
#         --save-tag v_quijote_s_${NEVT} \
#         --dataset quijote \
#         --num-classes 2 \
#         --size small \
#         --batch 4 \
#         --mode segmentation \
#         -o /pscratch/sd/v/vmikuni/Omnilearned/ \
#         --num-feat 3 \
#         --num-gen-classes 3 \
# 	--interaction-type astro \
# 	--local-interaction \
# 	--num-coord 3 \
# 	--k 20

#     omnilearned evaluate \
#         -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
#         --save-tag fine_tune_v_quijote_s_${NEVT} \
#         --dataset quijote \
#         --num-classes 2 \
#         --size small \
#         --batch 4 \
#         --mode segmentation \
#         -o /pscratch/sd/v/vmikuni/Omnilearned/ \
#         --num-feat 3 \
#         --num-gen-classes 3 \
# 	--interaction-type astro \
# 	--local-interaction \
# 	--num-coord 3 \
# 	--k 20
# done

