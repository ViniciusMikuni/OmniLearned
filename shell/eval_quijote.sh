# load libs
module load pytorch/2.8.0

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

    # omnilearned evaluate \
    # 		-i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    # 		--save-tag fine_tune_${dataset}_baseline_s_${NEVT} \
    # 		--dataset ${dataset} \
    # 		--num-classes 2 \
    # 		--size small \
    # 		--batch 4 \
    # 		--mode regression \
    # 		-o /pscratch/sd/v/vmikuni/Omnilearned/ \
    # 		--num-feat 3 \
    # 		--interaction-type astro \
    # 		--local-interaction \
    # 		--num-coord 3 \
    #             --k 20



    # omnilearned evaluate \
    # 		-i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    # 		--save-tag fine_tune_${dataset}_freeze_s_${NEVT} \
    # 		--dataset ${dataset} \
    # 		--num-classes 2 \
    # 		--size small \
    # 		--batch 4 \
    # 		--mode regression \
    # 		-o /pscratch/sd/v/vmikuni/Omnilearned/ \
    # 		--num-feat 3 \
    # 		--interaction-type astro \
    # 		--local-interaction \
    # 		--num-coord 3 \
    # 		--k 20


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
	--k 20
	
    
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

    # omnilearned evaluate \
    #     -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    #     --save-tag fine_tune_v_${dataset}_freeze_s_${NEVT} \
    #     --dataset ${dataset} \
    #     --num-classes 2 \
    #     --size small \
    #     --batch 4 \
    #     --mode segmentation \
    #     -o /pscratch/sd/v/vmikuni/Omnilearned/ \
    #     --num-feat 3 \
    #     --num-gen-classes 3 \
    # 	--interaction-type astro \
    # 	--local-interaction \
    # 	--num-coord 3 \
    # 	--k 20

    # omnilearned evaluate \
    #     -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    #     --save-tag fine_tune_v_${dataset}_gaussian_s_${NEVT} \
    #     --dataset ${dataset} \
    #     --num-classes 2 \
    #     --size small \
    #     --batch 4 \
    #     --mode segmentation \
    #     -o /pscratch/sd/v/vmikuni/Omnilearned/ \
    #     --num-feat 3 \
    #     --num-gen-classes 3 \
    # 	--interaction-type astro \
    # 	--local-interaction \
    # 	--num-coord 3 \
    # 	--k 20

    # omnilearned evaluate \
    #     -i /pscratch/sd/v/vmikuni/PET/checkpoints/ \
    #     --save-tag fine_tune_v_${dataset}_baseline_s_${NEVT} \
    #     --dataset ${dataset} \
    #     --num-classes 2 \
    #     --size small \
    #     --batch 4 \
    #     --mode segmentation \
    #     -o /pscratch/sd/v/vmikuni/Omnilearned/ \
    #     --num-feat 3 \
    #     --num-gen-classes 3 \
    # 	--interaction-type astro \
    # 	--local-interaction \
    # 	--num-coord 3 \
    # 	--k 20

    
done

