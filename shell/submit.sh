#python submit.py --save-tag pretrain_m --use-pid --use-add --batch 32 --lr 3e-5 --iterations 1000 --mode pretrain  --epoch 1000 --num-transf 12 --base-dim 512 --num-head 16 --use-event-loss --num-transf-heads 2 --wd 0.5 --resuming

#python submit.py --save-tag pretrain_s --use-pid --use-add --batch 128 --lr 5e-5 --iterations 1000 --mode pretrain  --epoch 1000 --num-transf 8 --base-dim 128 --num-head 8 --use-event-loss --num-transf-heads 2 --wd 0.1 --nodes 8 --resuming


#python submit.py --save-tag pretrain_m_class --use-pid --use-add --batch 32 --lr 3e-5 --iterations 1000 --mode classifier  --epoch 1000 --num-transf 12 --base-dim 512 --num-head 16 --use-event-loss --num-transf-heads 2 --wd 0.5 --resuming

#python submit.py --save-tag pretrain_s_class --use-pid --use-add --batch 128 --lr 5e-5 --iterations 1000 --mode classifier  --epoch 1000 --num-transf 8 --base-dim 128 --num-head 8 --use-event-loss --num-transf-heads 2 --wd 0.1 --nodes 8 --resuming

python submit.py --save-tag pretrain_l --use-pid --use-add --batch 8  --lr 1e-5 --iterations 1000 --mode pretrain  --epoch 750 --size large --use-event-loss --wd 0.1 --nodes 128

# python submit.py --save-tag pretrain_l_class --use-pid --use-add --batch 4 --lr 1e-5 --iterations 1000 --mode classifier  --epoch 1000 --num-transf 28 --base-dim 1024 --num-head 32 --use-event-loss --num-transf-heads 4 --wd 0.1 --nodes 128

