#!/bin/bash
python train_fast.py --extra_loss=1 --extra_w=1e-3 --n_deform_layer=1  --base_mae=196 --dataset_name='SHA' --mode='crop' --nThreads=4 --batch_size=2  --gpu_ids='0' --optimizer='adam' --start_eval_epoch=1 --max_epochs=1200 --lr=1e-4 --decay_rate=0.8 --name='CSRNet_deform_var_d1_s0_w1e-3_QRNF_decay_rate0.2_trainingcurve' --net_name='csrnet_deform_var' --eval_per_epoch=1 --model_name='/root/autodl-tmp/UCF_CC_50_output_ECA/MAE_196.8_MSE_322.53_Ep_991.pth'