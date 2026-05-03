base_dir="./output_dir_mesorch_mamba2"
mkdir -p ${base_dir}

# 注意替换这里的 checkpoint 路径为之前protocol2下跑出来的baseline模型路径
Base_CKPT="/home/bkai/cxx/Mesorch-main/ckpt_AI/checkpoint-98.pth" 

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./train.py \
    --model ProgressiveMesorch \
    --finetune ${Base_CKPT} \
    --use_look_twice \
    --freeze_backbone \
    --world_size 2 \
    --find_unused_parameters \
    --batch_size 8 \
    --data_path ./balanced_dataset2.json \
    --epochs 10 \
    --lr 5e-5 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "/home/bkai/datasets/AutoSplice_JPEG100_test" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 1 \
    --seed 50 \
    --test_period 1 \
    --num_workers 4 \
    --lt_steps 2 \
    --lt_tau_low 0.30 \
    --lt_tau_high 0.80 \
    --lt_deep_dice_weight 1.0 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log