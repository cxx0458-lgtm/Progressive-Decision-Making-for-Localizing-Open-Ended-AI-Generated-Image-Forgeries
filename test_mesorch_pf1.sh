base_dir="./eval_dir_mesorch_permutemamaba"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test.py \
    --model ProgressiveMesorch \
    --use_look_twice \
    --world_size 1 \
    --test_data_json "./test_dataset2.json" \
    --checkpoint_path "/home/bkai/cxx/uploadgithub/output_dir_mesorch_mamba_table1/" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --num_workers 4 \
    --f1_mode double \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log