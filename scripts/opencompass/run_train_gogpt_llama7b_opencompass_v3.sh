deepspeed --num_gpus=8  step3_train_sft_v2.py \
    --model_name_or_path /data/searchgpt/yq/GoGPT/vocab68k_pt_gogptv4_instruction_7b_output \
    --data_path /data/searchgpt/yq/GoGPT/data/demo \
    --bf16 True \
    --output_dir vocab68k_pt_gogptv4_instruction_7b_output_v3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --deepspeed "./configs/default_offlload_zero2.json" \
    --tf32 False \
    --model_max_length 1024

