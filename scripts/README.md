/home/searchgpt/data/train_0.5M_CN/Belle_open_source_0.5M.json


usage: torchrun [-h] [--nnodes NNODES] [--nproc-per-node NPROC_PER_NODE] [--rdzv-backend RDZV_BACKEND] [--rdzv-endpoint RDZV_ENDPOINT] [--rdzv-id RDZV_ID] [--rdzv-conf RDZV_CONF] [--standalone] [--max-restarts MAX_RESTARTS]
                [--monitor-interval MONITOR_INTERVAL] [--start-method {spawn,fork,forkserver}] [--role ROLE] [-m] [--no-python] [--run-path] [--log-dir LOG_DIR] [-r REDIRECTS] [-t TEE] [--node-rank NODE_RANK] [--master-addr MASTER_ADDR]
                [--master-port MASTER_PORT] [--local-addr LOCAL_ADDR]
                training_script ...
torchrun: error: the following arguments are required: training_script, training_script_args


torchrun --nproc_per_node=2 train_7b_local_ds_multi.py

torchrun --nproc_per_node=2 train_13b_local_ds.py

--rdzv_id=456 --rdzv_backend=c10d

torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0  --master-addr=10.208.63.28  --master-port=29500 train_7b_local_ds_multi.py 




export NCCL_SOCKET_IFNAME=eno;NCCL_IB_DISABLE=1;export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO;NCCL_DEBUG_SUBSYS=ALL torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0  --master-addr=10.208.63.28  --master-port=29500 train_7b_local_ds_multi.py 



export NCCL_SOCKET_IFNAME=eno;NCCL_IB_DISABLE=1;export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO;NCCL_DEBUG_SUBSYS=ALL torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1  --master-addr=10.208.63.28  --master-port=29500 train_7b_local_ds_multi.py 



export NCCL_SOCKET_IFNAME=eno1,eno2;NCCL_IB_DISABLE=1;export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO;NCCL_DEBUG_SUBSYS=ALL torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0  --master-addr=10.208.63.28  --master-port=29500 train_13b_local_ds_multi.py

export NCCL_SOCKET_IFNAME=eno1,eno2;NCCL_IB_DISABLE=1;export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO;NCCL_DEBUG_SUBSYS=ALL torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1  --master-addr=10.208.63.28  --master-port=29500 train_13b_local_ds_multi.py 




Toolkit:  Installed in /usr/local/cuda-11.7/

Please make sure that
 -   PATH includes /usr/local/cuda-11.7/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.7/lib64, or, add /usr/local/cuda-11.7/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.7/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall



export PATH=/usr/local/cuda-11.7/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/usr/local/cuda-11.7/




DATASET="LinkSoul/instruction_merge_set"

DATA_CACHE_PATH="hf_datasets_cache"
MODEL_PATH="/PATH/TO/TRANSFORMERS/VERSION/LLAMA2"

output_dir="./checkpoints_llama2"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_port=25003 \
        train.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATASET} \
        --data_cache_path ${DATA_CACHE_PATH} \
        --bf16 True \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 1200 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True











>>> dataset=datasets.load_dataset('json',data_files='/data/searchgpt/data/instruction_data/instruction_merge_setsample_df_532k.json')





DATASET="/data/searchgpt/data/instruction_data/instruction_merge_setsample_df_532k.json"
DATA_CACHE_PATH="hf_datasets_cache"
MODEL_PATH="/data/searchgpt/yq/Firefly/output/gowizardlm"

output_dir="./checkpoints_llama2"

torchrun --nnodes=1 --nproc_per_node=8 \
    --master_port=25003 \
        train.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATASET} \
        --data_cache_path ${DATA_CACHE_PATH} \
        --bf16 True \
        --output_dir ${output_dir} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy 'no' \
        --save_strategy 'steps' \
        --save_steps 1200 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True