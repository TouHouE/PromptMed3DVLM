#!/bin/bash

deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name PromptMed3DVLM-Qwen-2.5-7B-lora \
    --vision_tower prompt_dcformer \
    --pretrain_vision_model_status dcformer \
    --model_size small \
    --model_name_or_path /home/jovyan/workspace/Med3DVLM/models/VLM \
    --model_type vlm_qwen \
    --pretrain_vision_model /home/jovyan/workspace/Med3DVLM/models/dcformer/pretrained_ViT.bin \
    --mm_projector_type "mixer" \
    --lora_enable True \
    --vision_select_layer -2 \
    --data_root ./data \
    --bf16 True \
    --output_dir ./output/PromptMed3DVLM-Qwen-2.5-7B-lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4