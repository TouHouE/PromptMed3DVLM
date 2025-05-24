#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export OUTPUT_DIR=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-5Epoch

deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-5Epoch \
    --vision_tower "dcformer" \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --model_type vlm_qwen \
    --pretrain_vision_model /home/jovyan/shared/uc207pr4f57t9/cardiac/model/dcformer_vit/pretrained_ViT.bin \
    --mm_projector_type "mixer" \
    --tune_mm_mlp_adapter True \
    --lora_enable True \
    --vision_select_layer -2 \
    --data_root ./data \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8