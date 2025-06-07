#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export output_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput
export vision_encoder="/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/CardiacSigLIP_noPrompt_nnunet_E500/pretrained_ViT.bin"
export run_name="RetrainStage_0_CardiacSigLIP_noPrompt_nnunet_E500"

deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name $run_name \
    --vision_tower "dcformer" \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --model_type vlm_qwen \
    --pretrain_vision_model $vision_encoder \
    --mm_projector_type "mixer" \
    --vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --data_root ./data \
    --bf16 True \
    --output_dir $output_root/ReTrain/Stage0/$run_name \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4