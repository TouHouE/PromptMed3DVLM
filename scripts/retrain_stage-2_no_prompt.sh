#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export OUTPUT_DIR=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput
export vision_encoder="/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/CardiacSigLIP_noPrompt_nnunet_E500/pretrained_ViT.bin"
# export mm_proj_dir=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput//ReTrain/Stage0/RetrainStage_0_CardiacSigLIP_noPrompt_nnunet_E500
export run_name=RetrainStage_2_CardiacSigLIP_noPrompt_nnunet_E500
export STAGE_1=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/ReTrain/Stage1/RetrainStage_1_CardiacSigLIP_noPrompt_nnunet_E500/model_with_lora.bin

deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name $run_name \
    --vision_tower "dcformer" \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --model_type vlm_qwen \
    --pretrain_vision_model $vision_encoder \
    --freeze_vision_tower False \
    --mm_projector_type "mixer" \
    --lora_enable True \
    --vision_select_layer -2 \
    --pretrain_mllm_with_lora $STAGE_1 \
    --data_root ./data \
    --shape_mode resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR/Retrain/Stage2/$run_name \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4