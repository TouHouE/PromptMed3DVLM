#!/bin/bash
# HF_HOME=
# Orignal language_model_name_or_path is set to "medicalai/CLinicalBERT"
# Notice, In `DEC_CLIPConfig`, the `hidden_size` default is 512, but when doing training, the `ModelArguments` is 768
export pretrained_model="/home/jovyan/shared/uc207pr4f57t9/cardiac/model/dcformer_vit"
export data_root="/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei"
export cur_model_name=CardiacSigLIP_Prompt_nnunet_E500
export output_dir="/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/$cur_model_name"
export eval_output_dir=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/SigLIP/$cur_model_name
# export output_dir="/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/debug"

deepspeed src/train/train_clip.py \
    --deepspeed ./scripts/zero2.json \
    --pretrained_model $pretrained_model/model.safetensors \
    --language_model_name_or_path medicalai/ClinicalBERT \
    --wb_name $cur_model_name \
    --vision_encoder "mask_prompt_dcformer" \
    --loss_type "sigmoid" \
    --data_root $data_root \
    --max_length 512 \
    --append_keys image_fg;image_fgs \
    --loader_type unet-med3d-fgcrop \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 500 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8

python src/eval/cardiac_eval_clip.py \
    --model_name_or_path $output_dir \
    --data_root $data_root \
    --save_output True \
    --output_dir $eval_output_dir \
    --max_length 512 \
    --shape_mode resize \
    --test_size 100 150 200 250 -1
