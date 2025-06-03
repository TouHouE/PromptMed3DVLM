#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
# export EXP=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-Resize-5Epoch
export EXP=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-d10-Resize-5Epoch

python src/utils/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --pretrain_vision_model $EXP/newest_vision_tower.bin \
    --vision_tower "dcformer" \
    --model_with_lora $EXP/model_with_lora.bin \
    --output_dir $EXP/models/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-d10-Resize-5Epoch \
    --fix_vision_tower_prefix True
