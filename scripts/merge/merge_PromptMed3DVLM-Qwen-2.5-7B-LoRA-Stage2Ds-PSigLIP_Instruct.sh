#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
# export EXP=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-Resize-5Epoch
export output_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput
export EXP=$output_root/PromptMed3DVLM-Qwen-2.5-7B-LoRA-Stage2Ds-PSigLIP_Instruct
# In this scripts, I'm not apply vidion tower pretrained weights is because of I directly store all of 
# stuff into the `model_with_lora.bin`

python src/utils/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --vision_tower "prompt_dcformer" \
    --model_with_lora $EXP/model_with_lora.bin \
    --output_dir $EXP/models/PromptMed3DVLM-Qwen-2.5-7B-LoRA-Stage2Ds-PSigLIP_Instruct_merged \
    --new_sep_tokens True