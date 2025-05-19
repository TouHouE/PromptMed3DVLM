#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export output_loc=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/MagicXin-Med3DVLM-Qwen-2.5-7B

accelerate launch --mixed_precision=bf16 --num_processes=0 src/eval/cardiac_eval_vqa.py \
    --model_name_or_path MagicXin/Med3DVLM-Qwen-2.5-7B \
    --vision_tower dcformer \
    --data_root ./data \
    --max_length 512 \
    --proj_out_num 256 \
    --top_p=.9 \
    --temperature=0 \
    --output_dir $output_loc