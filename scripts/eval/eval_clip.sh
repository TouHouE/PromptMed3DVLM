#!/bin/bash
export model_name=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/debug
export output_dir=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/SigLIP/DCFormerSigLIP_noFineTune
python src/eval/eval_clip.py \
    --model_name_or_path $model_name \
    --data_root ./data \
    --save_output True \
    --output_dir .$output_dir \
    --max_length 512 \
    --shape_mode resize