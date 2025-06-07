#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export model_name=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/CardiacSigLIP_noPrompt_nnunet_E500
export output_dir=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/SigLIP/CardiacSigLIP_noPrompt_nnunet_E500
export data_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei
python src/eval/cardiac_eval_clip.py \
    --model_name_or_path $model_name \
    --data_root $data_root \
    --save_output True \
    --output_dir $output_dir \
    --max_length 512 \
    --shape_mode resize \
    --test_size 100 150 200 250 -1