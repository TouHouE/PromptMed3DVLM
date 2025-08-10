#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export model_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP
export output_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/SigLIP
export data_root=/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei

# export model_name=$model_root/CardiacSigLIP_noPrompt_nnunet_E500
# export output_dir=$output_root/CardiacSigLIP_noPrompt_nnunet_E500
# export model_name=PSigLIP_KLD

# /home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/SigLIP/AbalationStudy_PSigLIP_noEp
# export model_name=AbalationStudy-PSigLIP-LargeRoILambda

export model_name=CardiacSigLIP_Prompt_nnunet_E500/current_11500
# export model_name=$model_root/CardiacSigLIP_Prompt_nnunet_E500/current_11500
# export output_dir=$output_root/CardiacSigLIP_Prompt_nnunet_E500_i11500

# export model_name=$model_root/CardiacSigLIP_Prompt_nnunet_E500
# export output_dir=$output_root/CardiacSigLIP_Prompt_nnunet_E500/Final

python src/eval/cardiac_eval_clip.py \
    --model_name_or_path $model_root/$model_name \
    --data_root $data_root \
    --loader_type unet-med3d-resize \
    --save_output True \
    --output_dir $output_root/$model_name/w_mask \
    --max_length 512 \
    --test_size 100 \
    --top_k 10 \
    --test_method recall \
    --do_mask_prompt \
    