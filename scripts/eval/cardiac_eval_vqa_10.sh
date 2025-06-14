#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export output_loc=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/ReTrain/RetrainStage_2_CardiacSigLIP_noPrompt_nnunet_E500
export model=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/ReTrain/Stage2/RetrainStage_2_CardiacSigLIP_noPrompt_nnunet_E500/models/RetrainStage_2_CardiacSigLIP_noPrompt_nnunet_E500
# export model=GoodBaiBai88/M3D-LaMed-Phi-3-4B

python src/eval/cardiac_eval_vqa.py \
    --model_name_or_path $model \
    --vision_tower dcformer \
    --data_root ./data \
    --max_length 512 \
    --shape_mode resize \
    --input_size 256 256 128 \
    --axes_code RAS \
    --proj_out_num 256 \
    --top_p=.9 \
    --temperature=0 \
    --output_dir $output_loc \
    --output_name retest_result.json \
    --batch_size 20 \
    --worker 8
