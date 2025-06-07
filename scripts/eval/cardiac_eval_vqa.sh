#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export output_loc=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/M3D-LaMed-Llama-2-7B
# export model=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-Resize-5Epoch/models/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-Resize-5Epoch
export model=GoodBaiBai88/M3D-LaMed-Llama-2-7B

python src/eval/cardiac_eval_vqa.py \
    --model_name_or_path $model \
    --vision_tower dcformer \
    --data_root ./data \
    --max_length 512 \
    --shape_mode resize \
    --input_size 32 256 256 \
    --axes_code SRA \
    --proj_out_num 256 \
    --top_p=.9 \
    --temperature=0 \
    --output_dir $output_loc \
    --batch_size 64 \
    --worker 8
