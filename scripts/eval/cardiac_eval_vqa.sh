#!/bin/bash
export HF_HOME=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface
export output_loc=/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/eval/Med3DVLM-Qwen-2.5-7B-LLMLoRA-Baseline-Resize-5Epoch
export model=/home/jovyan/shared/uc207pr4f57t9/cardiac/huggingface/hub/models--MagicXin--Med3DVLM-Qwen-2.5-7B/snapshots/8e5d303406bf5d2de48db089f027ab3d8966056c


python src/eval/cardiac_eval_vqa.py \
    --model_name_or_path $model \
    --vision_tower dcformer \
    --data_root ./data \
    --max_length 512 \
    --proj_out_num 256 \
    --top_p=.9 \
    --temperature=0 \
    --output_dir ./output/exp \
    --batch_size 64 \
    --test_size 5 \
    --shape_mode resize \
    --do_jpeg
