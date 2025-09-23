# CardiacAgent: A Multimodal Agent for Cardiac CT Analysis



Accurate cardiovascular disease diagnosis is essential for precision medicine, but current general-purpose medical AI models struggle with cardiac CT images due to challenges such as detail loss, insufficient 3D analysis accuracy, and an inability to interpret complex anatomical structures. To overcome these limitations, this study introduces CardiacAgent, an innovative multimodal agent framework designed to significantly improve the accuracy, consistency, and scalability of cardiac CT analysis. The framework incorporates three key innovations: a Prompt SigLIP visual prompting mechanism to help the model precisely focus on critical regions like the coronary arteries, effectively mitigating detail loss; a Prompt Multimodal Large Language Model (Prompt MLLM) integrated with a specialized Cardiac Healthcare Toolbox to generate quantitative reports and provide precise diagnostic Q\&A; and an automated data synthesis pipeline to address the shortage of high-quality image-report pairs. Experimental results demonstrate our framework's superior performance across multiple tasks. In image-text retrieval, our model outperformed the baseline. In medical report generation and Vision Question Answering (VQA), the Prompt MLLM achieved ROUGE scores of 87.72 and 64.88, respectively, surpassing Med3DVLM and LaMed. Furthermore, fine-tuning with synthesized data was shown to effectively improve model performance. In conclusion, by combining precise visual feature extraction, an enhanced language model, and a dataset augmented with synthetic data, CardiacAgent provides a comprehensive and efficient solution for cardiac CT image analysis.

## Requirements
* Python>=3.10.12
* torch>=2.5.0
* monai==1.2.0
* transformers==4.45.1
* deepspeed==0.16.3

## Installation
First, clone the repository to your local machine:
```bash
git clone https://github.com/TouHouE/PromptMed3DVLM.git
cd PromptMed3DVLM
```
To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

You need to set the `PYTHONPATH` environment variable to the root directory of the project. You can do this by running the following command in your terminal:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```
or 
```bash
PYTHONPATH=. sh scripts/...
```

## Datasets

### Data Path Configuration
A critical step before running any training or inference is to configure the data paths correctly. The data loading mechanism, specifically the `load_make_sure_exists` function in `src/dataset/utils/myio.py`, uses hardcoded paths to locate the image and label files.

The data JSON files should contain entries with at least an `"image"` key, and optionally a `"label"` key, for example:
```json
{
  "image": "the_image_file_name.nii.gz",
  "label": "the_label_file_name.nii.gz",
  ...
}
```
The `load_make_sure_exists` function constructs the full path by combining roots and subdirectories defined in `DEF_ROOT_LIST` and `DEF_MID_LIST` respectively. It searches for the image file in `<root>/<mid_path>/<image_file_name>`.

- If the image file is not found in any of the specified paths, the entire data entry is skipped (the function returns `None`).
- If the `"label"` key is present but the corresponding file is not found, the `"label"` key is removed from the data entry, but the entry is still used.

**You must modify `DEF_ROOT_LIST` and `DEF_MID_LIST` in `src/dataset/utils/myio.py` to match your dataset locations.**

```python
# src/dataset/utils/myio.py

DEF_ROOT_LIST: Final[list[str]] = ['/path/to/your/dataset/root1', '/path/to/your/dataset/root2']
DEF_MID_LIST: Final[list[str]] = ['your_subfolder1', 'your_subfolder2']
```

While some scripts support a `--data_root` argument to override the data root directory, it is recommended to modify `DEF_ROOT_LIST` and `DEF_MID_LIST` directly in the script for consistency.

## Training

The training process is divided into several stages.

### 1. Prompt SigLIP Training
This stage trains the visual encoder with a sigmoid loss.

To start the training, run the following script. Make sure to modify the paths in the script to your environment.
```bash
sh scripts/siglip/prompt_siglip.sh
```

### 2. VLM Fine-tuning
This stage fine-tunes the Vision-Language Model. There are several scripts available for this stage, depending on the specific configuration you want to use.

Example script:
```bash
sh scripts/finetune_after_stage2/vqa_lora_prompt.sh
```
This script will fine-tune the model using LoRA. You can find other scripts in the `scripts/finetune_after_stage2/` directory for different settings.

### 3. Merging LoRA Weights
After fine-tuning with LoRA, you can merge the LoRA weights with the base model.

Example script:
```bash
sh scripts/merge/merge_lora_weights_and_save_hf_model.sh
```
This will save the merged model in the specified output directory.

## Evaluation and Inference

### Evaluation
To run evaluation on your trained model, use the `infer.py` script.

```bash
python infer.py \
    --model_name /path/to/your/model \
    --data_json_path /path/to/your/evaluation_data.json \
    --output_dir /path/to/your/output_directory \
    --output_name your_evaluation_results.json
```
The `infer.py` script will save the evaluation results in the specified output directory. After running the inference, you can use `more_indicator.py` to compute additional metrics like GREEN and RaTEScore.

```bash
python more_indicator.py --eval_path /path/to/your/output_directory/your_evaluation_results.csv
```