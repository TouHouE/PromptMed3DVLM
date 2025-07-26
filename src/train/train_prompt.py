import logging
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional, Literal, Type, Sequence

import numpy as np
import torch
import wandb
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
try:
    import safetensors as HFS
    load_st = HFS.torch.load_file
except Exception:
    from safetensors import torch as HFS
    load_st = HFS.load_file

from src.dataset.prompt_dataset import PromptCardiacDataset
from src.model.llm.qwen import VLMQwenForCausalLM
from src.model.encoder.prompt_dcformer import PromptDCFormerConfig, PromptDCFormer, MaskPromptDCFormer, MaskPromptDCFormerClassifier 
from src.train.trainer import MLLMTrainer, PromptTrainer


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    tot_table = dict()
    train_table = dict()
    for name, param in model.named_parameters():
        second_name = name.split('.')[1]
        all_param += param.numel()
        tot_table[second_name] = tot_table.get(second_name, 0) + param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            train_table[second_name] = train_table.get(second_name, 0) + param.numel()
        # print(f"{name}: requires_grad={param.requires_grad}, numel={param.numel()}")
    print(f'Module Name || Module Trainable(MB) || Pecentage Trainable(%)')
    for key in tot_table.keys():
        trainable = (2 * train_table.get(key, 0)) / (1024 ** 2)
        percent = 100 * (train_table.get(key, 0) / tot_table[key])
        print(f'{key:12} || {float(trainable):18.4f} || {float(percent):.4f}')
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="prompt_dcformer")
    freeze_dcformer: bool = field(default=True)    # For Vision Encoder
    freeze_prompt_encoder: bool = field(default=False)  # For my custom module.
    pretrained_vision_encoder: Optional[str] = field(default=None)
    pretrained_status: str = field(default='prompt_siglip')
    num_class: int = field(default=512)
    # image
    input_size: tuple = field(default=(256, 256, 128))
    patch_size: int = field(default=(16, 16, 16))
    dim: int = field(default=768)
    depth: int = field(default=12)        

@dataclass
class DataArguments:
    # shape_mode: str = field(default='resize')
    loader_type: str = field(default='unet-med3d-resize')            
    move_to_cuda: bool = field(default=False, metadata={'help': 'Setting monai.transforms.EnsureType to cuda(True) or cpu(False)'})
    data_root: str = field(
        default="./data/", metadata={"help": "Root directory for all data."}
    )
    fold: int = field(default=0)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,  # 512
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./output/Med3DVLM-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    need2return = False

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ["mm_projector", "embed_tokens", "embeddings"]

        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        need2return = True

    if getattr(trainer.args, 'tune_vision_encoder', False):
        keys2match = ['vision_tower']
        weight2save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys2match
        )
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank > 0:
            return

        if current_folder.startswith("checkpoint-"):
            visual_encoder_folder = os.path.join(parent_folder, "visual_encoder")
            os.makedirs(visual_encoder_folder, exist_ok=True)
            torch.save(
                weight2save,
                os.path.join(visual_encoder_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(
                weight2save, os.path.join(output_dir, f"visual_encoder.bin")
            )
            need2return = True

    if need2return:
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "mm_projector",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def get_trainable_parameter_when_lora(_lora_model) -> list[str]:
    m2save: list[str] = list()
    for name, param in _lora_model.named_parameters():
        if 'lora' in name:
            continue

        if param.requires_grad:
            m2save.append(name)
    return m2save


@dataclass
class DataCollator:    
    def __init__(self, keys=None, mapping_keys=None, append_keys_pair: list[tuple[str, str]]=None):        
        if keys is None:
            self.keys = ["image", "mask", 'label', 'image-file', 'label-file']
        if mapping_keys is None:
            self.dst_keys=['images', 'masks', 'labels', 'image-file', 'label-file']

        assert len(self.keys) == len(self.dst_keys), f"keys({len(self.keys)} and dst_keys({len(self.dst_keys)}) must have the same length."
        if append_keys_pair is not None:
            for (key_in_ds, key_in_model) in append_keys_pair:
                self.keys.append(key_in_ds)
                self.dst_keys.append(key_in_model)
        print(f'All of key come from Dataset: {self.keys}')
        print(f'All of key apply into Model: {self.dst_keys}')

    def __call__(self, batch: list) -> dict:        
        return_dict: dict[str, torch.Tensor | list[str]] = {
            dst_key: [b[key] for b in batch] for key, dst_key in zip(self.keys, self.dst_keys)
        }
        for key, list_value in return_dict.items():
            list_value: torch.Tensor | list[str]
            if torch.is_tensor(list_value[0]):
                return_dict[key] = torch.stack(list_value)        
        return return_dict


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.input_size = model_args.input_size
    data_args.num_class = model_args.num_class
    # Define and add special tokens

    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    model = MaskPromptDCFormerClassifier(PromptDCFormerConfig.small_config(input_size=model_args.input_size))
    model.config.use_cache = False

    if model_args.pretrained_vision_encoder is not None and model_args.pretrained_status == 'prompt_siglip':
        ckpt = load_st(model_args.pretrained_vision_encoder)
        ckpt = {k.replace('vision_encoder.', ''): v for k,v in ckpt.items() if 'vision_encoder' in k}
        model.mask_prompt_dcformer.load_state_dict(ckpt, strict=True)
    elif model_args.pretrained_vision_encoder is not None and model_args.pretrained_status == 'prompt_dcformer':
        ckpt = torch.load(model_args.pretrained_vision_encoder, map_location='cpu')
        model.mask_prompt_dcformer.load_state_dict(ckpt, strict=True)
    elif model_args.pretrained_vision_encoder is not None and model_args.pretrained_status == 'dcformer':
        ckpt = torch.load(model_args.pretrained_vision_encoder, map_location='cpu')
        model.mask_prompt_dcformer.load_dcformer_state(ckpt)


    if model_args.freeze_dcformer:
        model.freeze_dcformer()
    if model_args.freeze_prompt_encoder:
        model.freeze_prompt_encoder()
    
    if is_rank_zero():
        print_trainable_parameters(model)
            
    train_dataset = PromptCardiacDataset(data_args)        
    data_collator = DataCollator()

    rank0_print("=" * 20 + " Training " + "=" * 20)
    trainer = PromptTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if is_rank_zero():
        wandb.login()
        wandb.init(project="PromptClassifier", name=model_args.wb_name, config={
            'model': vars(model_args),
            'data': vars(data_args),
            "training": training_args
        })

    if os.path.exists(training_args.output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(training_args.output_dir, d))
            ],
            key=lambda x: int(x.split("-")[-1]) if "-" in x else 0,
        )
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            resume_ckpt = os.path.join(training_args.output_dir, last_checkpoint)
            rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
            # try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
            # except Exception as E:
            #     trainer.train(resume_from_checkpoint='/home/jovyan/shared/uc207pr4f57t9/cardiac/RunOutput/PromptMed3DVLM-Qwen-2.5-7B-LoRA-BaselineDS-5Epoch/checkpoint-1300')
        else:
            trainer.train()
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if training_args.lora_enable or training_args.tune_vision_encoder:
        state_dict_with_lora = model.state_dict()   # Save all parameter into `model_with_lora.bin`
        torch.save(
            state_dict_with_lora,
            os.path.join(training_args.output_dir, "model_with_lora.bin"),
        )
        state_dict_vision_encoder = model.get_model().vision_tower.state_dict()
        torch.save(
            state_dict_vision_encoder,
            os.path.join(training_args.output_dir, 'newest_vision_tower.bin')
        )
        model_args.vision_tower_config.to_json_file(os.path.join(training_args.output_dir, 'vision_tower_config.json'))
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    if is_rank_zero():
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
