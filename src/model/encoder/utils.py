from dataclasses import fields, is_dataclass

import transformers as HFT
# BUILTIN_CONFIG = HFT.PretrainedConfig()

def wrap_matched_keys(config_obj, config_class) -> dict[str, any]:
    """Wrap all keys in config_obj to config_class if not already."""
    if not isinstance(config_obj, dict):
        try:
            config_obj: dict[str, any] = vars(config_obj)
        except Exception as e:
            raise ValueError(f"Cannot convert config_obj to dict: {e}")
    if is_dataclass(config_class):        
        key_pool = set(field.name for field in fields(config_class))
    else:
        example_config = config_class()
        key_pool = set(example_config.to_dict().keys())
    filtered_config = {k: v for k, v in config_obj.items() if k in key_pool}
    return filtered_config