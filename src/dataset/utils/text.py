import os
os.environ['HF_HOME'] = r"D:\huggingface"
import torch
import transformers as HFT
IGNORE_TOKEN_ID = -100

def get_mask(image_path: str) -> torch.Tensor:
    """
    This tools can segmentation all of substructure in the heart, e.g.: left, right ventricle, left, right atrium, left ventricular myocardium, coronary arteries, aorta, and plaque on coronary arteries
    Args:
        image_path: the cache path of the image.

    Returns:
        the mask of the provided image.
    """


def making_textual_input_label(fully_chat, tokenizer: HFT.PreTrainedTokenizer, args):
    while fully_chat[-1]['role'] != 'assistant':
        fully_chat.pop(-1)

    fully_content = tokenizer.apply_chat_template(fully_chat, add_generation_prompt=False, tokenize=False)
    prompts = tokenizer.apply_chat_template(fully_chat[:-1], add_generation_prompt=True, tokenize=False)
    fully_content_ids = tokenizer(fully_content, return_tensors='pt')
    prompt_ids = tokenizer(prompts, return_tensors='pt')

def get_im_start_end(tokenizer):
    if (im_start := getattr(tokenizer, 'im_start_id', None)) is None:
        im_start = tokenizer("<|im_start|>").input_ids[0]

    if (im_end := getattr(tokenizer, 'im_end_id', None)) is None:
        im_end = tokenizer("<|im_end|>").input_ids[0]
    return {
        'im_start': im_start,
        'im_end': im_end,
    }



def preprocess(
    sources,
    tokenizer: HFT.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    im_map = get_im_start_end(tokenizer)
    im_start = im_map['im_start']
    im_end = im_map['im_end']
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        """
            :variable system: 
            <|im_start|>system[\n]
            <system token><|im_end|>[\n]
        """
        input_id += system
        # the First <|im_start|> and the <|im_end|>, new line should be learned.
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        image_token_is_added = False
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if sentence['role'] == 'user' and not image_token_is_added:
                sentence['value'] = '<im_patch>' * 256 + "\n" + sentence['value']
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            print(f"role: {role}")
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def random_text():
    from string import ascii_letters
    import random
    return "".join(random.choices(ascii_letters, k=random.randint(12, 33)))

if __name__ == '__main__':
    tokenizer = HFT.AutoTokenizer.from_pretrained(r"MagicXin/Med3DVLM-Qwen-2.5-7B")
    chat = [
        {'from': 'user', 'value': 'What organ in <box_start>0.1, 0.5, 0.1, 0.5, 0.75, 0.9<box_end>?'},
        {"from": "assistant", "value": "Left ventricle."},
        {"from": 'user', 'value': 'What is its volume in cubic centimeters?'},
        {"from": 'assistant', "value": "<tool_call>\n\{\"name\":\"get_mask\", \"arguments\": {\"image_path\": \"./cache_dir/current_image.nii.gz\"}}\n"}
    ]
    out = preprocess([chat], tokenizer, 1024)
    breakpoint()