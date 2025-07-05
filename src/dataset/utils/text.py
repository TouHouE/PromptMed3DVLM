import os
# os.environ['HF_HOME'] = r"D:\huggingface"
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


def making_textual_input_label(sources, tokenizer: HFT.PreTrainedTokenizer, max_len, image_tokens, **kwargs):
    if isinstance(sources[0], list):
        sources = sources[0]
    question = sources[0]['content']
    answer = sources[1]['content']
    question = image_tokens + " " + question
    text_tensor = tokenizer(
        question + " " + answer,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_id = text_tensor["input_ids"][0]
    attention_mask = text_tensor["attention_mask"][0]

    valid_len = torch.sum(attention_mask)
    if valid_len < len(input_id):
        input_id[valid_len] = tokenizer.eos_token_id

    question_tensor = tokenizer(
        question,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    question_len = torch.sum(question_tensor["attention_mask"][0])

    label = input_id.clone()
    label[:question_len] = IGNORE_TOKEN_ID
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        label[label == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        if valid_len < len(label):
            label[valid_len] = tokenizer.eos_token_id
    else:
        label[label == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    return {
        'input_ids': input_id[None],
        'attention_mask': attention_mask[None],
        'labels': label[None]
    }

def get_im_start_end(tokenizer):
    if (im_start := getattr(tokenizer, 'im_start_id', None)) is None:
        im_start = tokenizer("<|im_start|>").input_ids[0]

    if (im_end := getattr(tokenizer, 'im_end_id', None)) is None:
        im_end = tokenizer("<|im_end|>").input_ids[0]
    return {
        'im_start': im_start,
        'im_end': im_end,
    }



def qwen_preprocess(
    sources,
    tokenizer: HFT.PreTrainedTokenizer,
    max_len: int,
    image_tokens: str,
    system_message: str = "You are a helpful assistant.", **kwargs  
) -> dict:
    if not isinstance(sources[0], list):
        sources = [sources]
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
        if roles[source[0]["role"]] != roles["user"]:
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
            role = roles[sentence["role"]]
            if sentence['role'] == 'user' and not image_token_is_added:
                sentence['content'] = image_tokens + "\n" + sentence['content']
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["content"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            # print(f"role: {role}")
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
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess(
    sources,
    tokenizer: HFT.PreTrainedTokenizer,
    max_len: int, 
    image_tokens: str,
    args, **kwargs
):
    if getattr(args, 'chat_mode', True):        
        return qwen_preprocess(
            sources, tokenizer, max_len, image_tokens, **kwargs
        )
    return making_textual_input_label(
        sources, tokenizer, max_len, image_tokens
    )

def show_debug_pack(pack, tokenizer):
    for key, value in pack.items():
        print(f'{key} -> {value.shape}')
    for loc, (token, lab_token, attn) in enumerate(zip(pack['input_ids'][0], pack['labels'][0], pack['attention_mask'][0])):
        # print(token)
        # print(tokenizer("<|endoftext|>")['input_ids'])
        # break
        if token >= 0:            
            token = tokenizer.decode(token)
            if '\n' in token:
                token = token.replace("\n", "\\n")            
            if ' ' in token:
                token = token.replace(" ", "[sp]")
        if lab_token > 0:
            lab_token = tokenizer.decode(lab_token)
            if '\n' in lab_token:
                lab_token = lab_token.replace('\n', '\\n')
            if ' ' in lab_token:
                lab_token = lab_token.replace(" ", '[sp]')
        print(f'[{loc:03}]|{token:15}:{attn}:{lab_token}')


def random_text():
    from string import ascii_letters
    import random
    return "".join(random.choices(ascii_letters, k=random.randint(12, 33)))

if __name__ == '__main__':
    tokenizer = HFT.AutoTokenizer.from_pretrained(r"MagicXin/Med3DVLM-Qwen-2.5-7B")
    tokenizer.add_tokens("<|nvis_data_sep|>")    
    chat = [
        {'role': 'user', 'content': 'Based on provided data:\nA 66 year old female.<|nvis_data_sep|>Were are the LV?'},
        {"role": "assistant", "content": "Left ventricle located at <|box_start|>0.1, 0.5, 0.1, 0.5, 0.75, 0.9<|box_end|>."},
        # {"role": 'user', 'content': 'What is its volume in cubic centimeters?'},
        # {"role": 'assistant', "content": "<tool_call>\n\{\"name\":\"get_mask\", \"arguments\": {\"image_path\": \"./cache_dir/current_image.nii.gz\"}}\n"}
    ]
    class DYM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embeddings(len(tokenizer), 128)
            self.lm_head = nn.Linear(128, len(tokenizer), bias=False)
        def forward(self, inputs):
            emb = self.embed(inputs)
            return self.lm_head(emb)


    out = making_textual_input_label([chat], tokenizer, 1024, image_tokens="<im_patch>" * 1)
    out = preprocess([chat], tokenizer, 768, image_tokens="<im_patch>" * 256)
    show_debug_pack(out, tokenizer)

    # for key, value in out.items():
    #     print(f'{key} -> {value.shape}')

    # for loc, (token, lab_token, attn) in enumerate(zip(out['input_ids'][0], out['labels'][0], out['attention_mask'][0])):
    #     # print(token)
    #     # print(tokenizer("<|endoftext|>")['input_ids'])
    #     # break
    #     if token < 0 or token == tokenizer("<|endoftext|>")['input_ids'][0]:
    #         continue
    #     text = tokenizer.decode(token)
    #     if '\n' in text:
    #         text = text.replace("\n", "\\n")            
    #     if ' ' in text:
    #         text = text.replace(" ", "[sp]")
    #     if lab_token > 0:
    #         lab_token = tokenizer.decode(lab_token)
    #         if '\n' in lab_token:
    #             lab_token = lab_token.replace('\n', '\\n')
    #         if ' ' in lab_token:
    #             lab_token = lab_token.replace(" ", '[sp]')
    #     print(f'[{loc:03}]|{text:15}:{attn}:{lab_token}')

    
    # breakpoint()