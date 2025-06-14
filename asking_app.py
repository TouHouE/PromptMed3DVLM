import torch
import argparse
from monai import transforms as MT
import transformers as HFT
from src.model.llm import VLMQwenForCausalLM

model = None
tokenizer = None
image_loader = None
convs_hist = list()


def slice_scaler(pack, scaler):
    mz = pack.shape[-1]
    return torch.stack([scaler(pack[..., z]) for z in range(mz)], dim=-1)


def nnunet_scaler(pack):
    low, high = -395., 842.
    avg, std = 279.8117370605469, 253.5583953857422
    if isinstance(pack, dict):
        np.clip(pack['image'], -395.0, 842.0, out=pack['image'])
        pack['image'] -= avg
        pack['image'] /= std
    else:
        torch.clip(pack, low, high, out=pack)
        pack -= avg
        pack /= std
    return pack


def load_model(dst_model_name):
    try:
        __model = HFT.AutoModelForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16,
                                                           device_map='auto', trust_remote_code=True)
    except Exception as e:
        __model = VLMQwenForCausalLM.from_pretrained(dst_model_name, torch_dtype=torch.bfloat16, device_map='auto')
    return __model


@torch.inference_mode()
def asking(text, __image, temp=0, top_p=.9, max_length=512):
    text = "<im_patch>" * 256 + text + " "
    pack = tokenizer(text, return_tensors="pt")
    if isinstance(__image, str):
        __image = image_loader(__image)
    if __image is not None:
        if __image.ndim == 4:  # Adding batch_size
            __image = __image[None]
        __image = __image.to('cuda', torch.bfloat16)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            inputs=pack['input_ids'].to('cuda'),
            images=__image,
            max_new_tokens=max_length,
            do_sample=temp > 0,
            top_p=top_p,
            temperature=temp,
        )
    output_text = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )
    return {
        "AI": output_text,
        "temp": temp,
        "top_p": top_p,
        "max_length": max_length
    }


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
model = load_model(args.model_name)
tokenizer = HFT.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
backbone = [
    MT.LoadImage(image_only=True),
    MT.EnsureType(device='cuda'),
    MT.EnsureChannelFirst(),
    MT.Orientation("RAS")
]
comp = backbone + [
    MT.Lambda(nnunet_scaler),
    MT.Zoom(0.5, mode='trilinear'),
    MT.ResizeWithPadOrCrop((256, 256, 128)),
    MT.ToTensor(),
]

image_loader = MT.Compose(comp)



