from typing import Iterable, Literal

import numpy as np
import torch
from monai import transforms as MT


def nnunet_scaler(pack) -> dict | Iterable[float | int]:
    is_dict = isinstance(pack, dict)        
    is_torch = torch.is_tensor(pack['image']) if is_dict else torch.is_tensor(pack)

    cliper = torch.clip if is_torch else np.clip
    if is_dict:
        cliper(pack['image'], -395.0, 842.0, out=pack['image'])
        pack['image'] -= 279.8117370605469
        pack['image'] /= 253.5583953857422
    else:
        cliper(pack, -395.0, 842.0, out=pack)
        pack -= 279.8117370605469
        pack /= 253.5583953857422
    
    return pack


def adding_new_keys(pack: dict[str, torch.Tensor]):
    pack['image_Fg'] = pack['image'].clone()
    pack['mask_Fg'] = pack['label'].clone()
    return pack


def get_fg_loader(args) -> list[callable]:
    """
        the load_kwargs only contains `keys` and `allow_missing_keys` 2 keys
        basically when `args.loader_type == 'unet-med3d-fgcrop' will get into here.
    """
    comp: list[callable] = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes='RAS', keys=['image', 'label'], allow_missing_keys=True)
    ]
    comp.append(MT.Lambda(nnunet_scaler))
    comp.append(MT.Lambda(adding_new_keys))
    comp.append(
        DummyCropForeground(
            classes_range=[0, 10], source_key='mask_Fg', keys=['image_Fg', 'mask_Fg'], allow_missing_keys=True
        )
    )
    comp.append(MixedResizer(
        spatial_size=(256, 256, 128),
        padder_kwargs=dict(mode='constant', constant_values=0, method='end'),
        resizer_kwargs=dict(mode=('trilinear', 'trilinear', 'nearest'), size_mode='all'),
        keys=['image', 'image_Fg', 'label'], allow_missing_keys=True
    ))
    comp.append(MT.DeleteItemsd(keys=['mask_Fg']))
    comp.append(MT.ResizeWithPadOrCropd(keys=['image', 'label', 'image_Fg'], spatial_size=(256, 256, 128), allow_missing_keys=True))
    comp.append(MT.ToTensord(keys=['image', 'label', 'image_Fg']))

    return comp


def get_normal_loader(args):
    scaler_type, model_arch_type, shape_type = args.loader_type.split('-')
    input_size = args.input_size

    if scaler_type == 'unet':
        scaler = MT.Lambda(nnunet_scaler)
    elif scaler_type == 'jpeg':
        scaler = PseudoJPEGScaleIntensity(keys=['image'])
    elif scaler_type == 'minmax':
        scaler = MT.ScaleIntensityd(keys=['image'])
    else:
        raise NotImplementedError(f"Unexpected scaler type: {scaler_type}")
    stem = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        MT.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cuda'),
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes="RAS", keys=['image', 'label'], allow_missing_keys=True),
        scaler
    ]

    if model_arch_type == 'm3d':
        stem.append(MT.Orientationd(axcodes="SRA", keys=['image', 'label'], allow_missing_keys=True))
        input_size = (32, 256, 256)
    if shape_type == 'resize':
        stem.append(MT.Zoomd(factor=.5, mode=('trilinear', 'nearest'), keys=['image', 'label'], allow_missing_keys=True))
    stem.append(MT.ResizeWithPadOrCropd(spatial_size=input_size, keys=['image', 'label'], allow_missing_keys=True))
    stem.append(MT.ToTensord(keys=['image', 'label'], allow_missing_keys=True))
    return stem


def get_loader(args):
    if 'fgcrop' in args.loader_type:
        return get_fg_loader(args)
    return get_normal_loader(args)


class DummyCropForeground:
    def __init__(self, classes_range: range | list[int, int], **kwargs):
        """
            The classes_range is the range of the foreground classes if is a list the actual range will be:
            list(n0, n1) -> range(n0, n1 + 1)
        """
        # super().__init__(**kwargs)
        if isinstance(classes_range, list):
            classes_range = range(classes_range[0], classes_range[1] + 1)                        
        self.cropper = MT.CropForegroundd(select_fn=lambda x: x == 1, **kwargs)        
        self.full_class = classes_range
        
    def __call__(self, pack: dict):
        if 'label' not in pack:
            return pack
        
        if 'organ' not in pack:
            organ = self.full_class
        else:
            organ = pack['organ']

        for organ_id in organ:
            print(organ_id)
            pack['label'][pack['label'] == organ_id] = -1
        print(f'{type(pack["label"])}')
        pack['label'][pack['label'] > -1] = 0
        pack['label'][pack['label'] == -1] = 1
        breakpoint()
        return self.cropper(pack)


class MixedResizer:
    def __init__(self, keys, allow_missing_keys, spatial_size, padder_kwargs, resizer_kwargs):
        self.final_size = spatial_size
        self.padder = MT.ResizeWithPadOrCropd(
            keys=keys, allow_missing_keys=allow_missing_keys, 
            spatial_size=self.final_size, **padder_kwargs
        )
        if resizer_kwargs.get('size_mode') == 'longest' and not isinstance(spatial_size, int):
            spatial_size = max(spatial_size)
        self.resizer = MT.Resized(
            keys=keys, allow_missing_keys=allow_missing_keys, 
            spatial_size=spatial_size, **resizer_kwargs
        )
    
    def __call__(self, pack):        
        pack = self.resizer(pack)
        return self.padder(pack)
    

class PseudoJPEGScaleIntensity:
    def __init__(self, keys, tensor_type: Literal['numpy', 'torch']):
        self.ttype = tensor_type
        self.keys = keys
        self.to_jpeg = MT.ScaleIntensity(0, 255, dtype=torch.uint8)
        self.to_01 = MT.ScaleIntensity(dtype=torch.float)

    def on_slice(self, volume):
        depth = volume.shape[-1]
        return torch.stack([self.to_01(self.to_jpeg(volume[..., d])) for d in depth], dim=-1)

    def __call__(self, data):
        for key, value in data.items():
            if key not in self.keys:
                continue
            data[key] = self.on_slice(value)
        return data

