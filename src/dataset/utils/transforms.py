import os
from typing import Iterable, Literal
DEBUG: bool = os.environ.get("DEBUG", "0") == "1"


import numpy as np
import torch
from monai import transforms as MT


def debug(pack):
    print("="*30)
    for key, value in pack.items():
        if 'coord' in key:
            print(value)
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            print(f'{key} is a tensor| shape: {value.shape}')
            continue
        print(f'{key} is a {type(value)}| content: {value}')
    print("="*30)
    return pack


def nnunet_scaler(pack) -> dict | Iterable[float | int]:
    if DEBUG:
        print(f'Doing nnUNet CT normalize')
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
    if DEBUG:
        print(f'Into Adding new KEYS')
    pack['image_fg'] = pack['image'].clone()
    
    if 'label' not in pack:
        pack['label'] = torch.zeros_like(pack['image'])    
    pack['mask_fg'] = pack['label'].clone()

    if pack['image_fg'].shape != pack['mask_fg'].shape:
        pack['mask_fg'] = torch.zeros_like(pack['image_fg'])
    return pack


def get_fg_loader(args) -> list[callable]:
    """
        the load_kwargs only contains `keys` and `allow_missing_keys` 2 keys
        basically when `args.loader_type == 'unet-med3d-fgcrop' will get into here.
    """
    comp: list[callable] = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True, image_only=True),
        # MT.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device='cuda'),
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes='RAS', keys=['image', 'label'], allow_missing_keys=True),
        MT.Spacingd(pixdim=(.39, .39, .625), keys=['image', 'label'], allow_missing_keys=True, mode=('trilinear', 'nearest'))
    ]
    comp.append(MT.Lambda(nnunet_scaler))
    comp.append(MT.Lambda(adding_new_keys))
    if DEBUG:
        comp.append(MT.Lambda(debug))
    comp.append(
        DummyCropForeground(
            classes_range=[0, 10], source_key='mask_fg', keys=['image_fg', 'mask_fg'], allow_missing_keys=True
        )
    )
    if DEBUG:
        comp.append(MT.Lambda(debug))
    comp.append(MixedResizer(
        spatial_size=(256, 256, 128),
        padder_kwargs=dict(mode='constant', constant_values=0, method='end'),
        resizer_kwargs=dict(mode=('trilinear', 'trilinear', 'nearest'), size_mode='all'),
        keys=['image', 'image_fg', 'label'], allow_missing_keys=True
    ))
    comp.append(MT.DeleteItemsd(keys=['mask_fg']))
    comp.append(MT.ResizeWithPadOrCropd(keys=['image', 'label', 'image_fg'], spatial_size=(256, 256, 128), allow_missing_keys=True))
    if DEBUG:
        comp.append(MT.Lambda(debug))
    comp.append(MT.ToTensord(keys=['image', 'label', 'image_fg']))

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
    device = 'cuda' if getattr(args, "move_to_cuda", False) else 'cpu'
    stem = [
        MT.LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        MT.EnsureTyped(keys=['image', 'label'], allow_missing_keys=True, device=device), 
        MT.EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        MT.Orientationd(axcodes="RAS", keys=['image', 'label'], allow_missing_keys=True),
        scaler
    ]

    if model_arch_type == 'm3d':
        stem.append(MT.Orientationd(axcodes="SRA", keys=['image', 'label'], allow_missing_keys=True))
        input_size = (32, 256, 256)
    if shape_type in ['resize', 'zoom']:
        stem.append(MT.Zoomd(zoom=.5, mode=('trilinear', 'nearest'), keys=['image', 'label'], allow_missing_keys=True))
    stem.append(MT.ResizeWithPadOrCropd(spatial_size=input_size, keys=['image', 'label'], allow_missing_keys=True))
    stem.append(MT.ToTensord(keys=['image', 'label'], allow_missing_keys=True))
    return stem


def get_loader(args):
    if 'fgcrop' in args.loader_type:
        print("Choose Foreground Loader")
        return get_fg_loader(args)
    return get_normal_loader(args)


def select_fn(x):
    return x == 1


class DummyCropForeground(MT.Transform):
    def __init__(self, classes_range: range | list[int], source_key: str, **kwargs):
        """
            The classes_range is the range of the foreground classes if is a list the actual range will be:
            list(n0, n1) -> range(n0, n1 + 1)
        """
        # super().__init__(**kwargs)
        if isinstance(classes_range, list):
            classes_range = range(classes_range[0], classes_range[1] + 1)
        self.cropper = MT.CropForegroundd(select_fn=select_fn, source_key=source_key, **kwargs)
        # self.rand_cropper = MT.CenterSpatialCrop(roi_size=(128, 128, 128))
        self.source_key = source_key
        self.full_class = classes_range

    def __call__(self, pack: dict):
        src_k = self.source_key

        if 'organ' not in pack:            
            organ = self.full_class
        else:
            organ = pack['organ']
        if DEBUG and 'organ' not in pack:
            print(f"key `organ` not in input, using {self.full_class} instead")
        if DEBUG and 'organ' in pack:
            print(f'Organ: {pack["organ"]}')
            
        for organ_id in organ:
            pack[src_k][pack[src_k] == organ_id] = -1
        pack[src_k][pack[src_k] > -1] = 0
        pack[src_k][pack[src_k] == -1] = 1
        cache_pack = self.cropper(pack)
        
        if 0 in cache_pack[src_k].shape:
            return pack
        return cache_pack


class MixedResizer(MT.Transform):
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
        try:
            pack = self.resizer(pack)
            return self.padder(pack)
        except Exception as e:
            import traceback as tb
            tb.print_exc()
            raise e


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

