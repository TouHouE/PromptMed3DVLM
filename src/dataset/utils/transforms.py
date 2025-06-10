from typing import Iterable

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


def get_clip_loader(args, load_kwargs) -> list[callable]:
    """
        the load_kwargs only contains `keys` and `allow_missing_keys` 2 keys
    """
    comp = [MT.LoadImaged(**load_kwargs), MT.EnsureChannelFirstd(**load_kwargs), MT.Orientationd(axcodes='RAS', **load_kwargs)]
    if args.shape_mode == 'crop':
        spacing = (.39, .39, .625)        
    elif args.shape_mode == 'resize':
        spacing = (.78, .78, 1.25)
    
    if args.shape_mode != 'fgcrop':
        comp.append(MT.Spacingd(**load_kwargs, pixdim=spacing, mode=('trilinear', 'nearest')))
    
    comp.append(MT.Lambda(lambda pack: nnunet_scaler(pack)))

    if args.shape_mode == 'fgcrop':
        # keys=['image', 'label'], source_key='label', allow_missing_keys=True, classes_range=[0, 10]     
        comp.append(MT.Lambda(lambda pack: adding_new_keys(pack)))  # ['image', 'label', 'image_Fg', 'Fg_mask]
        comp.append(DummyCropForeground(classes_range=[0, 10], source_key='mask_Fg', keys=['image_Fg', 'mask_Fg'], allow_missing_keys=True))
        comp.append(MixedResizer(
            spatial_size=(256, 256, 128),
            padder_kwargs=dict(mode='constant', constant_values=0, method='end'),
            resizer_kwargs=dict(mode=('trilinear', 'trilinear', 'nearest'), size_mode='all'),
            keys=['image', 'image_Fg', 'label'], allow_missing_keys=True
        ))
        comp.append(MT.DeleteItemsd(keys=['mask_Fg'], allow_missing_keys=True))
    else:                
        comp.append(MT.ResizeWithPadOrCropd(spatial_size=(256, 256, 128), **load_kwargs))


    return comp



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