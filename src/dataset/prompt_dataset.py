import os
import random
from os.path import join, exists
from typing import Final, Optional

import torch
from torch.utils.data import Dataset 
from monai import transforms as MT

from src.dataset.utils import myio as UIO
from src.dataset.utils import transforms as UT
ROOT: Final[str] = '/home/jovyan/shared/uc207pr4f57t9/cardiac/nnUNet'




class PromptCardiacDataset(Dataset):
    SERIES_NAME: Final[str] = 'Dataset001_Cardiac'
    data_list: list

    def __init__(self, args):
        super().__init__()
        self.args = args        
        self.data_label = UIO.load_json(join(ROOT, 'raw_dir', self.SERIES_NAME, 'dataset.json'))['labels']
        self.data_list = UIO.load_json(join(ROOT, 'preprocessed_dir', self.SERIES_NAME, 'splits_final.json'))[args.fold]['train']
        stem = UT.get_loader(args)
        stem.extend([                                             
            MT.RandRotate90d(prob=0.5, spatial_axes=(0, 1), keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            MT.RandFlipd(prob=0.70, spatial_axis=0, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            MT.RandFlipd(prob=0.70, spatial_axis=1, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            MT.RandFlipd(prob=0.70, spatial_axis=2, keys=['image', 'label', 'image_fg'], allow_missing_keys=True),
            MT.RandScaleIntensityd(factors=0.5, prob=0.75, keys=['image', 'image_fg'], allow_missing_keys=True),
            MT.RandShiftIntensityd(offsets=0.5, prob=0.75, keys=['image', 'image_fg'], allow_missing_keys=True),
        ])
        self.trans = MT.Compose(stem)


    def __getitem__(self, item):
        name = self.data_list[item] # e.g.: "Cardiac_0001"
        image = join(ROOT, 'raw_dir', self.SERIES_NAME, 'imagesTr', f"{name}_0000.nii.gz")
        label = join(ROOT, 'raw_dir', self.SERIES_NAME, 'labelsTr', f"{name}.nii.gz")
        vpack = self.trans({'image': image, 'label': label})
        vpack['mask'] = vpack.pop('label')
        all_cate = torch.unique(vpack['mask'])
        random_class = 10
        if 10 not in all_cate:  # I wanna focus on plaque @@
            random_class = int(random.sample(all_cate.tolist(), 1)[0])
        
        label = torch.zeros((self.args.num_class, ))
        label[random_class] = 1
        vpack['label'] = label
        vpack['mask'][vpack['mask'] != random_class] = 0
        
        vpack['image-file'] = image
        vpack['label-file'] = label

        return vpack



        ...

    def __len__(self):
        return len(self.data_list)