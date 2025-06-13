import json
import os
from os.path import join, exists
from typing import Optional, Final
from itertools import product


DEF_ROOT_LIST: Final[list[str]] = ['/home/jovyan/shared/uc207pr4f57t9/cardiac/taipei/taipei', '/home/jovyan/shared/uc207pr4f57t9/cardiac/sub/taipei']
DEF_MID_LIST: Final[list[str]] = ['to_saturn', 'to_saturn_yeh', 'to_saturn_beato']


def load_make_sure_exists(pack, possible_root: Optional[list[str]] = None, possible_mid_path: Optional[list[str]] = None):
    if possible_root is None:
        possible_root: list[str] = DEF_ROOT_LIST
    if possible_mid_path is None:
        possible_mid_path: list[str] = DEF_MID_LIST

    for root, mid_path in product(possible_root, possible_mid_path):
        cur_abs_path = join(root, mid_path, pack['image'])

        if not exists(cur_abs_path):
            continue
        pack['image'] = cur_abs_path

        if 'label' in pack:
            cur_abs_label = join(root, mid_path, pack['label'])
            if not exists(cur_abs_label):
                pack.pop('label')
            else:
                pack['label'] = cur_abs_label
        return pack
    return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as loader:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in loader.readlines()]
        return json.load(loader)