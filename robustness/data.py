"""ISIC 2016 samples for robustness evaluation: the training pipeline's image, label and (seeded) click,
plus the binary lesion mask at model-input resolution used to place spots."""
import os
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from .repo import isic_module

SPLITS = {'test': 'Test', 'val': 'Training'}


def _scale_to_255(x):
    return x * 255


def isic_transforms(image_size, out_size):
    """Identical to the ISIC transforms in dataset/__init__.py (images in [0, 255], labels in {0, 1})."""
    image_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(_scale_to_255),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((out_size, out_size)),
        transforms.ToTensor(),
    ])
    return image_tf, mask_tf


class ISICRobustDataset(Dataset):
    """split: 'test' (all 379 test images) or 'val' (the -val_split hold-out of the training images)."""

    def __init__(self, data_path, split, image_size=1024, out_size=256, prompt_seed=0, val_split=0.1, split_seed=0,
                 limit=None):
        if split not in SPLITS:
            raise ValueError(f'split must be one of {sorted(SPLITS)}')
        isic = isic_module()
        image_tf, mask_tf = isic_transforms(image_size, out_size)
        self.mode = SPLITS[split]
        self.split_id = isic.MODE_IDS[self.mode]
        self.image_size = image_size
        self.base = isic.ISIC2016(SimpleNamespace(image_size=image_size), data_path, transform=image_tf,
                                  transform_msk=mask_tf, mode=self.mode, prompt='point1', prompt_seed=prompt_seed)
        if split == 'val':
            _, self.indices = isic.isic_train_val_indices(len(self.base), val_split, split_seed)
        else:
            self.indices = list(range(len(self.base)))
        if limit is not None:
            self.indices = self.indices[:limit]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        sample = self.base[index]
        # Same resize as ISIC2016 applies before sampling the click, so spots and click share one lesion mask.
        mask = Image.open(os.path.join(self.base.data_path, self.base.label_list[index])).convert('L')
        mask = mask.resize((self.image_size, self.image_size))
        lesion = np.asarray(mask, dtype=np.float32) / 255 > 0.5
        return {
            'image': sample['image'],                    # (3, S, S) float in [0, 255]
            'label': sample['label'].float(),            # (1, out, out) in {0, 1}
            'click': torch.from_numpy(sample['pt'][0]),  # (2,) = (x, y) in pixels
            'lesion': torch.from_numpy(lesion),          # (S, S) bool
            'index': index,
            'name': sample['image_meta_dict']['filename_or_obj'],
        }
