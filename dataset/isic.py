import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


MODE_IDS = {'Training': 0, 'Test': 1}


def isic_train_val_indices(n_images, val_fraction, seed):
    """Deterministically split range(n_images) into sorted (train, val) index lists."""
    if not 0 < val_fraction < 1:
        raise ValueError(f'val_fraction must be in (0, 1), got {val_fraction}')
    n_val = int(round(n_images * val_fraction))
    if n_val < 1 or n_val >= n_images:
        raise ValueError(f'val_fraction={val_fraction} leaves an empty split for {n_images} images')
    order = np.random.default_rng(seed).permutation(n_images)
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def prompt_rng(prompt_seed, mode, index):
    """Per-image generator so a seeded prompt does not depend on loading order or workers."""
    return np.random.default_rng([prompt_seed, MODE_IDS[mode], index])


def _sample_positive_points(mask, count, rng=None):
    """Sample `count` foreground clicks as (x, y); rng=None uses the global NumPy state."""
    randint = np.random.randint if rng is None else rng.integers
    choice = np.random.choice if rng is None else rng.choice
    foreground = np.argwhere(mask > 0.5)
    if len(foreground) == 0:
        points = np.column_stack((
            randint(0, mask.shape[1], size=count),
            randint(0, mask.shape[0], size=count),
        ))
        labels = np.zeros(count, dtype=np.int64)
    else:
        selected = choice(len(foreground), size=count, replace=len(foreground) < count)
        points = foreground[selected][:, ::-1]
        labels = np.ones(count, dtype=np.int64)
    return points.astype(np.float32), labels


def _make_bbox(mask, target_iou):
    """Create an XYXY box with target IoU against the tight foreground box."""
    height, width = mask.shape
    foreground = np.argwhere(mask > 0.5)
    if len(foreground) == 0:
        return np.array([0, 0, width, height], dtype=np.float32)

    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0) + 1
    box_width = x1 - x0
    box_height = y1 - y0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    target_area = min(box_width * box_height / target_iou, width * height)

    def expanded_box(scale):
        left = max(0, center_x - box_width * scale / 2)
        top = max(0, center_y - box_height * scale / 2)
        right = min(width, center_x + box_width * scale / 2)
        bottom = min(height, center_y + box_height * scale / 2)
        return np.array([left, top, right, bottom], dtype=np.float32)

    low, high = 1.0, 2.0 * max(width, height)
    best_box = expanded_box(low)
    best_error = abs((best_box[2] - best_box[0]) * (best_box[3] - best_box[1]) - target_area)
    for _ in range(48):
        scale = (low + high) / 2
        box = expanded_box(scale)
        area = (box[2] - box[0]) * (box[3] - box[1])
        error = abs(area - target_area)
        if error < best_error:
            best_box, best_error = box, error
        if area < target_area:
            low = scale
        else:
            high = scale
    return best_box


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'point1', plane = False, prompt_seed = None):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = 'point1' if prompt == 'click' else prompt
        if self.prompt not in ('point1', 'point3', 'bbox50', 'bbox75'):
            raise ValueError(f'Unsupported ISIC prompt type: {prompt}')
        self.img_size = args.image_size
        # None keeps the original behaviour: a fresh random click on every access.
        self.prompt_seed = prompt_seed

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        mask_array = np.asarray(mask, dtype=np.float32) / 255
        prompt_fields = {}
        if self.prompt in ('point1', 'point3'):
            point_count = 1 if self.prompt == 'point1' else 3
            rng = None if self.prompt_seed is None else prompt_rng(self.prompt_seed, self.mode, index)
            points, point_labels = _sample_positive_points(mask_array, point_count, rng)
            prompt_fields['pt'] = points
            prompt_fields['p_label'] = point_labels
        else:
            target_iou = 0.5 if self.prompt == 'bbox50' else 0.75
            prompt_fields['box'] = _make_bbox(mask_array, target_iou)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        sample = {
            'image':img,
            'label': mask,
            'image_meta_dict':image_meta_dict,
        }
        sample.update(prompt_fields)
        return sample