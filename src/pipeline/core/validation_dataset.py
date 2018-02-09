import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from core.custom_dataset import get_class, ORIENTATION_FLIP_ALLOWED, \
    load_img_fast_jpg, get_crop, random_manipulation, MANIPULATIONS

UNALT_MODE = 'unalt'
MANIP_MODE = 'manip'


class ValidationDataset(Dataset):
    def __init__(self, ids, crop_size, mode, more_feats=False):
        assert mode in (UNALT_MODE, MANIP_MODE)
        self._crop_size = crop_size
        self._mode = mode
        self._meta = self._get_meta(ids)
        self._more_feats = more_feats
        if self._more_feats:
            self._res50_pr = pd.read_hdf('submit/res50_val_pr.h5')

    def __len__(self):
        if self._mode == UNALT_MODE:
            return len(self._meta)
        else:
            return len(self._meta) * 8

    def __getitem__(self, item):
        path, should_flip_orientation = self._get_path_and_flip(item)
        img = load_img_fast_jpg(path)
        if should_flip_orientation:
            img = np.rot90(img, 1, (0, 1))
        if self._mode == MANIP_MODE:
            img = get_crop(img, 2 * self._crop_size, random_crop=False)
            manip_idx = item % 8
            img = random_manipulation(img, manipulation=MANIPULATIONS[manip_idx])
        img = get_crop(img, self._crop_size, random_crop=False)
        img = img.astype(np.float32)
        if not self._more_feats:
            return img.copy(), np.array([self._mode == MANIP_MODE], dtype=np.float32), \
                   get_class(path.split('/')[-2])
        if self._mode == UNALT_MODE:
            augm_feats = np.array([0.] * 9, dtype=np.float32)
        else:
            resnet50_detect_proba = self._get_res50_feats(path, manip_idx)
            augm_feats = np.array([1.] + resnet50_detect_proba, dtype=np.float32)
        return img.copy(), augm_feats, get_class(path.split('/')[-2])

    def _get_path_and_flip(self, item):
        if self._mode == UNALT_MODE:
            return self._meta[item]
        return self._meta[item // 8]

    @staticmethod
    def _get_meta(ids):
        meta = []
        for path in ids:
            class_idx = get_class(path.split('/')[-2])
            if ORIENTATION_FLIP_ALLOWED[class_idx]:
                meta.append((path, False))
                meta.append((path, True))
            else:
                meta.append((path, False))
        return meta

    def _get_res50_feats(self, path, m_idx):
        assert self._more_feats
        probs = list(np.array(
            self._res50_pr.loc[(self._res50_pr['fname'] == path)
                               & (self._res50_pr['augm'] == m_idx)].drop('fname', axis=1).drop('augm', axis=1),
            dtype=np.float32).flatten())
        return probs
