import os
import glob
import re
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.stats import gmean
from PIL import Image
from tqdm import tqdm

from . import utils
from . import train_utils


class OpenCVCropBase(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, img):
        i, j = self.get_params(img)
        return img[i: i + self._size, j: j + self._size]


class OpenCVCenterCrop(OpenCVCropBase):
    def __init__(self, size):
        super().__init__(size)

    def get_params(self, img):
        h, w, c = img.shape
        if h == self._size and w == self._size:
            return 0, 0
        th = tw = self._size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j


def _five_crop(img, size):
    h, w, c = img.shape
    assert c == 3, 'Something wrong with channels order'
    if size > w or size > h:
        raise ValueError(
            "Requested crop size {} is bigger than input size {}".format(size, (h, w)))
    tl = img[0: size, 0: size]
    tr = img[0: size, w - size: w]
    bl = img[h - size: h, 0: size]
    br = img[h - size: h, w - size: w]
    center = OpenCVCenterCrop(size)(img)
    return tl, tr, bl, br, center


def _get_res50_feats(df, path):
    probs = list(np.array(
        df.loc[df['fname'] == path].drop('fname', axis=1),
        dtype=np.float32).flatten())
    return probs


def predict_on_test(model, weights_path, test_folder, use_tta, ensembling, crop_size, more_feats=False):
    print('using OLD test_core')
    if use_tta:
        print('Predicting with TTA10: five crops + orientation flip')
    else:
        print('Prediction without TTA')
    ids = glob.glob(os.path.join(test_folder, '*.tif'))

    ids.sort()

    match = re.search(r'([^/]*)\.pth', weights_path)
    model_name = match.group(1) + ('_tta_' + ensembling if use_tta else '')
    csv_name = 'submit/submission_' + model_name + '.csv'

    model.eval()
    with open(csv_name, 'w') as csvfile:

        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['fname', 'camera'])
        classes = []
        preds = None
        names = []
        tta_preds = None
        tta_names = []
        if more_feats:
            augm_probas = pd.read_hdf('submit/averaged_test_augs.h5')
        for i, idx in enumerate(tqdm(ids)):
            img = np.array(Image.open(idx))
            assert img.shape == (512, 512, 3), img.shape
            assert len(img.shape) == 3, img.shape

            original_img = img

            if not more_feats:
                original_manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])
            elif idx.find('manip') != -1:
                original_manipulated = np.array([1.] + _get_res50_feats(augm_probas, idx.split('/')[-1]),
                                                dtype=np.float32)
            else:
                original_manipulated - np.array([0.] * 9, dtype=np.float32)

            if use_tta:
                transforms = [[], ['orientation']]
            else:
                transforms = [[]]
            batch_size = 1 if not use_tta else 5 * len(transforms)
            img_batch = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)
            manipulated_batch = np.zeros((batch_size, 1 if not more_feats else 9), dtype=np.float32)

            i = 0
            for transform in transforms:
                img = np.copy(original_img)
                if not more_feats:
                    manipulated = np.copy(original_manipulated)
                else:
                    raise NotImplementedError()

                if 'orientation' in transform:
                    assert img is not None
                    img = np.rot90(img, 1, (0, 1))

                if use_tta:
                    five_crops = _five_crop(img, crop_size)
                    for j in range(5):
                        img_batch[i] = five_crops[j].copy().astype(np.float32)
                        manipulated_batch[i] = manipulated
                        i += 1

            img_batch, manipulated_batch = train_utils.variable(torch.from_numpy(img_batch)), \
                                           train_utils.variable(torch.from_numpy(manipulated_batch))
            prediction = nn.functional.softmax(model(img_batch, manipulated_batch)).data.cpu().numpy()
            tta_names += [idx.split('/')[-1] + '_tta{}'.format(i) for i in range(prediction.shape[0])]
            tta_preds = np.vstack([tta_preds, prediction]) if preds is not None else prediction
            if prediction.shape[0] != 1:  # TTA
                if ensembling == 'geometric':
                    prediction = gmean(prediction, axis=0)
                else:
                    raise NotImplementedError()

            prediction_class_idx = np.argmax(prediction)

            csv_writer.writerow([idx.split('/')[-1], utils.CLASSES[prediction_class_idx]])
            classes.append(prediction_class_idx)
            names.append(idx.split('/')[-1])
            preds = np.vstack([preds, prediction]) \
                if preds is not None else prediction

        df_data = np.append(preds, np.array(names, copy=False, subok=True, ndmin=2).T, axis=1)
        df = pd.DataFrame(data=df_data, columns=utils.CLASSES + ['fname'])
        os.makedirs('submit', exist_ok=True)
        df.to_hdf('submit/{}_test_pr.h5'.format(model_name), 'prob')

        tta_data = np.append(tta_preds, np.array(tta_names, copy=False, subok=True, ndmin=2).T, axis=1)
        tta_df = pd.DataFrame(data=tta_data, columns=utils.CLASSES + ['fname'])
        tta_df.to_hdf('submit/{}_test_pr_with_tta.h5'.format(model_name), 'prob')

        print("Test set predictions distribution:")
        utils.print_distribution(None, classes=classes)
        print("Now you are ready to:")
        print("kg submit {}".format(csv_name))
