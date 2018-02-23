import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import gmean
from PIL import Image
from tqdm import tqdm

from . import utils
from . import train_utils
from .mytransforms import five_crop
from ..mymodels import densenet, dpn, resnext, se_resnet, seresnext


def get_batches(img_path, use_tta, crop_size):
    img = np.array(Image.open(img_path))
    assert len(img.shape) == 3, img.shape
    original_img = img
    original_manipulated = np.float32([1. if img_path.find('manip') != -1 else 0.])
    if use_tta:
        transforms = [[], ['orientation']]
    else:
        transforms = [[]]
    batch_size = 1 if not use_tta else 5 * len(transforms)
    img_batch = np.zeros((batch_size, crop_size, crop_size, 3), dtype=np.float32)
    manipulated_batch = np.zeros((batch_size, 1), dtype=np.float32)
    i = 0
    for transform in transforms:
        img = np.copy(original_img)
        manipulated = np.copy(original_manipulated)
        if 'orientation' in transform:
            assert img is not None
            img = np.rot90(img, 1, (0, 1))
        if use_tta:
            five_crops = five_crop(img, crop_size)
            for j in range(5):
                img_batch[i] = five_crops[j].copy().astype(np.float32)
                manipulated_batch[i] = manipulated
                i += 1
    return img_batch, manipulated_batch


def predict_test_proba(model, test_folder, use_tta, crop_size):
    print(test_folder)
    ids = glob.glob(os.path.join(test_folder, '*.tif'))
    ids.sort()
    model.eval()
    preds = None
    names = []
    for idx in tqdm(ids):
        img_batch, manipulated_batch = get_batches(idx, use_tta, crop_size)
        img_batch = train_utils.variable(torch.from_numpy(img_batch))
        manipulated_batch = train_utils.variable(torch.from_numpy(manipulated_batch))
        prediction = nn.functional.softmax(model(img_batch, manipulated_batch)).data.cpu().numpy()
        if prediction.shape[0] != 1:  # TTA
            prediction = gmean(prediction, axis=0)
        names.append(idx.split('/')[-1])
        preds = np.vstack([preds, prediction]) \
            if preds is not None else prediction
    df_data = np.append(preds, np.array(names, copy=False, subok=True, ndmin=2).T, axis=1)
    df = pd.DataFrame(data=df_data, columns=utils.CLASSES + ['fname'])
    return df


def make_model(weights_path):
    model_name = weights_path.split('/')[-1]
    if model_name.startswith('densenet201'):
        model = densenet.densenet201(num_classes=len(utils.CLASSES), pretrained=True)
    elif model_name.startswith('dpn92'):
        model = dpn.dpn92(num_classes=len(utils.CLASSES), pretrained='imagenet+5k')
    elif model_name.startswith('resnext101'):
        model = resnext.resnext101_32x4d(num_classes=len(utils.CLASSES), pretrained='imagenet')
    elif model_name.startswith('densenet161'):
        model = densenet.densenet161(num_classes=len(utils.CLASSES), pretrained=True)
    elif model_name.startswith('dpn98'):
        model = dpn.dpn98(num_classes=len(utils.CLASSES), pretrained='imagenet')
    elif model_name.startswith('se_resnet50'):
        model = se_resnet.se_resnet50(num_classes=len(utils.CLASSES)).cuda()
    elif model_name.startswith('se_resnext50'):
        model = seresnext.se_resnext50(num_classes=len(utils.CLASSES), pretrained=True).cuda()
    else:
        raise RuntimeError('Unknown model name: {}'.format(model_name))
    if not model_name.startswith('se_resnet50') and not model_name.startswith('se_resnext50'):
        model = nn.DataParallel(model).cuda()
    state_dict = torch.load(weights_path)['state_dict']
    model.load_state_dict(state_dict)
    return model


def get_pr(df):
    return np.array(df.drop('fname', axis=1), dtype=np.float128)


def get_gmeaned(arr):
    gm = pd.DataFrame(gmean(
        np.dstack([get_pr(df) for df in arr]), axis=2
    ))
    gm['fname'] = arr[0]['fname']
    gm.columns = arr[0].columns
    return gm


def predict_gmean_ensemble(checkpoints_paths, test_dir):
    predicts = []
    for i, weights_path in enumerate(checkpoints_paths):
        print('{}/{}: Predicting with model at {}'.format(i + 1, len(checkpoints_paths),
                                                          weights_path))
        model = make_model(weights_path)
        predicts.append(predict_test_proba(model, test_dir, use_tta=True, crop_size=480))
    final_proba = get_gmeaned(predicts)
    return final_proba
