import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core import utils
from core.train_core import run_training
from core.test_core import predict_on_test
from mymodels.nasnet import nasnetalarge

import logging

MODEL_NAME = 'nasnetalarge'


def to_ssd1(name):
    return name.replace('ssd1', 'ssd2')


def load_train_ids():
    kaggle_train_meta = pd.read_csv('../input/new_train_meta.csv')
    kaggle_train_meta = kaggle_train_meta[kaggle_train_meta['fold_id'] != 0]
    pseudo = pd.read_csv('../input/pseudo.csv')['fname']
    ids_train = list(pd.read_csv('../input/cleaner.csv')['fns']) \
                + list(kaggle_train_meta['filename']) + list(pseudo)
    # ids_train += [to_ssd1(path) for path in ids_train]
    return ids_train


def load_model(args):
    if args.use_imagenet_weights:
        print('Loading imagenet pretrained weights')
    model = nasnetalarge(num_classes=len(utils.CLASSES), pretrained='imagenet', input_size=args.crop_size)
    model = nn.DataParallel(model).cuda()
    if args.model:
        print("Loading model " + args.model)
        state_dict = torch.load(args.model)['state_dict']
        model.load_state_dict(state_dict)
    return model


logging.basicConfig(filename='log/{}'.format(MODEL_NAME), level=logging.INFO,
                    format='%(asctime)s %(message)s')

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

TRAIN_FOLDER = '../../data/train'
TEST_FOLDER = '../../data/test'


def main():
    args = utils.parse_args()
    model = load_model(args)

    if not args.test:
        logging.info('Starting training with arguments %s\n', args)
        ids_train = load_train_ids()
        ids_val = pd.read_csv('../input/small_val.csv')['filename']
        run_training(MODEL_NAME, model, ids_train, ids_val, args.crop_size, args.batch_size,
                     args.workers, args.learning_rate, args.max_epoch, args.patience, args.steps)
    else:
        predict_on_test(model, args.model, TEST_FOLDER, args.tta, args.ensembling, args.crop_size)


if __name__ == '__main__':
    main()
