import os
import logging
import random
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler

from . import utils
from . import train_utils
from .custom_dataset import IEEECameraDataset
from .custom_scheduler import ReduceLROnPlateau
from .validation_dataset import ValidationDataset
from .. import config


def run_training(model_name, model, ids_train, ids_val,
                 batch_size, steps_per_epoch, use_d4=False):
    if use_d4:
        logging.info('Training with D4 transforms')
    print("Training set distribution:")
    utils.print_distribution(ids_train)
    print("Validation set distribution:")
    utils.print_distribution(ids_val)

    classes_train = [utils.get_class(idx.split('/')[-2]) for idx in ids_train]
    class_w = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

    weights = [class_w[i_class] for i_class in classes_train]
    weights = torch.DoubleTensor(weights)
    train_sampler = sampler.WeightedRandomSampler(weights, steps_per_epoch * batch_size)
    train_dataset = IEEECameraDataset(ids_train, crop_size=config.CROP_SIZE, training=True, d4=use_d4)

    val_dataset_unalt = ValidationDataset(ids_val, crop_size=config.CROP_SIZE, mode='unalt')
    val_dataset_manip = ValidationDataset(ids_val, crop_size=config.CROP_SIZE, mode='manip')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loaders = {
        'unalt': DataLoader(val_dataset_unalt, batch_size=batch_size,
                            num_workers=config.NUM_WORKERS, pin_memory=True),
        'manip': DataLoader(val_dataset_manip, batch_size=batch_size,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    }
    optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=config.PATIENCE, min_lr=1e-9,
                                  epsilon=1e-5, verbose=1, mode='min')

    criterion = nn.CrossEntropyLoss()

    train_utils.train_and_validate(
        train_loader, valid_loaders, model,
        optimizer, scheduler, criterion, model_name)


def load_train_ids(use_pseudo):
    kaggle_train_meta = pd.read_csv(config.TABLES_DIR + '/kaggle_train_meta.csv')
    kaggle_train_meta = kaggle_train_meta[kaggle_train_meta['fold_id'] != 0]
    ids_train = list(pd.read_csv(config.TABLES_DIR + '/cleaner.csv')['fns']) \
                + list(kaggle_train_meta['filename'])
    if use_pseudo:
        pseudo = pd.read_csv(config.TABLES_DIR + '/pseudo.csv')['fname']
        ids_train += list(pseudo)
    return ids_train


def load_valid_ids():
    return pd.read_csv(config.TABLES_DIR + '/val.csv')['filename']


def log_config(use_d4):
    logging.info('\nStarting training with params:')
    logging.info(config.CONFIG_LOG_MESSAGE.format(config.CROP_SIZE, config.STEPS, config.INITIAL_LR,
                                                  config.PATIENCE, config.NUM_WORKERS, use_d4))
    logging.info('\n-------')


def train_model(model_name, use_pseudo):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    batch_size = config.BATCH_SIZES[model_name]
    if use_pseudo:
        model_name += '_pseudo'
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(filename='{}/{}'.format(config.LOGS_DIR, model_name), level=logging.INFO,
                        format='%(asctime)s %(message)s')
    model = train_utils.make_model(model_name)
    ids_train = load_train_ids(use_pseudo)
    ids_val = load_valid_ids()

    use_d4 = 'd4' in model_name
    log_config(use_d4)
    run_training(model_name, model, ids_train, ids_val, batch_size, config.STEPS, use_d4)
