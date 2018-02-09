import numpy as np
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


def run_training(model_name, model, ids_train, ids_val, crop_size,
                 batch_size, num_workers, initial_lr, max_epoch, patience,
                 steps_per_epoch, d4=False):
    if d4:
        print('Training with d4 transforms')
    print("Training set distribution:")
    utils.print_distribution(ids_train)

    print("Validation set distribution:")
    utils.print_distribution(ids_val)

    classes_train = [utils.get_class(idx.split('/')[-2]) for idx in ids_train]
    class_w = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)

    weights = [class_w[i_class] for i_class in classes_train]
    weights = torch.DoubleTensor(weights)
    train_sampler = sampler.WeightedRandomSampler(weights, steps_per_epoch * batch_size)
    train_dataset = IEEECameraDataset(ids_train, crop_size=crop_size, training=True, more_feats=False, d4=d4)

    val_dataset_unalt = ValidationDataset(ids_val, crop_size=crop_size, mode='unalt', more_feats=False)
    val_dataset_manip = ValidationDataset(ids_val, crop_size=crop_size, mode='manip', more_feats=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    valid_loaders = {
        'unalt': DataLoader(val_dataset_unalt, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True),
        'manip': DataLoader(val_dataset_manip, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    }
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, min_lr=1e-9, epsilon=1e-5, verbose=1,
                                  mode='min')

    criterion = nn.CrossEntropyLoss()

    best_val_loss = None
    train_utils.train_and_validate(
        train_loader,
        valid_loaders,
        model,
        optimizer,
        scheduler,
        criterion,
        max_epoch,
        1,
        best_val_loss,
        model_name
    )
