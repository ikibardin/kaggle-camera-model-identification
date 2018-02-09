import argparse

import numpy as np
from torch.utils.data.dataloader import default_collate

CLASSES = [
    'HTC-1-M7',
    'iPhone-6',
    'Motorola-Droid-Maxx',
    'Motorola-X',
    'Samsung-Galaxy-S4',
    'iPhone-4s',
    'LG-Nexus-5x',
    'Motorola-Nexus-6',
    'Samsung-Galaxy-Note3',
    'Sony-NEX-7']

EXTRA_CLASSES = [
    'htc_m7',
    'iphone_6',
    'moto_maxx',
    'moto_x',
    'samsung_s4',
    'iphone_4s',
    'nexus_5x',
    'nexus_6',
    'samsung_note3',
    'sony_nex7'
]

N_CLASSES = len(CLASSES)


def get_class(class_name):
    if class_name in CLASSES:
        class_idx = CLASSES.index(class_name)
    elif class_name in EXTRA_CLASSES:
        class_idx = EXTRA_CLASSES.index(class_name)
    else:
        assert False, class_name
    assert class_idx in range(N_CLASSES)
    return class_idx


def print_distribution(ids, classes=None):
    if classes is None:
        classes = [get_class(idx.split('/')[-2]) for idx in ids]
    classes_count = np.bincount(classes)
    for class_name, class_count in zip(CLASSES, classes_count):
        print('{:>22}: {:5d} ({:04.1f}%)'.format(class_name, class_count, 100. * class_count / len(classes)))


def default_collate_unsqueeze(batchs):
    new_batch = []
    for batch in batchs:
        for X, O, y in zip(*batch):
            new_batch.append([X, np.float32([O]), y])
    return default_collate(new_batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
    parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
    parser.add_argument('-cs', '--crop-size', type=int, default=480, help='Crop size')
    parser.add_argument('-w', '--workers', type=int, default=12, help='Num workers')
    parser.add_argument('-g', '--gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true', default=True,
                        help='Use imagenet weights (transfer learning)')
    parser.add_argument('-e', '--ensembling', type=str, default='arithmetic',
                        help='Type of ensembling: arithmetic|geometric for TTA')
    parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')
    parser.add_argument('-p', '--patience', type=int, default=4, help='Patience before lr reduce')
    parser.add_argument('-s', '--steps', type=int, default=1000, help='Steps per epoch')
    return parser.parse_args()
