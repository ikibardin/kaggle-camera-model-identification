import argparse
import glob
import re
import csv
from os.path import join
import pandas as pd

from sklearn.utils import class_weight
from tqdm import tqdm
from PIL import Image
from conditional import conditional
from scipy.stats import gmean

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler

from core.utils import *
from core.train_utils import *
from core.custom_dataset import IEEECameraDataset, preprocess_image
from core.custom_scheduler import ReduceLROnPlateau
from core.validation_dataset import ValidationDataset
from mymodels.bninception import bninception

import logging

EXPERIMENT_NAME = 'bninception'

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 64')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('-m', '--model', help='load hdf5 model including weights (and continue training)')
# parser.add_argument('-w', '--weights', help='load hdf5 weights only (and continue training)')
# parser.add_argument('-do', '--dropout', type=float, default=0.3, help='Dropout rate for FC layers')
# parser.add_argument('-doc', '--dropout-classifier', type=float, default=0., help='Dropout rate for classifier')
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
parser.add_argument('-tt', '--test-train', action='store_true', help='Test model on the training set')
parser.add_argument('-cs', '--crop-size', type=int, default=512, help='Crop size')
parser.add_argument('-w', '--workers', type=int, default=8, help='Num workers')
# parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
# parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use: avg|max|none')
# parser.add_argument('-nfc', '--no-fcs', action='store_true', help='Dont add any FC at the end, just a softmax')
# parser.add_argument('-kf', '--kernel-filter', action='store_true', help='Apply kernel filter')
# parser.add_argument('-lkf', '--learn-kernel-filter', action='store_true', help='Add a trainable kernel filter before classifier')
# parser.add_argument('-cm', '--classifier', type=str, default='ResNet50', help='Base classifier model to use')
parser.add_argument('-uiw', '--use-imagenet-weights', action='store_true',
                    help='Use imagenet weights (transfer learning)')
parser.add_argument('-x', '--extra-dataset', action='store_true',
                    help='Use dataset from https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235')
# parser.add_argument('-v', '--verbose', action='store_true', help='Pring debug/verbose info')
parser.add_argument('-e', '--ensembling', type=str, default='arithmetic',
                    help='Type of ensembling: arithmetic|geometric for TTA')
parser.add_argument('-tta', action='store_true', help='Enable test time augmentation')

args = parser.parse_args()

num_workers = args.workers

TRAIN_FOLDER = '../../data/train'
EXTRA_TRAIN_FOLDER = '../../data/external_full'  # not used -> ./good_imgs_train.txt
EXTRA_VAL_FOLDER = '../../data/val_images'  # not used -> ./good_imgs_val.txt
TEST_FOLDER = '../../data/test'

CROP_SIZE = args.crop_size


# def to_ssd1(name):
#     return '/mnt/ssd1/easygold/camera-model-identification/data/' + name[20:]


def load_train_ids():
    kaggle_train_meta = pd.read_csv('../input/new_train_meta.csv')
    kaggle_train_meta = kaggle_train_meta[kaggle_train_meta['fold_id'] != 0]
    ids_train = list(pd.read_csv('../input/cleaner.csv')['fns']) \
                + list(kaggle_train_meta['filename'])
    # ids_train += [to_ssd1(path) for path in ids_train]
    return ids_train


# MAIN
if args.use_imagenet_weights:
    pretrained = 'imagenet'
else:
    pretrained = False
model = bninception(num_classes=len(CLASSES), pretrained=pretrained)
if cuda_is_available:
    model = nn.DataParallel(model).cuda()
if args.model:
    print("Loading model " + args.model)
    state_dict = torch.load(args.model)['state_dict']
    model.load_state_dict(state_dict)

    # e.g. DenseNet201_do0.3_doc0.0_avg-epoch128-val_acc0.964744.hdf5
    # args.classifier = match.group(2)
    # CROP_SIZE = args.crop_size  = model.get_input_shape_at(0)[0][1]

if not (args.test or args.test_train):

    # TRAINING
    # ids = glob.glob(join(TRAIN_FOLDER, '*/*.jpg'))
    # ids.sort()

    if not args.extra_dataset:
        ids_train = load_train_ids()
        ids_val = pd.read_csv('../input/small_val.csv')['filename']
    else:
        raise RuntimeError()  # FIXME
        ids_train = [line.rstrip('\n') for line in open('good_imgs_train.txt')]
        ids_val = [line.rstrip('\n') for line in open('good_imgs_val.txt')]

        classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]
        classes_val_count = np.bincount(classes_val)
        max_classes_val_count = max(classes_val_count)

        # Balance validation dataset by filling up classes with less items from training set (and removing those from there)
        for class_idx in range(N_CLASSES):
            idx_to_transfer = [idx for idx in ids_train \
                               if get_class(idx.split('/')[-2]) == class_idx][
                              :max_classes_val_count - classes_val_count[class_idx]]

            ids_train = list(set(ids_train).difference(set(idx_to_transfer)))

            ids_val.extend(idx_to_transfer)

        # random.shuffle(ids_val)

    print("Training set distribution:")
    print_distribution(ids_train)

    print("Validation set distribution:")
    print_distribution(ids_val)

    classes_train = [get_class(idx.split('/')[-2]) for idx in ids_train]
    class_weight = class_weight.compute_class_weight('balanced', np.unique(classes_train), classes_train)
    classes_val = [get_class(idx.split('/')[-2]) for idx in ids_val]

    weights = [class_weight[i_class] for i_class in classes_train]
    weights = torch.DoubleTensor(weights)
    train_sampler = sampler.WeightedRandomSampler(weights, len(weights))
    train_dataset = IEEECameraDataset(ids_train, crop_size=CROP_SIZE, training=True)

    val_dataset_unalt = ValidationDataset(ids_val, crop_size=CROP_SIZE, mode='unalt')
    val_dataset_manip = ValidationDataset(ids_val, crop_size=CROP_SIZE, mode='manip')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    valid_loaders = {
        'unalt': DataLoader(val_dataset_unalt, batch_size=args.batch_size,
                            num_workers=num_workers, pin_memory=True),  # , collate_fn=default_collate_unsqueeze),
        'manip': DataLoader(val_dataset_manip, batch_size=args.batch_size,
                            num_workers=num_workers, pin_memory=True)  # , collate_fn=default_collate_unsqueeze)
    }
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-9, epsilon=1e-5, verbose=1, mode='min')

    criterion = nn.CrossEntropyLoss()

    best_val_loss = None
    train_and_validate(
        train_loader,
        valid_loaders,
        model,
        optimizer,
        scheduler,
        criterion,
        args.max_epoch,
        1,
        best_val_loss,
        EXPERIMENT_NAME
    )
else:
    # TEST
    if args.test:
        ids = glob.glob(join(TEST_FOLDER, '*.tif'))
    elif args.test_train:
        ids = glob.glob(join(TRAIN_FOLDER, '*/*.jpg'))
    else:
        assert False

    ids.sort()

    match = re.search(r'([^/]*)\.pth', args.model)
    model_name = match.group(1) + ('_tta_' + args.ensembling if args.tta else '')
    csv_name = 'submission_' + model_name + '.csv'

    model.eval()
    with conditional(args.test, open(csv_name, 'w')) as csvfile:

        if args.test:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['fname', 'camera'])
            classes = []
        else:
            correct_predictions = 0

        for i, idx in enumerate(tqdm(ids)):

            img = np.array(Image.open(idx))
            assert img.shape == (512, 512, 3), img.shape
            assert len(img.shape) == 3, img.shape

            if args.test_train:
                img = get_crop(img, 512 * 2, random_crop=False)

            original_img = img

            original_manipulated = np.float32([1. if idx.find('manip') != -1 else 0.])

            sx = img.shape[1] // CROP_SIZE
            sy = img.shape[0] // CROP_SIZE

            if args.test and args.tta:
                transforms = [[], ['orientation']]
            elif args.test_train:
                transforms = [[], ['orientation'], ['manipulation'], ['manipulation', 'orientation']]
            else:
                transforms = [[]]

            img_batch = np.zeros((len(transforms) * sx * sy, CROP_SIZE, CROP_SIZE, 3), dtype=np.float32)
            manipulated_batch = np.zeros((len(transforms) * sx * sy, 1), dtype=np.float32)

            i = 0
            for transform in transforms:
                img = np.copy(original_img)
                manipulated = np.copy(original_manipulated)

                if 'orientation' in transform:
                    assert img is not None
                    img = np.rot90(img, 1, (0, 1))
                if 'manipulation' in transform and not original_manipulated:
                    img = random_manipulation(img)
                    manipulated = np.float32([1.])

                if args.test_train:
                    img = get_crop(img, 512, random_crop=False)

                sx = img.shape[1] // CROP_SIZE
                sy = img.shape[0] // CROP_SIZE

                for x in range(sx):
                    for y in range(sy):
                        _img = np.copy(img[y * CROP_SIZE:(y + 1) * CROP_SIZE, x * CROP_SIZE:(x + 1) * CROP_SIZE])
                        img_batch[i] = preprocess_image(_img)
                        manipulated_batch[i] = manipulated
                        i += 1

            img_batch, manipulated_batch = variable(torch.from_numpy(img_batch)), variable(
                torch.from_numpy(manipulated_batch))
            prediction = nn.functional.softmax(model(img_batch, manipulated_batch)).data.cpu().numpy()
            if prediction.shape[0] != 1:  # TTA
                if args.ensembling == 'geometric':
                    prediction = gmean(prediction, axis=0)  # avoid numerical instability log(0)
                    # prediction = np.sum(prediction, axis=0)
                else:
                    raise NotImplementedError()

            prediction_class_idx = np.argmax(prediction)

            if args.test_train:
                class_idx = get_class(idx.split('/')[-2])
                if class_idx == prediction_class_idx:
                    correct_predictions += 1

            if args.test:
                csv_writer.writerow([idx.split('/')[-1], CLASSES[prediction_class_idx]])
                classes.append(prediction_class_idx)

        if args.test_train:
            print("Accuracy: " + str(correct_predictions / (len(transforms) * i)))

        if args.test:
            print("Test set predictions distribution:")
            print_distribution(None, classes=classes)
            print("Now you are ready to:")
            print("kg submit {}".format(csv_name))
