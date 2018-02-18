import os
import argparse
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.core import predict_utils

WEIGHTS_DIR = '../../weights'

WEIGHTS = ['densenet161_28_0.08377413648371115.pth',
           'densenet161_55_0.08159203971706519.pth',
           'densenet161_45_0.0813179751742137.pth',
           'dpn92_tune_11_0.1398952918197271.pth',
           'dpn92_tune_23_0.12260739478774665.pth',
           'dpn92_tune_29_0.14363511492280367.pth']


def get_top_n_checkpoints(model_name, num_checkpoints=3):
    all_checkpoints = os.listdir(config.CHECKPOINTS_DIR.format(model_name))
    scores = np.array([name.split('.')[0].split('_')[-1] for name in all_checkpoints], dtype=float)
    indices = np.argpartition(scores, -num_checkpoints)[-num_checkpoints:]
    return [all_checkpoints[idx] for idx in indices]


def get_weights_for_pseudo_labeling():
    weights = []
    for model_name in config.MODELS:
        weights += get_top_n_checkpoints(model_name)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory with test images')
    args = parser.parse_args()

    checkpoints_paths = get_weights_for_pseudo_labeling()
    print('Loaded {} checkpoints'.format(len(checkpoints_paths)))
    print('Generating proba for pseudo labeling for test directory at {}'.format(args.dir))
    proba = predict_utils.predict_utils.predict_gmean_ensemble(checkpoints_paths, args.dir)


if __name__ == '__main__':
    main()
