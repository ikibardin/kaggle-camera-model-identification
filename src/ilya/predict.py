import os
import argparse
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.core import predict_utils


def get_top_n_checkpoints(model_name, num_checkpoints=3):
    all_checkpoints = os.listdir(config.CHECKPOINTS_DIR.format(model_name))
    scores = np.array([name.split('.')[0].split('_')[-1] for name in all_checkpoints], dtype=float)
    indices = np.argpartition(scores, -num_checkpoints)[-num_checkpoints:]
    return [all_checkpoints[idx] for idx in indices]


def get_weights_for_final_prediction():
    weights = []
    for model_name in config.MODELS:
        model_name += '_pseudo'
        weights += get_top_n_checkpoints(model_name)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory with images to predict')
    parser.add_argument('-o', '--output', type=str, help='Path for output file')
    args = parser.parse_args()

    checkpoints_paths = get_weights_for_final_prediction()
    print('Loaded {} checkpoints'.format(len(checkpoints_paths)))
    print('Generating proba for test directory at {}'.format(args.dir))
    proba_df = predict_utils.predict_utils.predict_gmean_ensemble(checkpoints_paths, args.dir)
    cameras = proba_df.drop('fname', axis=1).idxmax(axis=1)
    predictions_df = pd.DataFrame({
        'fname': proba_df['fname'],
        'camera': cameras
    })
    predictions_df.to_csv('../../' + args.output, index=False)
    print('Submission saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
