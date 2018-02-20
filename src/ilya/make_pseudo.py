import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

from pipeline import config
from pipeline.core import predict_utils


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


def get_pseudo_cut(all_df, unalt):
    mask_unalt = pd.Series(['unalt' in name for name in all_df['fname']])
    tmp = all_df[mask_unalt] if unalt else all_df[~mask_unalt]
    idx = []
    for col in all_df.columns[:-1]:
        for _ in range(105):
            id_ = tmp[col].idxmax()
            idx.append(id_)
            tmp = tmp.drop(id_, axis=0)
    pseudo = all_df.iloc[idx]
    return pseudo


def shrink_to_pseudo(all_test_proba_df):
    pseudo_unalt = get_pseudo_cut(all_test_proba_df, unalt=True)
    pseudo_manip = get_pseudo_cut(all_test_proba_df, unalt=False)
    return pseudo_unalt.append(pseudo_manip)


def move_pseudo_to_separate_folder(pseudo_proba_df):
    cats = pseudo_proba_df.drop('fname', axis=1).idxmax(axis=1)
    for c, n in tqdm(zip(cats, pseudo_proba_df['fname'])):
        os.makedirs(config.PSEUDO_DIR + '/{}'.format(c), exist_ok=True)
        shutil.copyfile(config.TEST_DIR + '/{}'.format(n),
                        config.PSEUDO_DIR + '/{}/{}'.format(c, n))
    pseudo_paths = pd.DataFrame({
        'fname': [config.PSEUDO_DIR + '/{}/{}'.format(c, n) for c, n in
                  zip(cats, pseudo_proba_df['fname'])]
    })
    return pseudo_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory with test images')
    args = parser.parse_args()

    checkpoints_paths = get_weights_for_pseudo_labeling()
    print('Loaded {} checkpoints'.format(len(checkpoints_paths)))
    print('Generating proba for pseudo labeling for test directory at {}'.format(args.dir))
    proba_df = predict_utils.predict_utils.predict_gmean_ensemble(checkpoints_paths, args.dir)
    pseudo_proba_df = shrink_to_pseudo(proba_df)
    print('Chosen {} test images for pseudo labeling. Copying them from {} to {}'.format(
        pseudo_proba_df.shape[0], config.TEST_DIR, config.PSEUDO_DIR
    ))
    pseudo_paths = move_pseudo_to_separate_folder(pseudo_proba_df)
    pseudo_paths.to_csv(config.TABLES_DIR + '/pseudo.csv')
    print('Saved .csv file to {}'.format(config.TABLES_DIR + '/pseudo.csv'))


if __name__ == '__main__':
    main()
