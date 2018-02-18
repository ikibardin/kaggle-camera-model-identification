import argparse
import pandas as pd

from pipeline.core import predict_utils

WEIGHTS_DIR = '../../weights'

WEIGHTS = ['densenet161_28_0.08377413648371115.pth',
           'densenet161_55_0.08159203971706519.pth',
           'densenet161_45_0.0813179751742137.pth',
           'dpn92_tune_11_0.1398952918197271.pth',
           'dpn92_tune_23_0.12260739478774665.pth',
           'dpn92_tune_29_0.14363511492280367.pth']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory with images to predict')
    parser.add_argument('-o', '--output', type=str, help='Path for output file')
    args = parser.parse_args()

    checkpoints_paths = ['{}/{}'.format(WEIGHTS_DIR, weights_name) for weights_name in WEIGHTS]
    print('Loaded {} checkpoints'.format(len(checkpoints_paths)))
    print('Generating final submission for test directory at {}'.format(args.dir))
    final_proba = predict_utils.predict_gmean_ensemble(checkpoints_paths, args.dir)
    cameras = final_proba.drop('fname', axis=1).idxmax(axis=1)
    submission = pd.DataFrame({
        'fname': final_proba['fname'],
        'camera': cameras
    })
    submission.to_csv(args.output, index=False)
    print('Final submission saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
