import argparse
import pandas as pd

from pipeline.core import predict_utils

WEIGHTS_DIR = 'weights'

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
    predicts = []

    for i, weight in enumerate(WEIGHTS):
        print('{}. Predicting with model {}'.format(i, weight))
        model = predict_utils.make_model(weights_name=weight, weights_dir=WEIGHTS_DIR)
        predicts.append(predict_utils.predict_test_proba(model, args.dir, use_tta=True, crop_size=480))
    final_proba = predict_utils.get_gmeaned(predicts)
    cameras = final_proba.drop('fname', axis=1).idxmax(axis=1)
    submission = pd.DataFrame({
        'fname': final_proba['fname'],
        'camera': cameras
    })
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
