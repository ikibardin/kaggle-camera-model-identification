import glob
import os

import pandas as pd

if __name__ == '__main__':
    fns = glob.glob(os.path.join('../../data/merge', '*/*'))
    df = pd.DataFrame()
    df['filename'] = fns
    os.makedirs('../../tables', exist_ok=True)
    df.to_csv('../../tables/external.csv', index=False)
