import glob
import os
import subprocess
from multiprocessing import Pool

import jpeg4py
import numpy as np
import pandas as pd
import tqdm
from PIL import Image, ExifTags

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

cameras = [
    {
        'name': 'HTC-1-M7',
        'models': ['HTC6500LVW*', 'HTC One*'],
        'software': [None],
        'shapes': [(2688, 1520), (1520, 2688)]
    },
    {
        'name': 'iPhone-4s',
        'models': ['iPhone 4S'],
        'software': [str(_) + '*' for _ in range(10)],
        'shapes': [(2448, 3264), (3264, 2448)]
    },
    {
        'name': 'iPhone-6',
        'models': ['iPhone 6'],
        'software': [str(_) + '*' for _ in range(10)],
        'shapes': [(2448, 3264), (3264, 2448)]
    },
    {
        'name': 'LG-Nexus-5x',
        'models': ['Nexus 5X'],
        'software': ['bullhead*', 'HDR+*'],
        'shapes': [(4032, 3024), (3024, 4032)]
    },
    {
        'name': 'Motorola-Droid-Maxx',
        'models': ['XT1080', 'XT1060', 'XT1030'],
        'software': [None],
        'shapes': [(2432, 4320), (4320, 2432)]
    },
    {
        'name': 'Motorola-Nexus-6',
        'models': ['Nexus 6'],
        'software': [None, 'HDR+*'],
        'shapes': [(3120, 4160), (4160, 3120),
                   (3088, 4160), (4160, 3088)]
    },
    {
        'name': 'Motorola-X',
        'models': ['XT1085', 'XT1092', 'XT1093', 'XT1094',
                   'XT1095', 'XT1096', 'XT1097', 'XT1090', 'XT1091', 'XT1098', 'XT1099',
                   'XT1052', 'XT1053', 'XT1056', 'XT1058'],
        'software': [None],
        'shapes': [(3120, 4160), (4160, 3120),
                   (2340, 4160), (4160, 2340), ]
    },
    {
        'name': 'Samsung-Galaxy-Note3',
        'models': ['SAMSUNG-SM-N900A*', 'SM-N900*'],
        'software': ['N900*'],
        'shapes': [(2322, 4128), (4128, 2322),
                   (3096, 4128), (4128, 3096)]
    },
    {
        'name': 'Samsung-Galaxy-S4',
        'models': ['SCH-I545*', 'GT-I9505*'],
        'software': ['I9505*', 'I545*'],
        'shapes': [(2322, 4128), (4128, 2322)]
    },
    {
        'name': 'Sony-NEX-7',
        'models': ['NEX-7*'],
        'software': ['NEX-7 v*'],
        'shapes': [(4000, 6000), (6000, 4000)]
    }
]

SOURCES = ['merge']


def get_size_quality(filename):
    process = subprocess.Popen(stdout=subprocess.PIPE, args=['identify', '-format', '\'%W,%H,%Q\'', '-quiet', filename])
    out = process.stdout.read(100).decode()[1:-1]
    w, h, q = out.split(sep=',')
    return int(w), int(h), int(q)


def exif(img):
    if img._getexif() is not None:
        return {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in ExifTags.TAGS
        }
    else:
        return None


def sanitize(s):
    return s.strip().replace('\n', ' ').replace('\r', ' ')


def get_attrs(fn):
    try:
        img = jpeg4py.JPEG(fn).decode()
        if img.shape[0] > 0:
            pass
        else:
            return 0, 0, 10, 'None', 'soft'

        w, h, qual = get_size_quality(fn)
        tags = exif(Image.open(fn))
        model = sanitize(tags.get('Model', ''))
        software = sanitize(tags.get('Software', ''))
        return w, h, qual, model, software
    except:
        return 0, 0, 10, 'None', 'soft'


def folder_process(folder):
    fns = glob.glob(os.path.join(folder, '*'))
    with Pool() as p:
        out = p.map(get_attrs, fns)

    return np.array(out), fns


def collect_data():
    df = []
    for src in ['../../downloader/yandex', '../../downloader/flickr']:
        print(src)
        models = glob.glob(os.path.join(src, 'files', '*/*jpg'))
        for n, model in tqdm.tqdm(enumerate(models), total=len(models)):
            print(model)
            out, fns = folder_process(model)
            if len(fns) > 0:
                tdf = pd.DataFrame(out, columns=['w', 'h', 'q', 'model', 'soft'])
                tdf['fns'] = fns
                df.append(tdf)

    df = pd.concat(df)
    for col in ['w', 'h', 'q']:
        df[col] = np.int32(df[col].as_matrix())

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/all.csv', index=False)


def filter_clean_df():
    # df = pd.concat([pd.read_csv('data/all_flckr.csv'), pd.read_csv('data/all_attrs_all.csv')])
    df = pd.read_csv('data/all.csv')
    print(df.head())
    print(df.shape)

    df = df[df['q'] > 90]
    # df = df[df['q'] > 55]
    print(df.shape)

    df.dropna(axis=0, subset=['model'], inplace=True)
    df['soft'].fillna('hvvj', inplace=True)

    print(df.shape)
    out = []
    for cam in cameras:
        name = cam.get('name')
        tdf = []
        for shp in cam.get('shapes'):
            tdf.append(df[(df['w'] == shp[0]) & (df['h'] == shp[1])])

        tdf = pd.concat(tdf)

        tdf2 = []
        for mod in cam.get('models'):
            if '*' in mod:
                mod_tmp = mod[:-1]
                tdf['model'] = [el[:len(mod_tmp)] for el in tdf['model']]
                tdf2.append(tdf[tdf['model'] == mod_tmp])
            else:
                tdf2.append(tdf[tdf['model'] == mod])

        tdf2 = pd.concat(tdf2)
        # print(f'{tdf2.shape[0]} {name}')

        tdf3 = []
        for mod in cam.get('software'):
            if mod is None:
                tdf3.append(tdf2[tdf2['soft'] == 'hvvj'])


            elif '*' in mod:
                mod_tmp = mod[:-1]
                tdf2['soft'] = [el[:len(mod_tmp)] for el in tdf2['soft']]
                tdf3.append(tdf2[tdf2['soft'] == mod_tmp])
            else:
                tdf3.append(tdf2[tdf2['soft'] == mod])

        tdf3 = pd.concat(tdf3)
        print(f'{tdf3.shape[0]} {name}')
        tdf3['class'] = [name] * tdf3.shape[0]

        out.append(tdf3)

    out = pd.concat(out)
    print(f'total: {out.shape[0]}')

    out.to_csv('data/all_filter.csv', index=False)


def merge():
    df = pd.read_csv('data/all_filter.csv')
    for cls in CLASSES:
        dst = os.path.join('../../data', 'merge', cls)
        os.makedirs(dst, exist_ok=True)
        tdf = df[df['class'] == cls]
        for n, fn in tqdm.tqdm(enumerate(tdf['fns'].tolist()), total=tdf.shape[0]):
            os.rename(fn, os.path.join(dst, os.path.basename(fn)))


if __name__ == '__main__':
    collect_data()
    filter_clean_df()
    merge()
