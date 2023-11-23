import json

import numpy as np
import pandas as pd

from config.config import CFG
from utils.config import Config

import os

config = Config.from_json(CFG)


def split_data():
    root_dir = config.root_dir
    raw_data_dir_list = config.data.raw_data_list

    # Raw data column list:
    # 'ts', 'longitude', 'latitude', 'tgtheight'(always 0), 'course', 'speed', 'fusiontype'(2 or 11 or 1 or 13), 
    # 'srcmmsi', 'isconnecting'(always 0), 'connecttgttype'(always 0), 'withinborder'(always 0 or 1), 
    # 'starttime', 'batchnum'(unique id), 'tracetype'(always 0 or 1)

    print('Raw data splitting started.')

    for raw_data_dir in raw_data_dir_list:
        print(f'\nSplitting raw data file: {raw_data_dir}')

        # Load data and group by batchnum
        data = pd.read_csv(os.path.join(root_dir, raw_data_dir))
        i, row_count = 0, data['batchnum'].unique().shape[0]

        for batch_id, group in data.groupby('batchnum'):
            print(f'Splitted {i} / {row_count}', end='\r')
            # Get same batch id data and save them to a single csv file
            group.to_csv(os.path.join(root_dir, config.data.raw_folder, f'{batch_id}.csv'), index=False)
            i += 1

    print('\nRaw data splitting completed.')


def preprocess():
    # List all files in '/data/raw/' and keep csv files only
    root_dir = config.root_dir
    files = os.listdir(os.path.join(root_dir, config.data.raw_folder))
    files = [file for file in files if file.endswith('.csv')]
    labels = {}
    classes = set([])

    # Need to be kept column list and type:
    # 'ts', 'longitude', 'latitude', 'course', 'speed', 'fusiontype'(2 or 11 or 1 or 13), withinborder

    print('\nData preprocessing started.')

    i, count = 1, len(files)
    for file in files:
        # Load data
        print(f'Preprocessed {i} / {count}', end='\r')
        df = pd.read_csv(os.path.join(root_dir, 'data', 'raw', file))

        # Labels and classes
        labels[file] = int(df['tracetype'][0])
        classes.add(df['tracetype'][0])

        # Drop useless columns
        df.drop(labels=['tgtheight', 'course', 'srcmmsi', 'isconnecting',
                        'connecttgttype', 'starttime', 'batchnum', 'tracetype'],
                axis=1, inplace=True)

        # Split datetime to day of the year and second of the day, then normalize them
        df_time = pd.to_datetime(df['ts'])
        df.insert(0, 'day', df_time.dt.dayofyear / 366.)
        df['ts'] = ((df_time.dt.hour * 60 + df_time.dt.minute) * 60 + df_time.dt.second) / 86400.
        df.rename(columns={'ts': 'time'})

        # Normalize other variables
        df['longitude'] = df['longitude'] / 180.  # West to East (-180° to 180°)
        df['latitude'] = df['latitude'] / 90.  # South to North (-90° to 90°)
        df['speed'] = df['speed'] / 80.  # Knots (0 to 80)
        df['fusiontype'] = df['fusiontype'].map({1: 0, 2: 1, 11: 2, 13: 3}) / 3.

        df.to_csv(os.path.join(root_dir, config.data.proc_folder, file), index=False, header=False)
        i += 1

    # Save labels to /data/processed/data.json
    with open(os.path.join(root_dir, config.data.proc_folder, 'data.json'), 'w') as f:
        json.dump(labels, f, indent=2)

    # Save label classes to /data/processed/label_classes.npy
    with open(os.path.join(root_dir, config.data.proc_folder, 'label_classes.txt'), 'wb') as f:
        np.savetxt(f, list(classes), fmt='%d')

    print('\nData preprocessing completed.')


if __name__ == '__main__':
    split_data()
    preprocess()
