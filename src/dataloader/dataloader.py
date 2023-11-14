import json
import os.path

import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_classes(config):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.loadtxt(os.path.join(config.root_dir,
                                                     config.data.proc_folder,
                                                     'label_classes.txt'),
                                        dtype=str)
    return label_encoder.classes_


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, inference: bool = False):
        with open(os.path.join(self.config.root_dir, self.config.data.proc_folder, 'data.json'), 'r') as f:
            data = json.load(f)
        files, labels = np.array(list(data.keys())), np.array(list(data.values()))
        return self.train_val_test_split(files=files,
                                         labels=labels,
                                         train_portion=self.config.train.train_portion,
                                         val_portion=self.config.train.val_portion,
                                         shuffle=True,
                                         inference=inference)

    def train_val_test_split(self,
                             files: np.ndarray,
                             labels: np.ndarray,
                             train_portion: float = 0.7,
                             val_portion: float = 0.2,
                             shuffle: bool = False,
                             inference: bool = False) -> dict:
        """
        Split dataset into train, val and test dataset
        """
        if shuffle:
            np.random.seed(self.config.train.seed)
            permutation = np.random.permutation(len(files))
            np.savetxt(os.path.join(self.config.root_dir, self.config.data.proc_folder, 'permutation.txt'),
                       permutation, fmt='%d')
            files = files[permutation]
            labels = labels[permutation]

        train_size = int(len(files) * train_portion)
        val_size = int(len(files) * val_portion)

        if not inference:
            return {
                'train': {
                    'files': files[:train_size],
                    'labels': labels[:train_size]
                },
                'val': {
                    'files': files[train_size:train_size + val_size],
                    'labels': labels[train_size:train_size + val_size]
                }
            }
        else:
            return {
                'test': {
                    'files': files[train_size + val_size:],
                    'labels': labels[train_size + val_size:]
                }
            }
