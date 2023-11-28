import json
import os.path

import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        with open(os.path.join(self.config.root_dir, self.config.data.proc_folder, 'data.json'), 'r') as f:
            data = json.load(f)

        # Set the file name type as object to prevent string data truncation by numpy
        files, labels = np.array(list(data.keys()), dtype=object), np.array(list(data.values()))

        return self.train_val_test_split(files=files,
                                         labels=labels,
                                         train_portion=self.config.train.train_portion,
                                         val_portion=self.config.train.val_portion,
                                         shuffle=True)

    def train_val_test_split(self,
                             files: np.ndarray,
                             labels: np.ndarray,
                             train_portion: float = 0.7,
                             val_portion: float = 0.2,
                             shuffle: bool = False) -> dict:
        """
        Split dataset into train, val and test dataset
        """
        if shuffle:
            # Fix random seeds for reproducibility
            # Ensure that seed() and permutation() appear in pairs at the same time
            np.random.seed(self.config.train.seed)
            permutation = np.random.permutation(len(files))

            # Randomize the files and labels in the same order
            files = files[permutation]
            labels = labels[permutation]

            # Save permutation sequence to a text file
            np.savetxt(os.path.join(self.config.root_dir, self.config.data.proc_folder, 'permutation.txt'),
                       permutation, fmt='%d')

        train_size = int(len(files) * train_portion)
        val_size = int(len(files) * val_portion)

        return {
            'train': {
                'files': files[:train_size],
                'labels': labels[:train_size]
            },
            'val': {
                'files': files[train_size:train_size + val_size],
                'labels': labels[train_size:train_size + val_size]
            },
            'test': {
                'files': files[train_size + val_size:],
                'labels': labels[train_size + val_size:]
            }
        }

    def label_classes(self):
        """
        Load all classes and encode them by using LabelEncoder()
        :return:
        """
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.loadtxt(
            fname=os.path.join(self.config.root_dir, self.config.data.proc_folder, 'label_classes.txt'),
            dtype=str
        )
        return label_encoder.classes_
