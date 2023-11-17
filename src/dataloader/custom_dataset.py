import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, data):
        self.config = config
        self.files = data['files']
        self.labels = data['labels']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        input_data = torch.from_numpy(np.loadtxt(os.path.join(self.config.root_dir,
                                                              self.config.data.proc_folder,
                                                              self.files[index]),
                                                 delimiter=',', dtype=np.float32))
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return input_data, label
