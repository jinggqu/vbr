# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "raw_data": "data/data_231121.csv",
        "raw_folder": "data/raw/",
        "proc_folder": "data/processed/",
    },
    "train": {
        "seed": 42,
        "train_batch_size": 128,
        "val_batch_size": 1024,
        "epoch": 10,
        "train_portion": 0.7,
        "val_portion": 0.2,
        "lr": 1e-4,
        "gamma": 0.99,
        "loss_weight": [1.0, 5.0]       # Different loss weights corresponding to normal and abnormal
    },
    "model": {
        "input_size": 7,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.25,
        "n_classes": 2,
    }
}
