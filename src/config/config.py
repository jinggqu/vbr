# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "raw_data_list": [
            "data/data_231121.csv",
            "data/abnormal_data_aug.csv"
        ],
        "raw_folder": "data/raw/",
        "proc_folder": "data/processed/",
    },
    "train": {
        "seed": 42,
        "train_batch_size": 256,
        "val_batch_size": 2048,
        "epoch": 200,
        "train_portion": 0.7,
        "val_portion": 0.2,
        "lr": 2e-5,
        "gamma": 0.985,
        "loss_weight": [  # Different loss weights corresponding to normal and abnormal
            1.0,
            5.0
        ]
    },
    "model": {
        "input_size": 7,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.25,
        "n_classes": 2,
    }
}
