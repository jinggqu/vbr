# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "raw_data": "data/ai_trace_data.csv",
        "raw_folder": "data/raw/",
        "proc_folder": "data/processed/",
    },
    "train": {
        "seed": 42,
        "batch_size": 16,
        "epoch": 50,
        "train_portion": 0.7,
        "val_portion": 0.2,
        "lr": 1e-4
    },
    "model": {
        "input_size": 7,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.25,
        "n_classes": 2,
    }
}
