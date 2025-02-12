# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 10th Feb 2025

"""Model config in json format"""

CFG = {

    "train": {
        "batch_size": 32,
        "epochs": 2,
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
        "timesteps": 10,
        "CNN_layer_1": 64,
        "CNN_layer_2": 32,
        "LSTM_layer_1": 32,
        "LSTM_layer_2": 32,
        "LSTM_layer_3": 32,
        "Dropout": 0.22,
        "Output_layer": 1,
        "kernel_size": 3,
        "pool_size": 2,
        "num_heads_MHA": 4, #8
        "key_dim_MHA": 32,
        "learning_rate": 1e-4,
        "reduce_lr_factor": 0.95,
        "reduce_lr_patience": 25,
        "early_stop_patience": 30
        },
}