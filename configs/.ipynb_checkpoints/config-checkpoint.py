# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 9th June 2024

"""Model config in json format"""

CFG = {

    "train": {
        "batch_size": 512,
        "epochs": 75,
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
        "learning_rate": 1e-4
        },
}