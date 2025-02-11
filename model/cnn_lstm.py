# -*- coding: utf-8 -*-
"""CNN-LSTM Model"""

'''
 * Copyright (C) 2024 Chayan Chatterjee <chayan.chatterjee@vanderbilt.edu>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
'''

####################################################### IMPORTS #################################################################

# Internal
from model.base_model import BaseModel
from dataloader.dataloader import DataLoader

# External
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import os
import logging
import random
from matplotlib import pyplot as plt

plt.switch_backend('agg')

tfd = tfp.distributions

log_level = logging.INFO
logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

#################################################################################################################################

class TimeDistributedMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim
        )

    def call(self, inputs):
        # Extract the dynamic shape
        shape = tf.shape(inputs)  # Shape is a Tensor, not a tuple
        batch_size = shape[0]
        num_subsequences = shape[1]
        subsequence_length = shape[2]
        features = shape[3]

        # Reshape for MultiHeadAttention
        reshaped_inputs = tf.reshape(
            inputs, [batch_size * num_subsequences, subsequence_length, features]
        )

        # Compute attention
        attention_output = self.multi_head_attention(
            query=reshaped_inputs, key=reshaped_inputs, value=reshaped_inputs
        )

        # Reshape back to original format
        output_shape = [batch_size, num_subsequences, subsequence_length, features]
        return tf.reshape(attention_output, output_shape)

        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_heads": self.num_heads, "key_dim": self.key_dim}


class CNN_LSTM(BaseModel):
    """CNN-LSTM Model Class"""
    
    def __init__(self, config):
        super().__init__(config)
        self._set_params()

    def _set_params(self):
        """Set model parameters from config"""
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epochs
        self.lr = self.config.model.learning_rate
        self.reduce_lr_factor = self.config.model.reduce_lr_factor
        self.reduce_lr_patience = self.config.model.reduce_lr_patience
        self.early_stop_patience = self.config.model.early_stop_patience
        self.timesteps = self.config.model.timesteps

        self.cnn_filters_1 = self.config.model.layers.CNN_layer_1
        self.cnn_filters_2 = self.config.model.layers.CNN_layer_2
        self.lstm_1 = self.config.model.layers.LSTM_layer_1
        self.lstm_2 = self.config.model.layers.LSTM_layer_2
        self.lstm_3 = self.config.model.layers.LSTM_layer_3
        self.kernel_size = self.config.model.layers.kernel_size
        self.pool_size = self.config.model.layers.pool_size
        self.dropout = self.config.model.layers.Dropout
        self.num_heads = self.config.model.layers.num_heads_MHA
        self.key_dim = self.config.model.layers.key_dim_MHA

    @staticmethod
    def negloglik(y, rv_y):
        """Negative log likelihood loss function"""
        return -rv_y.log_prob(y)

    def build(self, input_shape, model_path):
        """Builds and compiles the model"""

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.BatchNormalization()(inputs)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_1, self.kernel_size, padding='same', activation='relu'))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_2, self.kernel_size, padding='same', activation='relu'))(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = TimeDistributedMultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_1, activation='tanh', return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation='tanh', return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_3, activation='tanh', return_sequences=True))(x)

        x = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(1))(x)
        outputs = tfp.layers.IndependentNormal(1)(x)

        self.model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(optimizer=optimizer, loss=self.negloglik, metrics=['accuracy'])

        self.model.summary()

        logging.info('Model was built successfully')

        if model_path:
            self.model = self.load_saved_model(model_path)
            
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)        

        return self.model, checkpoint

    def load_saved_model(self, model_path):
        """Loads a saved model"""

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model = tf.keras.models.load_model(self.model_path, custom_objects={
            'TimeDistributedMultiHeadAttention': self.TimeDistributedMultiHeadAttention,
            'IndependentNormal': tfp.layers.IndependentNormal,
            'negloglik': self.negloglik
        })
        model.compile(optimizer=optimizer, loss=self.negloglik, metrics=['accuracy'])
        model.summary()

        return model
    


            
