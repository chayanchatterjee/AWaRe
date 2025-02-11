# -*- coding: utf-8 -*-
"""Training code for AWaRe - Attention-boosted Waveform Reconstruction Network"""

'''
 * Copyright (C) 2025 Chayan Chatterjee <chayan.chatterjee@vanderbilt.edu>
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

from model.base_model import BaseModel
from dataloader.dataloader import DataLoader
from model.cnn_lstm import CNN_LSTM
from configs.config import CFG

# External
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from argparse import ArgumentParser
import logging
import os
import random
from matplotlib import pyplot as plt

plt.switch_backend('agg')

tfd = tfp.distributions

def train(Network, X_train_noisy, X_train_pure, output_dir, batch_size, learning_rate, epochs, val_split, 
          reduce_lr_factor, reduce_lr_patience, early_stop_patience, train_from_checkpoint):
    """Trains the model"""
    checkpoint_dir = "checkpoints/Saved_checkpoint"
    checkpoint_directory = "{}/tmp_{}".format(checkpoint_dir, str(hex(random.getrandbits(32))))
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        
    if train_from_checkpoint:
        checkpoint.restore(train_from_checkpoint)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
    callbacks_list = [reduce_lr, early_stopping]

    logging.info('Training started')

    model_history = Network.fit(
        X_train_noisy, X_train_pure,
        epochs=epochs, batch_size=batch_size,
        validation_split=val_split, callbacks=callbacks_list
    )

    checkpoint.save(file_prefix=checkpoint_prefix)
    Network.save(checkpoint_directory)
    
    return model_history

def plot_loss_curves(loss, val_loss, output_dir):
    """Plots loss curves"""
    plt.figure(figsize=(6, 4))
    plt.plot(loss, "r--", label="Loss on training data")
    plt.plot(val_loss, "r", label="Loss on validation data")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(output_dir, dpi=200)


def main():
    parser = ArgumentParser(description="Training script for AWaRe. Written by Chayan Chatterjee.")

    parser.add_argument('-d', '--dataset-file', type=str, help="Path to the file where the datasets are stored.")
    parser.add_argument('-det', '--detector', type=str, help="Detector for which the data is to be fetched (H1/L1/V1).")
    parser.add_argument('-o', '--output-dir', type=str, help="Path to the directory where the outputs will be stored. The directory must exist.")
    parser.add_argument('--val-split', type=int, help="Fraction of training dataset to be used for validation.")
    parser.add_argument('--model-path', type=str, default=None, help="Path where trained model will be saved.")
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Checkpoint path for resuming training from a certain epoch.")
    
    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
        
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    TrainDS = DataLoader(args.detector)
    
    strain_train, signal_train = TrainDS.load_data(args.dataset_file)
    strain_train = TrainDS._preprocess_data(strain_train)

    logging.info('Scaling the amplitudes of the pure signals by 100...')
    signal_train = signal_train / 100.0

    X_train_noisy, X_train_pure = TrainDS.reshape_sequences(strain_train.shape[0], strain_train, signal_train)

    X_train_noisy = X_train_noisy[..., None]
    X_train_pure = X_train_pure[..., None]

    X_train_noisy = X_train_noisy.astype("float32")
    X_train_pure = X_train_pure.astype("float32")
       
    cnn_lstm = CNN_LSTM(CFG)
    Network, checkpoint = cnn_lstm.build(X_train_noisy.shape[1:], args.model_path)

    model_history = train(Network=Network, X_train_noisy=X_train_noisy, X_train_pure=X_train_pure, output_dir=args.output_dir,
                    batch_size=cnn_lstm.batch_size, epochs=cnn_lstm.epochs, val_split=args.val_split,
                    reduce_lr_factor=cnn_lstm.reduce_lr_factor, reduce_lr_patience=cnn_lstm.reduce_lr_patience,
                    early_stop_patience=cnn_lstm.early_stop_patience, train_from_checkpoint=args.checkpoint_path)  
    
    plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'], args.output_dir)  
    

    if __name__=='__main__':
        main()


