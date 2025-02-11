# -*- coding: utf-8 -*-
"""Data Loader"""
import pandas as pd
import h5py
import sys
import logging
import numpy as np

log_level = logging.INFO
logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

class DataLoader:
    """Data Loader class"""

    def load_data(self, path):
        """Loads dataset from path"""

        # Check if the file exists in the path
        if os.path.exists(path):
            with h5py.File(path, 'r') as file:
                print(f"Successfully loaded file: {path}")

            # Check if both 'injection_samples' and 'noise_samples' exist in the file
            if 'injection_samples' in file.keys() and 'noise_samples' in file.keys():
                
                # Load strain data from both injection and noise samples
                injection_strain_data = file['injection_samples'][self.det_code + '_strain'][()]
                noise_strain_data = file['noise_samples'][self.det_code + '_strain'][()]
    
                # Attempt to load signal data from injection parameters
                try:
                    signal_data = file['injection_parameters'][self.det_code + '_signal_whitened'][()]
                    noise_signal_data = file['noise_parameters'][self.det_code + '_signal_whitened'][()]
                except KeyError:
                    raise ValueError(f"Whitened pure waveform not found for {self.det_code}.")
        
            else:
                missing_keys = []
                if 'injection_samples' not in file.keys():
                    missing_keys.append("'injection_samples'")
                if 'noise_samples' not in file.keys():
                    missing_keys.append("'noise_samples'")
    
                raise ValueError(f"Missing keys in the file: {', '.join(missing_keys)}.")
                

        else:
            raise FileNotFoundError(f"File not found: {path}")
            

        # Concatenate the arrays
        concatenated_array = np.concatenate((signal_data, noise_signal_data))

        # Shuffle the indices
        shuffled_indices = np.random.permutation(len(concatenated_array))

        # Use the shuffled indices to create a new shuffled array
        shuffled_array_signal = concatenated_array[shuffled_indices]

        # Concatenate the arrays
        concatenated_array_strain = np.concatenate((strain, noise_strain))

        shuffled_array_strain = concatenated_array_strain[shuffled_indices]
    
        strain_data = shuffled_array_strain
        signal_data = shuffled_array_signal
        
        return strain_data, signal_data


    def _preprocess_data(self, data):
        """
        Scales the amplitudes of the signals to lie between -1 and 1.

        This method iterates through each signal in the dataset and normalizes its amplitude. 
        Positive values are scaled by dividing by the maximum value, while negative values are 
        scaled by dividing by the absolute minimum value.

        Args:
            data (np.ndarray): The input data containing noisy signals. Shape should be (n_samples, signal_length).

        Returns:
            np.ndarray: The preprocessed data with amplitudes scaled between -1 and 1.
        """
        logging.info('Scaling the noisy strain data to lie between -1 and 1...')
        new_array = []
        for i in range(data.shape[0]):
            dataset = data[i]
            if dataset.max() != 0.0 and dataset.min() != 0.0:
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                dataset = np.where(dataset > 0, dataset / maximum, dataset / minimum)
            new_array.append(dataset)
        return np.array(new_array)    


    def split_sequence(self, sequence_noisy, sequence_pure, n_steps):
        """
        Splits a univariate sequence into samples for training the model.

        This method takes in a noisy sequence and its corresponding pure sequence, and splits 
        them into smaller sequences of a fixed length (n_steps). Each smaller sequence from 
        the noisy sequence serves as the input, while the value immediately following each 
        smaller sequence in the pure sequence serves as the target output.

        Args:
            sequence_noisy (np.ndarray): The noisy input sequence to be split.
            sequence_pure (np.ndarray): The pure target sequence to be split.
            n_steps (int): The number of time steps in each smaller sequence.

        Returns:
            tuple: A tuple containing two numpy arrays:
                X (np.ndarray): The input sequences.
                y (np.ndarray): The target outputs corresponding to each input sequence.
        """
        X, y = [], []
        for i in range(len(sequence_noisy)):
            end_ix = i + n_steps
            if end_ix > len(sequence_noisy) - 1:
                break
            seq_x, seq_y = sequence_noisy[i:end_ix], sequence_pure[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    
    def reshape_sequences(self, num, data_noisy, data_pure):
        """
        Reshapes data into overlapping sequences for model training.

        This method prepares the dataset by splitting each signal into overlapping 
        subsequences of a fixed length (timesteps). It pads the signals at both ends 
        with zeros to ensure the output sequences match the input length.

        Args:
            num (int): The number of signals in the dataset.
            data_noisy (np.ndarray): The noisy input data.
            data_pure (np.ndarray): The pure target data.

        Returns:
            tuple: A tuple containing two numpy arrays:
                arr_noisy (np.ndarray): The reshaped noisy input sequences.
                arr_pure (np.ndarray): The reshaped pure target sequences.
        """
        logging.info('Splitting the waveforms into overlapping subsequences...')
        n_steps = self.timesteps
        arr_noisy, arr_pure = [], []

        for i in range(num):
            X_noisy, X_pure = data_noisy[i], data_pure[i]
            X_noisy = np.pad(X_noisy, (n_steps, n_steps), 'constant', constant_values=(0, 0))
            X_pure = np.pad(X_pure, (n_steps, n_steps), 'constant', constant_values=(0, 0))
            X, y = self.split_sequence(X_noisy, X_pure, n_steps)
            arr_noisy.append(X)
            arr_pure.append(y)

        return np.asarray(arr_noisy), np.asarray(arr_pure)


    