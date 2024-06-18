import numpy as np
import h5py
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import seaborn as sns
import os
import random
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import argparse
import sys
from pycbc.filter import match, overlap
from pycbc.types.timeseries import TimeSeries, FrequencySeries
from scipy.fft import fft, ifft
import matplotlib_latex_bridge as mlb
mlb.setup_page(**mlb.formats.article_letterpaper_10pt_singlecolumn)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

tfd = tfp.distributions

device_type = 'GPU'
n_gpus = 4
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

LOG = get_logger('cnn-lstm')

def Overlap_calc(hp, sp, psd):
    """
    Calculate the match and overlap between two signals.

    Args:
        hp (TimeSeries): The first signal.
        sp (TimeSeries): The second signal.
        psd (FrequencySeries): The power spectral density of the noise.

    Returns:
        float: The match value.
    """
    f_low = 20
    m, i = match(hp, sp, psd=psd, low_frequency_cutoff=f_low)
    o = overlap(hp, sp, psd=psd, low_frequency_cutoff=f_low)
    return m

def _preprocess_data(data):
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
    LOG.info('Scaling the noisy strain data to lie between -1 and 1...')
    new_array = []
    for i in range(data.shape[0]):
        dataset = data[i]
        if dataset.max() != 0.0 and dataset.min() != 0.0:
            maximum = np.max(dataset)
            minimum = np.abs(np.min(dataset))
            dataset = np.where(dataset > 0, dataset / maximum, dataset / minimum)
        new_array.append(dataset)
    return np.array(new_array)

def split_sequence(sequence_noisy, n_steps):
    """
    Splits a univariate sequence into samples for training the model.

    This method takes in a noisy sequence and splits them into smaller sequences of a 
    fixed length (n_steps). Each smaller sequence from the noisy sequence serves as the 
    input.

    Args:
        sequence_noisy (np.ndarray): The noisy input sequence to be split.
        n_steps (int): The number of time steps in each smaller sequence.

    Returns:
        np.ndarray: The split sequences.
    """
    X, y = [], []
    for i in range(len(sequence_noisy)):
        end_ix = i + n_steps
        if end_ix > len(sequence_noisy) - 1:
            break
        seq_x = sequence_noisy[i:end_ix]
        X.append(seq_x)
    return np.array(X)

def reshape_sequences(num, data_noisy, timesteps):
    """
    Reshapes data into overlapping sequences for model training.

    This method prepares the dataset by splitting each signal into overlapping 
    subsequences of a fixed length (timesteps). It pads the signals at both ends 
    with zeros to ensure the output sequences match the input length.

    Args:
        num (int): The number of signals in the dataset.
        data_noisy (np.ndarray): The noisy input data.
        timesteps (int): The number of timesteps for each sequence.

    Returns:
        np.ndarray: The reshaped noisy input sequences.
    """
    LOG.info('Splitting the waveforms into overlapping subsequences...')
    arr_noisy = []
    for i in range(num):
        X_noisy = data_noisy[i]
        X_noisy = np.pad(X_noisy, (timesteps, timesteps), 'constant', constant_values=(0, 0))
        X = split_sequence(X_noisy, timesteps)
        arr_noisy.append(X)
    return np.asarray(arr_noisy)

def reshape_and_print(data):
    """
    Reshapes arrays to fit into Keras model and prints their shapes.

    Args:
        data (np.ndarray): The input data to reshape.

    Returns:
        np.ndarray: The reshaped data.
    """
    LOG.info('Reshaping the data into the correct shapes...')
    data = data[..., None]
    print('Input data shape:', data.shape)
    data = data.astype("float32")
    return data

class TimeDistributedMultiHeadAttention(tf.keras.layers.Layer):
    """
    A custom Keras layer that applies multi-head attention to time-distributed inputs.
    """
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        shape = tf.shape(inputs)
        static_shape = inputs.shape
        batch_size, num_subsequences, subsequence_length = shape[0], static_shape[1], static_shape[2]
        features = static_shape[3]
        reshaped_inputs = tf.reshape(inputs, [-1, subsequence_length, features])
        attention_output = self.multi_head_attention(reshaped_inputs, reshaped_inputs)
        output_shape = [-1, num_subsequences, subsequence_length, features]
        return tf.reshape(attention_output, output_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "num_heads": self.num_heads, "key_dim": self.key_dim}

class NegativeLogLikelihood:
    """
    A custom loss function that calculates the negative log likelihood.
    """
    def __call__(self, y, rv_y):
        return -rv_y.log_prob(y)

def predict_with_uncertainty(x_test, model):
    """
    Predicts with uncertainty using a trained model.

    Args:
        x_test (np.ndarray): The test input data.
        model (tf.keras.Model): The trained model.

    Returns:
        tuple: The mean predictions, lower bound, and upper bound.
    """
    distribution = model(x_test[np.newaxis, ...])
    mean_preds = distribution.mean().numpy().squeeze()
    std_preds = distribution.stddev().numpy().squeeze()
    lower_bound = mean_preds - 1.645 * std_preds
    upper_bound = mean_preds + 1.645 * std_preds
    return mean_preds, lower_bound, upper_bound

def evaluate_model(strain, model_path):
    """
    Evaluates the model on test data.

    Args:
        strain (np.ndarray): The input strain data.
        model_path (str): The path to the trained model.

    Returns:
        tuple: The mean, lower bound, and upper bound predictions.
    """
    negloglik = NegativeLogLikelihood()
    model = tf.keras.models.load_model(model_path, custom_objects={'TimeDistributedMultiHeadAttention': TimeDistributedMultiHeadAttention, 'IndependentNormal': tfp.layers.IndependentNormal, 'negloglik': negloglik})
    LOG.info(f'Generating reconstructions for test dataset')
    mean, lower, upper = [], [], []
    for i in range(strain.shape[0]):
        mean_preds, lower_bound_preds, upper_bound_preds = predict_with_uncertainty(strain[i], model)
        mean.append(mean_preds * 100.0)
        lower.append(lower_bound_preds * 100.0)
        upper.append(upper_bound_preds * 100.0)
    mean = np.array(mean)
    lower = np.array(lower)
    upper = np.array(upper)
    return mean, lower, upper

def read_data(file_path, det, index):
    """
    Reads data from an HDF5 file for the specified detector and index.

    Args:
        file_path (str): The path to the HDF5 file.
        det (str): The detector name ('H1', 'L1', or 'both').
        index (int): The index of the sample to read.

    Returns:
        tuple: The strain data, signal data, and PSD data.
    """
    f1 = h5py.File(file_path, 'r')
    if det != 'both':
        detector = {'H1': 'h1', 'L1': 'l1'}
        strain_data = f1['injection_samples'][detector[det] + '_strain'][index]
        signal_data = f1['injection_parameters'][detector[det] + '_signal_whitened'][index]
        psd = f1['injection_parameters']['psd_noise_' + detector[det]][index]
    else:
        detectors = ['h1', 'l1']
        strain_data = {}
        signal_data = {}
        psd = {}
        for detector in detectors:
            strain_data[detector] = f1['injection_samples'][detector + '_strain'][index]
            signal_data[detector] = f1['injection_parameters'][detector + '_signal_whitened'][index]
            psd[detector] = f1['injection_parameters']['psd_noise_' + detector][index]
    f1.close()
    return strain_data, signal_data, psd

def plot_reconstructed(strain_data, mean_reconstruction, lower_90, upper_90, signal_data, psd_data, before, after, index, add_zoom, detector):
    """
    Plots the reconstructed signal, the original strain data, and the confidence intervals.

    Args:
        strain_data (dict): The strain data.
        mean_reconstruction (dict): The mean reconstructed signal.
        lower_90 (dict): The lower 90% confidence interval.
        upper_90 (dict): The upper 90% confidence interval.
        signal_data (dict): The original signal data.
        psd_data (dict): The power spectral density data.
        before (float): Seconds before the event for zoomed plots.
        after (float): Seconds after the event for zoomed plots.
        index (int): The index of the sample.
        add_zoom (bool): Whether to add a zoomed plot.
        detector (str): The detector name ('H1', 'L1', or 'both').

    Returns:
        None
    """
    time_buffer = 1
    sample_rate = 2048
    tlen = int(sample_rate * time_buffer)
    flen = tlen // 2 + 1
    delta_f = 1.0 / time_buffer

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # Determine the number of subplots based on the detector
    if detector == 'both':
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = np.atleast_1d(axes)

    palette_flare = sns.color_palette("flare", n_colors=6)
    palette_crest = sns.color_palette("crest", n_colors=6)

    def pad_to_match_length(data, target_length):
        """
        Pads the data to match the target length.

        Args:
            data (np.ndarray): The data to pad.
            target_length (int): The target length.

        Returns:
            np.ndarray: The padded data.
        """
        if target_length > len(data):
            return np.pad(data, (0, target_length - len(data)), 'constant')
        else:
            return data

    count = 0

    for ax, det in zip(axes, strain_data):
        strain = strain_data[det]
        decoded_signal = mean_reconstruction[det].squeeze()
        lower_90_det = lower_90[det].squeeze()
        upper_90_det = upper_90[det].squeeze()
        pure_signal = signal_data[det]
        psd = psd_data[det]

        max_index = np.argmax(abs(pure_signal))
        start_time = -max_index / (sample_rate * 1.0)
        end_time = time_buffer - max_index / (sample_rate * 1.0)
        time = np.linspace(start_time, end_time, len(decoded_signal))

        pure_signal_padded = pad_to_match_length(pure_signal, len(decoded_signal))
        strain_padded = pad_to_match_length(strain, len(decoded_signal))
        lower_90_padded = pad_to_match_length(lower_90_det, len(decoded_signal))
        upper_90_padded = pad_to_match_length(upper_90_det, len(decoded_signal))

        ax.plot(time, strain_padded * 1.5, label=f'{det.upper()} Whitened strain data', color='lightgrey', alpha=0.4)
        ax.plot(time, decoded_signal, label=f'{det.upper()} AWaRe mean reconstruction', color=palette_flare[1])
        ax.plot(time, pure_signal_padded, label=f'{det.upper()} Injection waveform', color=palette_crest[3])
        ax.fill_between(time, lower_90_padded, upper_90_padded, label=f'{det.upper()} AWaRe 90% C.I.', color=palette_flare[1], alpha=0.5)
        ax.set_title(det.upper() + ' reconstruction', fontsize=16)

        count = count + 1

        if add_zoom:
            zoom_width = 0.5
            zoom_height = 0.5
            zoom_x = 1.1
            if count == 1:
                zoom_y = 0.55
            elif count == 2:
                zoom_y = 0.01

            # Define the region of interest for zooming
            x1, x2 = -0.05, 0.05

            low_index = max(max_index - int(before * sample_rate), 0)
            upper_index = min(max_index + int(after * sample_rate), sample_rate)

            pure_signal_zoom = pure_signal[low_index:upper_index]
            decoded_signal_zoom = decoded_signal[low_index:upper_index]
            strain_zoom = strain[low_index:upper_index]
            upper_90_zoom = upper_90_det[low_index:upper_index]
            lower_90_zoom = lower_90_det[low_index:upper_index]
            time_zoom = time[low_index:upper_index]

            ax_zoom = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])
            ax_zoom.plot(time_zoom, strain_zoom * 1.5, label=f'{det.upper()} Whitened strain data', color='lightgrey', alpha=0.4)
            ax_zoom.plot(time_zoom, decoded_signal_zoom, label=f'{det.upper()} AWaRe mean reconstruction', color=palette_flare[1])
            ax_zoom.plot(time_zoom, pure_signal_zoom, label=f'{det.upper()} Injection waveform', color=palette_crest[3])
            ax_zoom.fill_between(time_zoom, lower_90_zoom, upper_90_zoom, color=palette_flare[1], alpha=0.5)
            ax_zoom.set_xlim(x1, x2)
            ax_zoom.tick_params(axis='y', labelsize=16)
            ax_zoom.tick_params(axis='x', labelsize=16)
            mark_inset(ax, ax_zoom, loc1=2, loc2=4, fc="none", ec="0.5")

        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel('Strain amplitude', fontsize=16)
        ax.set_xlabel('Time from merger (s)', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)

    if detector == 'both':
        axes[0].legend(loc='lower left', fontsize=16)
        axes[1].legend(loc='lower left', fontsize=16)
    else:
        axes[0].legend(loc='lower left', fontsize=16)

    output_dir = 'evaluation/Plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f'Reconstruction_comparison_{index}_{detector}_det_b={before}_a={after}_with_strain.png'), bbox_inches='tight', dpi=400)
    print('Plot generated!')

    # Calculate and print the overlap for each detector
    for det in strain_data:
        pure_signal = signal_data[det].astype('float64')
        decoded_signal = mean_reconstruction[det].squeeze().astype('float64')
        psd = psd_data[det].astype('float64')
        pure_signal_padded = pad_to_match_length(pure_signal, len(decoded_signal))

        X_test_pure_ts = TimeSeries(pure_signal_padded, delta_t=1.0 / 2048)
        decoded_signal_ts = TimeSeries(decoded_signal, delta_t=1.0 / 2048)
        psd = FrequencySeries(psd, delta_f=delta_f)
        X_test_pure_ts.resize(tlen)
        decoded_signal_ts.resize(tlen)
        psd.resize(flen)
        match = Overlap_calc(X_test_pure_ts, decoded_signal_ts, psd)
        print(f'{det.upper()} Overlap = {np.round(match, 2)}')

def main():
    """
    Main function to parse arguments and call the appropriate functions to read data,
    preprocess it, evaluate the model, and plot the results.
    """
    parser = argparse.ArgumentParser(description="Generate the reconstructed GW waveform from AWaRe and plot the reconstruction and the actual injection")
    parser.add_argument("test_filename", type=str, help="Provide the name of the test data file")
    parser.add_argument("test_index", type=int, help="Provide the index of the test sample")
    parser.add_argument("detector", type=str, help="The detector name ('H1'/'L1/both')", default='L1')
    parser.add_argument("add_zoom_plot", type=int, help="Add a zoom plot or not? 0=False, 1=True", default=0)
    args = parser.parse_args()
    
    if args.detector not in ['H1', 'L1', 'both']:
        print(f"Error: Detector must be one of ['H1', 'L1'] or both.")
        sys.exit(1)

    if args.add_zoom_plot == 1:
        before = float(input("Enter seconds before the event for zoomed plots (<= 0.8): "))
        after = float(input("Enter seconds after the event for zoomed plots (<= 0.2): "))
        if before > 0.8:
            print("Error: Seconds before merger cannot be greater than 0.8.")
            sys.exit(1)
        if after > 0.2:
            print("Error: Seconds after merger cannot be greater than 0.2.")
            sys.exit(1)
    else:
        before, after = 0.8, 0.2

    file_directory = 'evaluation/Test_data/'
    file_path = os.path.join(file_directory, args.test_filename)

    if not os.path.isfile(file_path):
        print(f"{args.test_filename} does not exist in {file_directory}")
        sys.exit(1)
    else:
        strain_data, signal_data, psd_data = read_data(file_path, args.detector, args.test_index)
        model_path = 'model/Saved_models/Trained_model.h5'

        if args.detector == 'both':
            mean_reconstruction = {}
            lower_90 = {}
            upper_90 = {}
            for det in ['h1', 'l1']:
                strain = _preprocess_data(strain_data[det][None, :])
                strain = reshape_sequences(strain.shape[0], strain, 10)
                strain = reshape_and_print(strain)
                mean_reconstruction[det], lower_90[det], upper_90[det] = evaluate_model(strain, model_path)
            plot_reconstructed(strain_data, mean_reconstruction, lower_90, upper_90, signal_data, psd_data, before, after, args.test_index, args.add_zoom_plot, args.detector)
        else:
            det = args.detector.lower()
            strain = _preprocess_data(strain_data[None, :])
            strain = reshape_sequences(strain.shape[0], strain, 10)
            strain = reshape_and_print(strain)
            mean_reconstruction, lower_90, upper_90 = evaluate_model(strain, model_path)
            plot_reconstructed({det: strain_data}, {det: mean_reconstruction}, {det: lower_90}, {det: upper_90}, {det: signal_data}, {det: psd_data}, before, after, args.test_index, args.add_zoom_plot, det)
    
if __name__ == "__main__":
    main()
