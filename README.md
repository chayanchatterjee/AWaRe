# AWaRe - Attention-boosted Waveform Reconstruction network
This repository contains code for generating rapid reconstructions of binary black hole gravitational wave signals from noisy LIGO detector data using deep learning. Our model, called AWaRe, or Attention-boosted Waveform Reconstruction network, is based on an Encoder-Decoder architecture with a probabilistic output layer that produces accurate waveform reconstructions of gravitational waves, with associated prediction uncertainties in less than a second. The Encoder model, consisting of convolutional layers takes as input a 1 sec long segment of whitened noisy LIGO strain data, and generates a vector of embeddings which is enhanced by a multi-head self-attention module. The Decoder network, consisting of a stack of Long Short-Term Memory layers, uses the embeddings to produce the reconstructed whitened waveform.

The architecture of the AWaRe model is shown below: ![below](AWaRe.png):



## Dependencies

This package requires the following packages:
- numpy
- pandas
- matplotlib
- tensorflow
- tensorflow-probability
- scipy
- pycbc

## Installation

To install this package, first clone this repository using:
```
git clone https://github.com/chayanchatterjee/AWaRe.git
```
Then create a Python virtual environment for AWaRe and activate it:
```
python -m venv aware-env
source aware-env/bin/activate
```
Install AWaRe in this virtual environment by running:

```
cd AWaRe
pip install -e .
```

## Usage

To run the code and generate reconstruction plots, follow these steps:

1. To generate reconstruction of both H1 and L1 signals together:
```
aware-evaluate --test_filename test_data.hdf --test_index 0 --detector both --add_zoom_plot 1
```
This command will read the input strain data from ```test_data.hdf``` located in the folder ```evaluation/Test_data``` and plot the reconstruction for the data at index ```0``` for ```both``` H1 and L1 detectors. The entry ```1``` at the end means, we want to plot a zoomed-in version of the plots. The program will then prompt you to enter the number of seconds before and after the merger you want to zoom in.

The plot for the reconstruction will then be saved in the directory ```evaluation/Plots```. 

## Citation

If you use this code please give credit by citing the following papers:
1. [Chatterjee and Jani 2024](https://arxiv.org/abs/2403.01559).
2. [Chatterjee and Jani 2024](https://arxiv.org/abs/2406.06324).
