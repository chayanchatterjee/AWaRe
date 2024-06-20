# AWaRe - Attention-boosted Waveform Reconstruction network
This repository contains code for generating rapid reconstructions of binary black hole gravitational wave signals from noisy LIGO detector data using deep learning. Our model, called AWaRe, or Attention-boosted Waveform Reconstruction network, is based on an Encoder-Decoder architecture with a probabilistic output layer that produces accurate waveform reconstructions of gravitational waves, with associated prediction uncertainties in less than a second. The Encoder model, consisting of convolutional layers takes as input a 1 sec long segment of whitened noisy LIGO strain data, and generates a vector of embeddings which is enhanced by a multi-head self-attention module. The Decoder network consisting of a stack of Long Short-Term Memory layers uses the embeddings to produce the whitened gravitational wave reconstruction.  \\

The figure below

## Dependencies

This package requires the following python packages:
- numpy
- pandas
- matplotlib
- argparse

## Installation

To install this package run:
```
pip install .
```

## Usage

To run the code use the following command:
```
filter_and_plot
```

This will create a `pulsar_plot.png` file in the current directory.

## Citation

If you use this code please give credit by citing [Swainston et al 2023](link_to_paper)
