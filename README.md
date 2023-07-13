# PathLossModelEstimator

This repository contains the implementation of an algorithm to estimate the path loss exponent, transmitted power, and Gaussian noise variance in a wireless channel using drive test data.

## Description

The PathLossModelEstimator uses field measurements (received signal strength and geographical coordinates) to estimate key parameters of the wireless channel. The parameters estimated are:

- Transmitted Power (P0)
- Path Loss Exponent (beta)
- Gaussian Noise Variance

The estimation is based on a log-distance path loss model and a linear regression method.

## Installation

To clone and run this application, you'll need [Git](https://git-scm.com) and [Python](https://www.python.org/downloads/) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone https://github.com/AliNazariii/path-loss-model-estimator

# Go into the repository
$ cd PathLossModelEstimator

# Install dependencies
$ pip install -r requirements.txt
```

## Usage
Add instructions on how to use your project for the end-user.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
