# Plot score for signal and background, comparing training and testing
from math import log, sqrt
from typing import Callable

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas
import uproot


def compare_train_test(
    y_pred_train,
    y_train,
    y_pred,
    y_test,
    high_low=(0, 1),
    bins=30,
    xlabel="",
    ylabel="Arbitrary units",
    title="",
    weights_train=np.array([]),
    weights_test=np.array([]),
    density=True,
):
    if weights_train.size != 0:
        weights_train_signal = weights_train[y_train == 1]
        weights_train_background = weights_train[y_train == 0]
    else:
        weights_train_signal = None
        weights_train_background = None
    plt.hist(
        y_pred_train[y_train == 1],
        color="r",
        alpha=0.5,
        range=high_low,
        bins=bins,
        histtype="stepfilled",
        density=density,
        label="S (train)",
        weights=weights_train_signal,
    )  # alpha is transparancy
    plt.hist(
        y_pred_train[y_train == 0],
        color="b",
        alpha=0.5,
        range=high_low,
        bins=bins,
        histtype="stepfilled",
        density=density,
        label="B (train)",
        weights=weights_train_background,
    )

    if weights_test.size != 0:
        weights_test_signal = weights_test[y_test == 1]
        weights_test_background = weights_test[y_test == 0]
    else:
        weights_test_signal = None
        weights_test_background = None
    hist, bins = np.histogram(
        y_pred[y_test == 1],
        bins=bins,
        range=high_low,
        density=density,
        weights=weights_test_signal,
    )
    scale = len(y_pred[y_test == 1]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="S (test)")

    hist, bins = np.histogram(
        y_pred[y_test == 0],
        bins=bins,
        range=high_low,
        density=density,
        weights=weights_test_background,
    )
    scale = len(y_pred[y_test == 0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="B (test)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")


def amsasimov(s, b):  # asimov (or Poisson) significance
    if b <= 0 or s <= 0:
        return 0
    try:
        return sqrt(2 * ((s + b) * log(1 + float(s) / b) - s))
    except ValueError:
        print(1 + float(s) / b)
        print(2 * ((s + b) * log(1 + float(s) / b) - s))
    # return s/sqrt(s+b)


def load_training_file() -> pandas.DataFrame:
    """Load the ATLAS open data for the WW dataset.

    * This data has already been pre-processed into a numpy-like array.
    * It contains both signal and background events.

    The data we return:
    * Is in `pandas` format
    * has positive-only mcWeights
    * has at least 2 leptons.

    This is typically how you will store training data. And
    script will source the original Monte Carlo signal and
    background, and produce a "training file".
    """
    # Load the data and fetch the tree.
    filename = "dataWW_d1.root"
    file = uproot.open(filename)
    tree = file["tree_event"]

    # Next, lets load this data we already know is rect-a-linear
    # into a pandas array.
    d_fall = tree.arrays(library="pd")  # type: ignore

    # Lets look only at events that have positive weights and have
    # at least 2 leptons.

    full_data = d_fall[(d_fall.lep_n == 2) & (d_fall.mcWeight > 0)]

    return full_data


def update_batch(update: Callable, params, rng, opt_state, batch_size: int, x, y):
    """Do one update, but train in batches to keep memory use low.

    Args:
        params (): The NN weights
        rng (): Random number key
        opt_state (): Optimizer state (for adam, etc.)
        batch_size (): Rows in each batch
        x (): Training Data
        y (): Training Target
    """

    # Get the number of data points
    num_data = x.shape[0]

    # Calculate the number of batches per epoch
    num_batches = num_data // batch_size

    # Batch training loop
    for batch_idx in range(num_batches):
        # Get the batch data
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        X_batch = x[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]

        # Compute the loss and gradients
        params, opt_state = update(params, rng, opt_state, X_batch, y_batch)
