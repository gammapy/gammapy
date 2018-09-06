# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ....utils.distributions import GeneralRandomArray


def plot_simple_1d():
    import matplotlib.pyplot as plt

    # A 1d example with 5 bins
    pdf = np.array([1, 1, 1, 1, 1])
    r = GeneralRandomArray(pdf)
    indices = r.draw(1000)
    print(indices.dtype, indices.shape, indices.ndim)
    print(indices.min(), indices.max())
    plt.figure()
    plt.hist(indices, 30)
    # plt.show()


def plot_1d():
    """
    A simple example that shows how to compare
    the true to the random distribution
    """
    import matplotlib.pyplot as plt

    # A vector of probabilities
    p = np.array([1, 3, 4, 2, 1])
    nbins = p.size
    i = np.arange(nbins)

    # Draw a random sample
    r = GeneralRandomArray(p)
    nsamples = 1000
    samples = r.draw(nsamples)

    # Overplot expected and observed counts
    # Also draw sqrt(nentries) error bars for the observed counts
    counts_exp = nsamples * p / p.sum()
    counts_obs = np.histogram(samples, bins=nbins)[0]
    counts_obs_err = np.sqrt(counts_obs)

    plt.errorbar(i, counts_obs, counts_obs_err, fmt="bo", label="observed counts")
    plt.plot(counts_exp, "ro", label="expected counts")
    plt.legend()
    plt.xlim(-0.5, nbins - 0.5)
    plt.ylim(-0.1, counts_exp.max() * 1.5)
    # plt.show()


def plot_simple_2d():
    import matplotlib.pyplot as plt

    # A 2d example with 6 bins
    pdf = np.array([1, 1, 1, 1, 1, 1])
    pdf.shape = 2, 3
    r = GeneralRandomArray(pdf)
    indices = r.draw(1000)
    plt.figure()
    counts = np.histogramdd(indices, bins=(pdf.shape))[0]
    plt.imshow(counts, interpolation="nearest")
    plt.colorbar()
    # plt.show()


def plot_2d_example():
    """
    Show how to draw random numbers from a 2D distribution
    given by an array.
    This is e.g. what we need when sampling photons from an
    image representing a brightness distribution.
    """
    import matplotlib.pyplot as plt

    # Generate some 2D array for demonstration.
    # In reality this could e.g. be read from a FITS file.
    shape = (100, 200)
    y, x = np.indices(shape)
    sigma = 10.
    brightness_map = 1 + 3. * np.exp(
        -((x - 50) ** 2 + (y - 50) ** 2) / (2 * sigma ** 2)
    )
    # Generate random positions from this distribution
    r = GeneralRandomArray(brightness_map)
    photon_list = r.draw(1e5)

    # Bin the photons in a count map
    count_map = np.histogramdd(photon_list, bins=shape)[0]

    # Compare count and brightness map
    ratio_map = count_map / brightness_map

    # Plot the distributions
    for map in [brightness_map, count_map, ratio_map]:
        plt.figure()
        plt.imshow(map, interpolation="nearest")
        plt.colorbar()
    # plt.show()
