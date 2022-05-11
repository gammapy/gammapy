# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions to deal with arrays and quantities."""
import numpy as np
import scipy.ndimage
import scipy.signal
from astropy.convolution import Gaussian2DKernel

__all__ = [
    "array_stats_str",
    "round_up_to_even",
    "round_up_to_odd",
    "shape_2N",
    "shape_divisible_by",
    "symmetric_crop_pad_width",
]


def is_power2(n):
    """Check if an integer is a power of 2."""
    return (n > 0) & ((n & (n - 1)) == 0)


def array_stats_str(x, label=""):
    """Make a string summarising some stats for an array.

    Parameters
    ----------
    x : array-like
        Array
    label : str, optional
        Label

    Returns
    -------
    stats_str : str
        String with array stats
    """
    x = np.asanyarray(x)

    ss = ""
    if label:
        ss += f"{label:15s}: "

    min = x.min()
    max = x.max()
    size = x.size

    fmt = "size = {size:5d}, min = {min:6.3f}, max = {max:6.3f}\n"
    ss += fmt.format(**locals())

    return ss


def shape_2N(shape, N=3):
    """
    Round a given shape to values that are divisible by 2^N.

    Parameters
    ----------
    shape : tuple
        Input shape.
    N : int (default = 3), optional
        Exponent of two.

    Returns
    -------
    new_shape : Tuple
        New shape extended to integers divisible by 2^N
    """
    shape = np.array(shape)
    new_shape = shape + (2**N - np.mod(shape, 2**N))
    return tuple(new_shape)


def shape_divisible_by(shape, factor):
    """
    Round a given shape to values that are divisible by factor.

    Parameters
    ----------
    shape : tuple
        Input shape.
    factor : int
        Divisor.

    Returns
    -------
    new_shape : Tuple
        New shape extended to integers divisible by factor
    """
    shape = np.array(shape)
    new_shape = shape + (shape % factor)
    return tuple(new_shape)


def round_up_to_odd(f):
    """Round float to odd integer

    Parameters
    ----------
    f : float
        Float value

    Returns
    -------
    int : int
        Odd integer
    """
    return (np.ceil(f) // 2 * 2 + 1).astype(int)


def round_up_to_even(f):
    """Round float to even integer

    Parameters
    ----------
    f : float
        Float value

    Returns
    -------
    int : int
        Odd integer
    """
    return (np.ceil(f + 1) // 2 * 2).astype(int)


def symmetric_crop_pad_width(shape, new_shape):
    """
    Compute symmetric crop or pad width.

    To obtain a new shape from a given old shape of an array.

    Parameters
    ----------
    shape : tuple
        Old shape
    new_shape : tuple or str
        New shape
    """
    xdiff = abs(shape[1] - new_shape[1])
    ydiff = abs(shape[0] - new_shape[0])

    if (np.array([xdiff, ydiff]) % 2).any():
        raise ValueError(
            "For symmetric crop / pad width, difference to new shape "
            "must be even in all axes."
        )

    ywidth = (ydiff // 2, ydiff // 2)
    xwidth = (xdiff // 2, xdiff // 2)
    return ywidth, xwidth


def _fftconvolve_wrap(kernel, data):
    # wrap gaussian filter as a special case, because the gain in
    # performance is factor ~100
    if isinstance(kernel, Gaussian2DKernel):
        width = kernel.model.x_stddev.value
        norm = kernel.array.sum()
        return norm * scipy.ndimage.gaussian_filter(data, width)
    else:
        return scipy.signal.fftconvolve(
            data.astype(np.float32), kernel.array, mode="same"
        )


def scale_cube(data, kernels):
    """
    Compute scale space cube.

    Compute scale space cube by convolving the data with a set of kernels and
    stack the resulting images along the third axis.

    Parameters
    ----------
    data : `~numpy.ndarray`
        Input data.
    kernels: list of `~astropy.convolution.Kernel`
        List of convolution kernels.

    Returns
    -------
    cube : `~numpy.ndarray`
        Array of the shape (len(kernels), data.shape)
    """
    return np.dstack([_fftconvolve_wrap(kernel, data) for kernel in kernels])
