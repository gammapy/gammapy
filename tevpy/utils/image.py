import numpy as np

__all__ = ['tophat_correlate', 'ring_correlate']

def _get_structure_indices(radius):
    """
    Get arrays of indices for a symmetric structure,
    i.e. with an odd number of pixels and 0 at the center
    """
    radius = int(radius)
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    return x, y


def binary_disk(radius):
    """
    Generate a binary disk.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(radius)
    structure = x ** 2 + y ** 2 <= radius ** 2
    return structure


def binary_ring(r_in, r_out):
    """
    Generate a binary ring.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def tophat_correlate(data, radius, mode='constant'):
    """
    Correlate with disk of given radius
    """
    from scipy.ndimage import convolve
    structure = binary_disk(radius)
    return convolve(data, structure, mode=mode)


def ring_correlate(data, r_in, r_out, mode='constant'):
    """
    Correlate with ring of given radii
    """
    from scipy.ndimage import convolve
    structure = binary_ring(r_in, r_out)
    return convolve(data, structure, mode=mode)
