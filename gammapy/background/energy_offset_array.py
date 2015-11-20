# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = [
    'EnergyOffsetArray',
]


class EnergyOffsetArray(object):
    """
    Energy offset dependent array.

    Parameters
    ----------
    energy : `~numpy.ndarray`
        Energy array (1D)
    offset : `~numpy.ndarray`
        offset array (1D)
    data : `~numpy.ndarray`
        data array (2D)

    """

    def __init__(self, energy, offset, data):
        self.energy = energy
        self.offset = offset
        self.data = data


    @classmethod
    def from_fits_image(cis, filename):
        hdu_list=fits.open(filename)
