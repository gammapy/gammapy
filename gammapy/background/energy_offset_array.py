# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = [
    'EnergyOffsetArray',
]


class EnergyOffsetArray(object):
    """
    Energy offset dependent array.


    """

    def __init__(self, energy, offset, data):
        self.energy = energy
        self.offset = offset
        self.data = data

    @classmethod
    def from_fits_image(cis, filename):
        hdu_list=fits.open(filename)
        energy=