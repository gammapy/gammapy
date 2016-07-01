# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity

__all__ = ['SkyImageList']


class SkyImageList(object):
    """
    Class to represent connection between `~gammapy.image.SkyMap` and `~gammapy.cube.SkyCube`.

    Keeps list of images and has methods to convert between them and SkyCube.

    Parameters
    ----------
    name : str
        Name of the sky image list.
    skymaps : list of `~gammapy.image.SkyMap`
        Data array as list of skymaps.
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array
    """

    def __init__(self, name=None, skymaps=None, wcs=None, energy=None):
        self.name = name
        self.skymaps = skymaps
        self.wcs = wcs
        self.energy = energy

    def to_cube(self):
        """Convert a list of image HDUs into one `~gammapy.cube.SkyCube`.
        """
        from ..cube import SkyCube
        data = Quantity([skymap.data for skymap in self.skymaps],
                        self.skymaps[0].data.unit)
        return SkyCube(name=self.name, data=data, wcs=self.wcs, energy=self.energy)
