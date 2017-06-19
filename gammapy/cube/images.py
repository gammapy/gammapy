# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from .core import SkyCube
from ..spectrum import LogEnergyAxis

__all__ = [
    'SkyCubeImages',
]


class SkyCubeImages(object):
    """
    Class to represent connection between `~gammapy.image.SkyImage` and `~gammapy.cube.SkyCube`.

    Keeps list of images and has methods to convert between them and SkyCube.

    Parameters
    ----------
    name : str
        Name of the sky image list.
    images : list of `~gammapy.image.SkyImage`
        Data array as list of images.
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array
    meta : dict
        Dictionary to store meta data.
    """

    def __init__(self, name=None, images=None, wcs=None, energy=None, meta=None):
        self.name = name
        self.images = images
        self.wcs = wcs
        self.energy = energy
        self.meta = meta

    def to_cube(self):
        """Convert this list of images to a `~gammapy.cube.SkyCube`.
        """
        if hasattr(self.images[0].data, 'unit'):
            unit = self.images[0].data.unit
        else:
            unit = None
        data = Quantity([image.data for image in self.images], unit)
        energy_axis = LogEnergyAxis(self.energy)
        return SkyCube(name=self.name, data=data, wcs=self.wcs,
                       energy_axis=energy_axis, meta=self.meta)
