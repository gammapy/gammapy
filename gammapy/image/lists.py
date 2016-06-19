from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.wcs import WCS
from ..utils.testing import requires_dependency, requires_data
from . import catalog
from ..image import SkyMap
from ..irf import EnergyDependentTablePSF
from ..cube import SkyCube
from ..datasets import FermiGalacticCenter
import numpy as np

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
        self.name = None
        self.skymaps = skymaps
        self.wcs = wcs
        self.energy = energy

    def to_cube(self):
        """Convert a list of image HDUs into one cube.
        """
        data = Quantity([skymap.data for skymap in self.skymaps],
                        self.skymaps[0].data.unit)
        return SkyCube(name=self.name, data=data, wcs=self.wcs, energy=self.energy)
