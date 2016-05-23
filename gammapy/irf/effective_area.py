#Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
from ..utils.nddata import NDDataArray, DataAxis, BinnedDataAxis

class EffectiveArea2D(NDDataArray):
    """2D effective area table

    **Disclaimer**: This is an experimental class to test the usage of the
    `~gammapy.utils.nddata.NDDataArray` base class. It is meant to replace
    `~gammapy.irf.EffectiveAreaTable2D` in the future but is currently not used
    anywhere in gammapy.

    Parameters
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    offset : `~astropy.units.Quantity`, `~gammapy.utils.nddata.DataAxis`
        Nodes of Offset axis
    data : `~astropy.units.Quantity`
        Effective area

    Examples
    --------
    Create `~gammapy.irf.EffectiveArea2D` from scratch

    >>> from gammapy.irf import EffectiveArea2D
    >>> import astropy.units as u
    >>> import numpy as np

    >>> energy = np.logspace(0,1,11) * u.TeV
    >>> offset = np.linspace(0,1,4) * u.deg
    >>> data = np.ones(shape=(10,4)) * u.cm * u.cm

    >>> eff_area = EffectiveArea2D(energy=energy, offset=offset, data= data)
    >>> print(eff_area)
    Data array summary info
    energy         : size =    11, min =  1.000 TeV, max = 10.000 TeV
    offset         : size =     4, min =  0.000 deg, max =  1.000 deg
    Data           : size =    40, min =  1.000 cm2, max =  1.000 cm2
    """

    energy = BinnedDataAxis(interpolation_mode='log')
    """Primary axis: Energy"""
    offset = DataAxis()
    """Secondary axis: Offset from pointing position"""
    axis_names = ['energy', 'offset']

    @classmethod
    def from_table(cls, t):
        """This is a read for the format specified at
        http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/effective_area/index.html#aeff-2d-format
        """
        raise NotImplementedError()

    def plot():
        """Plot image"""
        raise NotImplementedError()
