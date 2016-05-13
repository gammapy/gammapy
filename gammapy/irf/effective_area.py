#Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
from ..utils.nddata import NDDataArray, DataAxis, BinnedDataAxis
import numpy as np

class EffectiveArea2D(NDDataArray):
    """2D effective area table

    Axes: offset, energy

    Parameteres
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.BinnedDataAxis`
        Energy axis (binned)
    offset : `~astropy.units.Quantity`, `~gammapy.utils.DataAxis
        Offset axis (unbinned)
    data : `~astropy.units.Quantity`
        Effective area
    """
    def __init__(self, **kwargs):
    	# support array input for energy
        if 'energy' in kwargs and isinstance(kwargs['energy'], np.ndarray):
            kwargs['energy'] = BinnedDataAxis(kwargs['energy'], name='energy')

        super(EffectiveArea2D, self).__init__(**kwargs)
        self.axes[1].interpolation_mode = 'log'
        self.add_linear_interpolator(bounds_error=False)

    def check_integrity(self):
        """
        Perform basic checks that the `~gammapy.irf.EffectiveArea2D` is valid
        """
        if self.data is None:
            raise ValueError("Data not set")
        if self.dim != 2:
            raise ValueError("Too many axes: {}".format(self.dim))
	    if self.axes[0].name != 'offset' or not isinstance(self.axes[0], DataAxis):
		    raise ValueError("Invalid offset axis: {}".format(self.axes[0]))
	    if self.axes[1].name != 'energy' or not isinstance(self.axes[1], BinnedDataAxis):
	        raise ValueError("Invalid energy axis: {}".format(self.axes[1]))

    @classmethod
    def from_table(cls, t):
        """This is a read for the format specified at
        http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/effective_area/index.html#aeff-2d-format
        """
        pass
