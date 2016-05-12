#Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
from ..utils.nddata import NDDataArray, DataAxis, BinnedDataAxis


class EffectiveArea2D(NDDataArray):
    """2D effective area table
     
    Axes: energy, offset
    
    Parameteres
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.BinnedDataAxis`
	Energy axis (binned)
    offset : `~astropy.units.Quantity`, `~gammapy.utils.DataAxis
	Offset axis (unbinned)
    effective_area : `~astropy.units.Quantity`
	Effective area
    """
    def __init__(self, energy, offset, effective_area):

        super(EffectiveArea2D, self).__init__()
	energy = BinnedDataAxis(energy, name='energy')
	energy.interpolation_mode = 'log'
	self.add_axis(energy)
	offset = DataAxis(offset, name='offset')
	self.add_axis(offset)
	self.data = effective_area
	self.add_linear_interpolator(bounds_error=False)
