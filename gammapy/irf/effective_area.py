#Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
from ..utils.nddata import NDDataArray, DataAxis, BinnedDataAxis
from collections import OrderedDict
import numpy as np

class EffectiveArea2D(NDDataArray):
    """2D effective area table

    Axes: ``THETA``, ``ENERG``

    Parameteres
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.BinnedDataAxis`
        Energy axis (binned)
    offset : `~astropy.units.Quantity`, `~gammapy.utils.DataAxis
        Offset axis (unbinned)
    effective_area : `~astropy.units.Quantity`
        Effective area

    Examples
    --------
    Get effective area as a function of energy for a given offset and energy binning:

    .. code-block:: python

        from gammapy.irf import EffectiveAreaTable
        from gammapy.datasets import gammapy_extra
        import astropy.units as u

        filename = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
        aeff2D = EffectiveAreaTable2D.read(filename)
        energy = np.logspace(0, 1, 50) * u.TeV   
        offset = 0.3 * u.deg
        aeff2D.evaluate(THETA=offset, ENERG=energy, method='linear') 
    """
    def __init__(self, **kwargs):
        if 'energy' in kwargs and 'offset' in kwargs and 'effective_area' in kwargs:
            axes, data = self._kwargs_to_arrays(kwargs)
        elif 'axes' in kwargs and 'data' in kwargs:
            axes = kwargs['axes']
            data = kwargs['data']
        else:
            raise ValueError('Invalid initialisation: {}'.format(kwargs))

        super(EffectiveArea2D, self).__init__(axes=axes, data=data)
        self.axes[1].interpolation_mode = 'log'
        self.add_linear_interpolator(bounds_error=False)

    @staticmethod 
    def _kwargs_to_arrays(kwargs):
        """Convert kwargs to useful info for NDData.__init__()"""

    	# support array input for energy
        if isinstance(kwargs['energy'], np.ndarray):
            kwargs['energy'] = BinnedDataAxis(kwargs['energy'], name='energy')

        axes = OrderedDict(THETA=kwargs['offset'], ENERG=kwargs['energy'])
        data = kwargs['effective_area']
        return axes, data

    def check_integrity(self):
        """
        Perform basic checks that the `~gammapy.irf.EffectiveArea2D` is valid
        """
        if self.data is None:
            raise ValueError("Data not set")
        if self.dim != 2:
            raise ValueError("Too many axes: {}".format(self.dim))
        if self.axis_names != ['THETA', 'ENERG']:
            raise ValueError("Invalid axis names: {}".format(self.axis_names))
        if not isinstance(self.axes[0], DataAxis):
            raise ValueError("THETA axis must be DataAxis")
        if not isinstance(self.axes[1], BinnedDataAxis):
            raise ValueError("ENERG axis must be BinnedDataAxis")
        return True

    @classmethod
    def from_table(cls, t):
        """This is a read for the format specified at
        http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/effective_area/index.html#aeff-2d-format
        The ``EFFAREA_RECO`` column is discarded.
        """
        if 'EFFAREA_RECO' in t.colnames:
            t.remove_column('EFFAREA_RECO')
        return super(EffectiveArea2D, cls).from_table(t)
