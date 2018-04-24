# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Dark matter spectra
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from astropy.utils import lazyproperty
from ...utils.scripts import make_path


__all__ = [
    'DMFluxFactory'
]


class DMFluxFactory(object):
    """Generate DM-annihilation gamma-ray spectra
    
    Based on the precompute models by `Cirelli et al.
    <http://http://www.marcocirelli.net/PPPC4DMID.html>`_. The spectra will be
    available as `~gammapy.spectrum.models.TableModel` for a chosen dark matter
    mass and annihilation channel
    """
    def __init__(self):
        self._table = None

    @property
    def table(self):
        """Lookup table"""
        if self._table is None:
            filename = '$GAMMAPY_EXTRA/datasets/dark_matter_spectra/AtProduction_gammas.dat'
            self._table = Table.read(str(make_path(filename)),
                                     format='ascii.fast_basic',
                                     guess=False,
                                     delimiter=' ')
        return self._table

    def make_model(self, mDM, channel):
        """Generate `~gammapy.spectrum.models.TableModel`
        """
        pass

