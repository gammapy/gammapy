# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Dark matter spectra
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from astropy.utils import lazyproperty
from ...utils.scripts import make_path
from ...spectrum.models import TableModel


__all__ = [
    'PrimaryFluxes'
]


class PrimaryFluxes(object):
    """Generate DM-annihilation gamma-ray spectra
    
    Based on the precompute models by `Cirelli et al.
    <http://http://www.marcocirelli.net/PPPC4DMID.html>`_. The spectra will be
    available as `~gammapy.spectrum.models.TableModel` for a chosen dark matter
    mass and annihilation channel
    """
    def __init__(self):
        self._table = None

    @lazyproperty
    def table(self):
        """Lookup table"""
        filename = '$GAMMAPY_EXTRA/datasets/dark_matter_spectra/AtProduction_gammas.dat'
        self._table = Table.read(str(make_path(filename)),
                                 format='ascii.fast_basic',
                                 guess=False,
                                 delimiter=' ')
        return self._table

    def _get_mDM(self, mDM):
        """Choose subtable for a given dark matter mass"""
        mDM_vals = self.table['mDM'].data
        mDM_ = mDM.to('GeV').value
        interp_idx = np.argmin(np.abs(mDM_vals - mDM_))
        return self.table[mDM_vals == mDM_vals[interp_idx]]

    def _validate_channel(self, channel):
        """Validate chosen channel"""
        allowed = self.table.colnames[2:]
        if channel not in allowed:
            msg = "Invalid channel {}\n"
            msg += "Available: {}\n"
            raise ValueError(msg.format(channel, allowed))


    def table_model(self, mDM, channel):
        """Generate `~gammapy.spectrum.models.TableModel`
        
        A lookup value for mDM is found using nearest neighbour interpolation.
        A list of availabe channels is given on
        http://www.marcocirelli.net/PPPC4DMID.html

        Parameters
        ----------
        mDM : `~astropy.units.Quantity`
            Dark matter mass
        channel : str
            Annihilation channel
        """
        self._validate_channel(channel) 
        subtable = self._get_mDM(mDM)

        energies = (10 ** subtable['Log[10,x]']) * mDM
        dN_dlogx = subtable[channel]
        dN_dE = dN_dlogx / (energies * np.log(10))

        return TableModel(
            energy=energies,
            values=dN_dE,
            scale_logy=False
        )
