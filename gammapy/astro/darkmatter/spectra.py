# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from astropy.utils import lazyproperty
from ...utils.scripts import make_path
from ...spectrum.models import TableModel

__all__ = ["PrimaryFlux"]


class PrimaryFlux(object):
    """DM-annihilation gamma-ray spectra.

    Based on the precomputed models by `Cirelli et al.
    <http://www.marcocirelli.net/PPPC4DMID.html>`_. All available
    annihilation channels can be found there. The dark matter mass will be set
    to the nearest available value. The spectra will be available as
    `~gammapy.spectrum.models.TableModel` for a chosen dark matter mass and
    annihilation channel.

    References
    ----------
    * `2011JCAP...03..051 <http://adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    channel_registry = {
        "eL": "eL",
        "eR": "eR",
        "e": "e",
        "muL": "\\[Mu]L",
        "muR": "\\[Mu]R",
        "mu": "\\[Mu]",
        "tauL": "\\[Tau]L",
        "tauR": "\\[Tau]R",
        "tau": "\\[Tau]",
        "q": "q",
        "c": "c",
        "b": "b",
        "t": "t",
        "WL": "WL",
        "WT": "WT",
        "W": "W",
        "ZL": "ZL",
        "ZT": "ZT",
        "Z": "Z",
        "g": "g",
        "gamma": "\\[Gamma]",
        "h": "h",
        "nu_e": "\\[Nu]e",
        "nu_mu": "\\[Nu]\\[Mu]",
        "nu_tau": "\\[Nu]\\[Tau]",
        "V->e": "V->e",
        "V->mu": "V->\\[Mu]",
        "V->tau": "V->\\[Tau]",
    }

    def __init__(self, mDM, channel):
        self.mDM = mDM
        self.channel = channel

    @lazyproperty
    def table(self):
        """Lookup table (`~astropy.table.Table`)."""
        filename = "$GAMMAPY_DATA/dark_matter_spectra/AtProduction_gammas.dat"
        return Table.read(
            str(make_path(filename)),
            format="ascii.fast_basic",
            guess=False,
            delimiter=" ",
        )

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @mDM.setter
    def mDM(self, mDM):
        mDM_vals = self.table["mDM"].data
        mDM_ = Quantity(mDM).to("GeV").value
        interp_idx = np.argmin(np.abs(mDM_vals - mDM_))
        self._mDM = Quantity(mDM_vals[interp_idx], "GeV")

    @property
    def allowed_channels(self):
        """List of allowed annihilation channels."""
        return list(self.channel_registry.keys())

    @property
    def channel(self):
        """Annihilation channel (str)."""
        return self._channel

    @channel.setter
    def channel(self, channel):
        if channel not in self.allowed_channels:
            msg = "Invalid channel {}\n"
            msg += "Available: {}\n"
            raise ValueError(msg.format(channel, self.allowed_channels))
        else:
            self._channel = channel

    @property
    def table_model(self):
        """Spectrum as `~gammapy.spectrum.models.TableModel`."""
        subtable = self.table[self.table["mDM"] == self.mDM.value]
        energies = (10 ** subtable["Log[10,x]"]) * self.mDM
        channel_name = self.channel_registry[self.channel]
        dN_dlogx = subtable[channel_name]
        dN_dE = dN_dlogx / (energies * np.log(10))

        return TableModel(energy=energies, values=dN_dE, values_scale="lin")
