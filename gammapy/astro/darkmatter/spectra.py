# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.utils import lazyproperty
from ...utils.scripts import make_path
from ...utils.fitting import Parameter, Parameters
from ...spectrum.models import SpectralModel, TableModel

__all__ = ["PrimaryFlux", "DMAnnihilModel"]


class PrimaryFlux:
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
        mDM_ = Quantity(mDM).to_value("GeV")
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


class DMAnnihilModel(SpectralModel):

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermal relic cross section"""

    def __init__(self, mass, channel, scale=1, jfactor=1, z=0, k=2):
        self.parameters = Parameters(
            [
                Parameter("scale", scale),
            ]
        )
        self.k = k
        self.z = z
        self.mass = mass
        self.channel = channel
        self.jfactor = jfactor
        self.table_model = PrimaryFlux(mass, channel=self.channel).table_model

    def evaluate(self, energy, scale):
        flux = (
                scale
                * self.jfactor
                * self.THERMAL_RELIC_CROSS_SECTION
                * self.table_model.evaluate(energy=energy * (1 + self.z), norm=1)
                / self.k
                / self.mass
                / self.mass
                / (4 * np.pi)
        )
        return flux