# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""
import numpy as np
import astropy.units as u
from astropy.table import Table
from ...utils.scripts import make_path
from ...utils.fitting import Parameter
from ...spectrum.models import SpectralModel, TableModel

__all__ = ["PrimaryFlux", "DMAnnihilation"]


class PrimaryFlux:
    """DM-annihilation gamma-ray spectra.

    Based on the precomputed models by Cirelli et al. (2016). All available
    annihilation channels can be found there. The dark matter mass will be set
    to the nearest available value. The spectra will be available as
    `~gammapy.spectrum.models.TableModel` for a chosen dark matter mass and
    annihilation channel.

    References
    ----------
    * `2011JCAP...03..051 <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    * Cirelli et al (2016): http://www.marcocirelli.net/PPPC4DMID.html
    """

    channel_registry = {
        "eL": "eL",
        "eR": "eR",
        "e": "e",
        "muL": r"\[Mu]L",
        "muR": r"\[Mu]R",
        "mu": r"\[Mu]",
        "tauL": r"\[Tau]L",
        "tauR": r"\[Tau]R",
        "tau": r"\[Tau]",
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
        "gamma": r"\[Gamma]",
        "h": "h",
        "nu_e": r"\[Nu]e",
        "nu_mu": r"\[Nu]\[Mu]",
        "nu_tau": r"\[Nu]\[Tau]",
        "V->e": "V->e",
        "V->mu": r"V->\[Mu]",
        "V->tau": r"V->\[Tau]",
    }

    table_filename = "$GAMMAPY_DATA/dark_matter_spectra/AtProduction_gammas.dat"

    def __init__(self, mDM, channel):

        self.table_path = make_path(self.table_filename)
        if not self.table_path.exists():
            message = (
                "\n\nFile {} not found.\n"
                "You may download the dataset needed with the following command:\n"
                "gammapy download datasets --src dark_matter_spectra"
                "".format(self.table_filename)
            )

            raise FileNotFoundError(message)
        else:
            self.table = Table.read(
                str(self.table_path),
                format="ascii.fast_basic",
                guess=False,
                delimiter=" ",
            )

        self.mDM = mDM
        self.channel = channel

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @mDM.setter
    def mDM(self, mDM):
        mDM_vals = self.table["mDM"].data
        mDM_ = u.Quantity(mDM).to_value("GeV")
        interp_idx = np.argmin(np.abs(mDM_vals - mDM_))
        self._mDM = u.Quantity(mDM_vals[interp_idx], "GeV")

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


class DMAnnihilation(SpectralModel):
    r"""Spectral model for dark matter annihilation.

    The gamma-ray flux is computed as follows:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\langle \sigma\nu \rangle}{4\pi k m^2_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Dark matter mass
    channel : str
        Annihilation channel for `~gammapy.astro.darkmatter.PrimaryFlux`
    scale : float
        Scale parameter for model fitting
    jfactor : `~astropy.units.Quantity`
        Integrated J-Factor needed when `~gammapy.image.models.SkyPointSource` spatial model is used
    z: float
        Redshift value
    k: int
        Type of dark matter particle (k:2 Majorana, k:4 Dirac)

    Examples
    --------
    This is how to instantiate a `DMAnnihilation` model::

        from astropy import units as u
        from gammapy.astro.darkmatter import DMAnnihilation

        channel = "b"
        massDM = 5000*u.Unit("GeV")
        jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
        modelDM = DMAnnihilation(mass=massDM, channel=channel, jfactor=jfactor)

    References
    ----------
    * `2011JCAP...03..051 <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051>`_
    """

    __slots__ = ["mass", "channel", "scale", "jfactor", "z", "k", "primary_flux"]

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section"""

    def __init__(self, mass, channel, scale=1, jfactor=1, z=0, k=2):
        self.scale = Parameter("scale", scale)
        self.k = k
        self.z = z
        self.mass = mass
        self.channel = channel
        self.jfactor = jfactor
        self.primary_flux = PrimaryFlux(mass, channel=self.channel).table_model
        super().__init__([self.scale])

    def evaluate(self, energy, scale):
        """Evaluate dark matter annihilation model."""
        flux = (
            scale
            * self.jfactor
            * self.THERMAL_RELIC_CROSS_SECTION
            * self.primary_flux.evaluate(energy=energy * (1 + self.z), norm=1)
            / self.k
            / self.mass
            / self.mass
            / (4 * np.pi)
        )
        return flux
