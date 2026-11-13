# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""

import copy
import warnings
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    PRIOR_REGISTRY,
    SpectralModel,
    TemplateNDSpectralModel,
    GaussianSpectralModel,
)
from gammapy.utils.deprecation import (
    GammapyDeprecationWarning,
    deprecated,
    deprecated_renamed_argument,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_map_columns

__all__ = [
    "ContinuumPrimaryFlux",
    "DarkMatterSpectralModel",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "PrimaryFlux",
    "MonochromaticPrimaryFlux",
    "VIBPrimaryFlux",
    "BoxPrimaryFlux",
]
log = logging.getLogger(__name__)


class MonochromaticPrimaryFlux(SpectralModel):
    r"""Monochromatic gamma-ray line from dark matter annihilation.

    Spectral shape of :math:`\chi\chi \to \gamma\gamma` (two photons per
    annihilation) and :math:`\chi\chi \to \gamma X` (one photon plus a
    massive counterpart :math:`X`, e.g. :math:`\chi\chi \to \gamma Z` or
    :math:`\chi\chi \to \gamma h`). The line is centered at the energy

    .. math::
        E_0 = m_{\mathrm{DM}} \left( 1 - \frac{m_X^2}{4 m_{\mathrm{DM}}^2} \right),

    which equals :math:`m_{\mathrm{DM}}` for :math:`\chi\chi \to \gamma\gamma`
    (:math:`m_X = 0`), and is approximated by a narrow Gaussian:

    .. math::
        \frac{\mathrm{d}N}{\mathrm{d}E} = \frac{N_\gamma}{\sqrt{2\pi}\,\sigma}
        \exp\left( -\frac{(E - E_0)^2}{2 \sigma^2} \right),
        \qquad \sigma = \sigma_{\mathrm{rel}} \, E_0,

    where :math:`N_\gamma` is the number of photons per annihilation: 2 if
    :math:`m_X = 0` and 1 otherwise.

    This class gives the photon yield per annihilation, in GeV⁻¹, not a
    flux: it is meant to be used as ``primary_flux`` of
    `DarkMatterSpectralModel`, which provides the normalization and the
    astrophysical factor.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be strictly positive.
    counterpart_mass : `~astropy.units.Quantity`, optional
        Mass :math:`m_X` of the particle produced together with the photon.
        Default is 0 GeV, which corresponds to :math:`\chi\chi \to \gamma\gamma`.
        Reference values: Z boson 91.1876 GeV, Higgs boson 125.20 GeV.
    sigma_rel : float, optional
        Relative width :math:`\sigma_{\mathrm{rel}}` of the Gaussian
        approximation to the line. Default is 0.01 (1%).

    Notes
    -----
    In a dataset, the line is folded with the energy dispersion of the
    instrument, which is usually much wider than ``sigma_rel``. The true
    energy axis of the dataset should be binned well below the energy
    resolution, otherwise the predicted line position moves in steps when
    scanning ``mDM``.

    Examples
    --------
    Two-photon line at 1 TeV, used as the primary flux of a dark matter
    spectral model::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import (
        ...     DarkMatterSpectralModel,
        ...     MonochromaticPrimaryFlux,
        ... )
        >>> flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV)
        >>> model = DarkMatterSpectralModel(
        ...     mDM=1 * u.TeV,
        ...     channel=None,
        ...     primary_flux=flux,
        ...     factor=3.41e19 * u.Unit("GeV2 cm-5"),
        ... )

    One-photon line with a Z boson as counterpart::

        >>> flux = MonochromaticPrimaryFlux(
        ...     mDM=1 * u.TeV, counterpart_mass=91.1876 * u.GeV
        ... )

    References
    ----------
    `Bergström & Snellman (1988), "Observable monochromatic photons from cosmic
    photino annihilation"
    DOI:10.1103/PhysRevD.37.3737 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.37.3737>`_
    """

    tag = ["MonochromaticPrimaryFlux", "dm-mono"]

    def __init__(self, mDM, counterpart_mass=0 * u.GeV, sigma_rel=0.01):
        self.mDM = mDM
        self.counterpart_mass = counterpart_mass
        self.sigma_rel = sigma_rel
        self.get_line_energy()  # raises if kinematically forbidden
        super().__init__()

    def get_line_energy(self):
        r"""Compute the gamma-ray line energy.

        .. math::
            E_0 = m_{\mathrm{DM}} \left( 1 - \frac{m_X^2}{4 m_{\mathrm{DM}}^2} \right)

        Returns
        -------
        E_0 : `~astropy.units.Quantity`
            Energy of the monochromatic gamma-ray line.

        Raises
        ------
        ValueError
            If ``counterpart_mass >= 2 * mDM`` (kinematically forbidden).
        """
        if self.counterpart_mass >= 2 * self.mDM:
            raise ValueError(
                f"Kinematically forbidden: counterpart_mass ({self.counterpart_mass}) "
                f"must be smaller than 2 * mDM ({2 * self.mDM})."
            )
        return self.mDM * (1 - self.counterpart_mass**2 / (4 * self.mDM**2))

    def evaluate(self, energy):
        """Evaluate the monochromatic primary flux spectrum dN/dE.

        Gaussian of relative width ``sigma_rel`` centered on the line energy
        (see `get_line_energy`), normalized to `n_gamma_photons`.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the spectrum.

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy, in units of GeV⁻¹.
        """
        E_0 = self.get_line_energy()
        return GaussianSpectralModel.evaluate(
            energy,
            amplitude=self.n_gamma_photons,
            mean=E_0,
            sigma=self.sigma_rel * E_0,
        ).to("GeV-1")

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @mDM.setter
    @u.quantity_input(value="energy")
    def mDM(self, value):
        if value <= 0 * u.GeV:
            raise ValueError("mDM must be strictly positive.")
        self._mDM = value

    @property
    def n_gamma_photons(self):
        """Number of photons per annihilation: 2 if ``counterpart_mass`` is 0, else 1."""
        return 2 if self.counterpart_mass == 0 * u.GeV else 1

    @property
    def counterpart_mass(self):
        """Mass of the counterpart particle (0 for a two-photon line)."""
        return self._counterpart_mass

    @counterpart_mass.setter
    @u.quantity_input(value="energy")
    def counterpart_mass(self, value):
        if value < 0 * u.GeV:
            raise ValueError("counterpart_mass cannot be negative.")
        self._counterpart_mass = value

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Parameters
        ----------
        full_output : bool, optional
            Unused; present for interface compatibility. Default is False.

        Returns
        -------
        data : dict
            Dictionary representation containing the model type, ``mDM``,
            ``counterpart_mass`` and ``sigma_rel``, suitable for
            round-tripping via `from_dict`.
        """
        return {
            "type": "MonochromaticPrimaryFlux",
            "mDM": self.mDM.to_string(),
            "counterpart_mass": self.counterpart_mass.to_string(),
            "sigma_rel": self.sigma_rel,
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a `MonochromaticPrimaryFlux` from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by `to_dict`, containing ``mDM``,
            ``counterpart_mass`` and ``sigma_rel``.

        Returns
        -------
        flux : `MonochromaticPrimaryFlux`
            New instance reconstructed from ``data``.
        """
        return cls(
            mDM=u.Quantity(data["mDM"]),
            counterpart_mass=u.Quantity(data["counterpart_mass"]),
            sigma_rel=data["sigma_rel"],
        )


class VIBPrimaryFlux(SpectralModel):
    r"""Virtual Internal Bremsstrahlung (VIB) spectral shape.

    Gamma-ray spectrum from internal bremsstrahlung in dark matter
    annihilation, :math:`\chi\chi \to f \bar{f} \gamma`, valid for Majorana
    dark matter annihilating to Standard Model fermion pairs in the limit of
    large dark matter mass and nearly degenerate mediating sfermions. With
    :math:`x = E / m_{\mathrm{DM}}`, the spectrum is

    .. math::
        \frac{\mathrm{d}N}{\mathrm{d}x} = A_{\mathrm{VIB}} \,
        \frac{x \left( x^3 - 4x^2 + 6x - 4 \right) - 4 (x - 1)^2 \ln(1 - x)}
        {(x - 2)^3}
        \qquad \text{for } 0 < x < 1,

    and zero otherwise, with
    :math:`\mathrm{d}N/\mathrm{d}E = (\mathrm{d}N/\mathrm{d}x) / m_{\mathrm{DM}}`.
    The constant :math:`A_{\mathrm{VIB}} = 6 / (21 - 2\pi^2)` normalizes the
    spectrum to one photon per annihilation,
    :math:`\int_0^1 (\mathrm{d}N/\mathrm{d}x) \, \mathrm{d}x = 1`.

    This class gives the photon yield per annihilation, in GeV⁻¹, not a
    flux: it is meant to be used as ``primary_flux`` of
    `DarkMatterSpectralModel`, which provides the normalization and the
    astrophysical factor. Because of the normalization above, the
    cross-section then refers to the process
    :math:`\chi\chi \to f \bar{f} \gamma`.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be strictly positive.

    Examples
    --------
    ::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import VIBPrimaryFlux
        >>> flux = VIBPrimaryFlux(mDM=1 * u.TeV)

    References
    ----------
    `Bringmann et al. (2008), "New Gamma-Ray Contributions to Supersymmetric
    Dark Matter Annihilation"
    DOI:10.1088/1126-6708/2008/01/049 <https://arxiv.org/abs/0710.3169>`_
    """

    tag = ["VIBPrimaryFlux", "dm-vib"]

    A_VIB = 6.0 / (21.0 - 2.0 * np.pi**2)

    def __init__(self, mDM):
        self.mDM = mDM
        super().__init__()

    def evaluate(self, energy):
        """Evaluate the VIB primary flux spectrum dN/dE.

        Zero outside ``0 < energy / mDM < 1``; small negative numerical
        artifacts are clipped to zero.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the spectrum.

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy, in units of GeV⁻¹.
        """
        x = energy / self.mDM

        # Mask: only defined for 0 < x < 1
        valid = (x > 0) & (x < 1)
        dN_dx = np.zeros_like(x)

        xv = x[valid]
        numerator = xv * (xv**3 - 4 * xv**2 + 6 * xv - 4) - 4 * (xv - 1) ** 2 * np.log(
            1 - xv
        )
        denominator = (xv - 2) ** 3

        result = self.A_VIB * numerator / denominator
        dN_dx[valid] = np.clip(result, 0, None)

        return (dN_dx / self.mDM).to("GeV-1")

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @mDM.setter
    @u.quantity_input(value="energy")
    def mDM(self, value):
        if value <= 0 * u.GeV:
            raise ValueError("mDM must be strictly positive.")
        self._mDM = value

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Parameters
        ----------
        full_output : bool, optional
            Unused; present for interface compatibility. Default is False.

        Returns
        -------
        data : dict
            Dictionary representation containing the model type and
            ``mDM``, suitable for round-tripping via `from_dict`.
        """
        return {
            "type": "VIBPrimaryFlux",
            "mDM": self.mDM.to_string(),
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a `VIBPrimaryFlux` from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by `to_dict`, containing ``mDM``.

        Returns
        -------
        flux : `VIBPrimaryFlux`
            New instance reconstructed from ``data``.
        """
        return cls(mDM=u.Quantity(data["mDM"]))


class BoxPrimaryFlux(SpectralModel):
    r"""Box-shaped gamma-ray spectrum from dark matter annihilation.

    Spectrum from dark matter annihilation into two neutral intermediate
    states, :math:`\chi\chi \to \phi_1 \phi_2`, each decaying into two
    photons, :math:`\phi_i \to \gamma\gamma`. The photons are monochromatic
    in the rest frame of each :math:`\phi_i`; boosting to the dark matter
    rest frame gives a flat ("box-shaped") spectrum. The energies of the
    intermediate states and the common width of the boxes are

    .. math::
        E_{\phi_{1,2}} = m_{\mathrm{DM}} \pm
        \frac{m_{\phi_1}^2 - m_{\phi_2}^2}{4 m_{\mathrm{DM}}},
        \qquad
        \Delta E = \sqrt{E_{\phi_1}^2 - m_{\phi_1}^2},

    and each :math:`\phi_i` contributes two photons uniformly distributed
    between :math:`E_i^\pm = (E_{\phi_i} \pm \Delta E) / 2`:

    .. math::
        \frac{\mathrm{d}N}{\mathrm{d}E} = \sum_{i=1}^{2} \frac{2}{\Delta E} \,
        \Theta\left( E - E_i^- \right) \, \Theta\left( E_i^+ - E \right).

    The spectrum therefore contains four photons per annihilation. For
    identical intermediate states the two boxes coincide, giving
    :math:`\mathrm{d}N/\mathrm{d}E = 4 / \Delta E`; overlapping boxes simply
    add up.

    This class gives the photon yield per annihilation, in GeV⁻¹, not a
    flux: it is meant to be used as ``primary_flux`` of
    `DarkMatterSpectralModel`, which provides the normalization and the
    astrophysical factor.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be strictly positive.
    mPhi : `~astropy.units.Quantity`
        Mass of the intermediate states. A single value (or a one-element
        array) means two identical states, :math:`\chi\chi \to \phi\phi`;
        two values give :math:`m_{\phi_1}` and :math:`m_{\phi_2}`.

    Examples
    --------
    ::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import BoxPrimaryFlux
        >>> flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=100 * u.GeV)
        >>> flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[100, 50] * u.GeV)

    References
    ----------
    `Ibarra et al. (2012), "Dark matter constraints from box-shaped gamma-ray
    features"
    DOI:10.1088/1475-7516/2012/07/043 <https://arxiv.org/abs/1205.0007>`_
    """

    tag = ["BoxPrimaryFlux", "dm-box"]

    def __init__(self, mDM, mPhi):
        self.mDM = mDM
        self.mPhi = mPhi
        self.delta_E  # raises if kinematically forbidden
        super().__init__()

    def evaluate(self, energy):
        """Evaluate the box-shaped primary flux spectrum dN/dE.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (scalar or array of any shape) at which to
            evaluate the spectrum.

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy, in units of GeV⁻¹,
            with the same shape as ``energy``.
        """
        delta_E = self.delta_E
        dN_dE = np.zeros(energy.shape) / u.GeV

        for E_phi in self.energies_phi:
            E_min = (E_phi - delta_E) / 2
            E_max = (E_phi + delta_E) / 2
            in_box = (energy >= E_min) & (energy <= E_max)
            dN_dE = dN_dE + np.where(in_box, 2 / delta_E, 0 / delta_E)

        return dN_dE.to("GeV-1")

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @mDM.setter
    @u.quantity_input(value="energy")
    def mDM(self, value):
        if value <= 0 * u.GeV:
            raise ValueError("mDM must be strictly positive.")
        self._mDM = value

    @property
    def mPhi(self):
        """Masses of the two intermediate states, as a two-element Quantity."""
        return self._mPhi

    @mPhi.setter
    @u.quantity_input(value="energy")
    def mPhi(self, value):
        value = np.atleast_1d(value)
        if value.size not in (1, 2):
            raise ValueError(f"mPhi must contain 1 or 2 values, got {value.size}.")
        if np.any(value <= 0 * u.GeV):
            raise ValueError("mPhi must be strictly positive.")
        self._mPhi = u.Quantity([value[0], value[-1]])

    @property
    def mPhi1(self):
        """Mass of the first intermediate state."""
        return self.mPhi[0]

    @property
    def mPhi2(self):
        """Mass of the second intermediate state."""
        return self.mPhi[1]

    @property
    def energies_phi(self):
        """Energies of the two intermediate states in the dark matter rest frame."""
        E_phi1 = self.mDM + (self.mPhi1**2 - self.mPhi2**2) / (4 * self.mDM)
        E_phi2 = self.mDM + (self.mPhi2**2 - self.mPhi1**2) / (4 * self.mDM)
        return E_phi1, E_phi2

    @property
    def delta_E(self):
        """Width of the boxes (momentum of the intermediate states)."""
        if self.mPhi1 + self.mPhi2 >= 2 * self.mDM:
            raise ValueError(
                f"Kinematically forbidden: mPhi1 + mPhi2 ({self.mPhi1 + self.mPhi2}) "
                f"must be smaller than 2 * mDM ({2 * self.mDM})."
            )
        E_phi1, _ = self.energies_phi
        return np.sqrt(E_phi1**2 - self.mPhi1**2)

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Parameters
        ----------
        full_output : bool, optional
            Unused; present for interface compatibility. Default is False.

        Returns
        -------
        data : dict
            Dictionary representation containing the model type, ``mDM``,
            and ``mPhi`` (as a two-element list ``[mPhi1, mPhi2]``),
            suitable for round-tripping via `from_dict`.
        """
        return {
            "type": "BoxPrimaryFlux",
            "mDM": self.mDM.to_string(),
            "mPhi": [self.mPhi1.to_string(), self.mPhi2.to_string()],
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a `BoxPrimaryFlux` from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by `to_dict`, containing ``mDM`` and
            ``mPhi``.

        Returns
        -------
        flux : `BoxPrimaryFlux`
            New instance reconstructed from ``data``.
        """
        return cls(
            mDM=u.Quantity(data["mDM"]),
            mPhi=u.Quantity([u.Quantity(m) for m in data["mPhi"]]),
        )


class ContinuumPrimaryFlux(TemplateNDSpectralModel):
    """Continuum gamma-ray spectrum from dark matter annihilation.

    Based on the precomputed tables of PPPC4 DM ID and CosmiXs. All
    available annihilation channels can be found in those tables. For a
    requested dark matter mass and channel, this class builds a
    `~gammapy.modeling.models.TemplateNDSpectralModel` over a 2D grid of
    (``Log[10,x]``, ``mDM``), enabling interpolation in dark matter mass
    while keeping ``mDM`` itself frozen as a model parameter.

    .. note::

        Since Gammapy v2.2, the default source has changed from "pppc4" to "cosmixs".

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass as rest mass energy. Must lie within the
        mass range tabulated by the chosen ``source``.
    channel : str
        Annihilation channel, e.g. ``"b"`` for bb̄. See
        `allowed_channels` for the full list of supported channel labels.
        Availability of a given channel may depend on ``source``.
    source : str, optional
        Data source for the spectral tables. Options are:

        * ``"cosmixs"`` (default): Arina et al. (2024) / CosmiXs tables.
        * ``"pppc4"``: Cirelli et al. (2011, 2016) PPPC4DMID
          tables.
        * A path to a custom file readable by `astropy.table.Table.read`
          (extensions ``.dat``, ``.txt``, ``.csv``, or ``.ecsv``).

        If a custom file path is provided, it must contain ``"mDM"`` and
        ``"Log[10,x]"`` columns (after applying ``mapping_dict`` if given),
        plus columns named after the requested annihilation channel(s)
        using the internal channel registry naming convention.

    mapping_dict : dict, optional
        Mapping dictionary used to rename the columns of a custom source
        file to the expected internal column names. Only used when
        ``source`` is a custom file path. Format is
        ``{actual_column_name: expected_column_name}``, and must cover the
        mandatory columns ``"mDM"`` and ``"Log[10,x]"``.


    Notes
    -----
    Internally, the spectral table is read and reshaped into a 2D
    `~gammapy.maps.RegionGeom` over (``Log[10,x]``, ``mDM``) axes, wrapped
    in a `~gammapy.maps.Map`, and passed to the
    `~gammapy.modeling.models.TemplateNDSpectralModel` constructor with
    linear interpolation and zero-valued extrapolation. The ``mass``
    parameter inherited from `TemplateNDSpectralModel` is frozen after
    initialization, since ``mDM`` is treated as a fixed configuration value
    rather than a fit parameter.

    References
    ----------
    .. [1] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [2] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for
       indirect dark matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [3] `Di Mauro et al. (2025), "Nailing down the theoretical
       uncertainties of Dbar spectrum produced from dark matter"
       <https://arxiv.org/abs/2411.04815>`_ and available on
       `GitHub <https://github.com/ajueid/CosmiXs>`_
    """

    # noqa: E501

    _unavailable_channels = {
        "pppc4": {
            "aZ": "choose another channel or use CosmiXs ('cosmixs') as source",
            "HZ": "choose another channel or use CosmiXs ('cosmixs') as source",
            "d": (
                "choose the equivalent channel 'q' or use CosmiXs ('cosmixs') as source"
            ),
            "u": (
                "choose the equivalent channel 'q' or use CosmiXs ('cosmixs') as source"
            ),
            "s": (
                "choose the equivalent channel 'q' or use CosmiXs ('cosmixs') as source"
            ),
        },
        "cosmixs": {
            "V->e": "choose another channel or use PPPC4 ('pppc4') as source",
            "V->mu": "choose another channel or use PPPC4 ('pppc4') as source",
            "V->tau": "choose another channel or use PPPC4 ('pppc4') as source",
            "q": (
                "choose an equivalent channel such as 'd', 'u', or 's' "
                "or use PPPC4 ('pppc4') as source"
            ),
        },
    }

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
        "aZ": "aZ",
        "HZ": "HZ",
        "d": "d",
        "u": "u",
        "s": "s",
    }

    mandatory_keys = ["mDM", "Log[10,x]"]

    mapping_dict_PPPC4_to_CosmiXs = {
        "DM": "mDM",
        "Log10[x]": "Log[10,x]",
        "dNdLog10x[eL]": "eL",
        "dNdLog10x[eR]": "eR",
        "dNdLog10x[e]": "e",
        "dNdLog10x[muL]": "\\[Mu]L",
        "dNdLog10x[muR]": "\\[Mu]R",
        "dNdLog10x[mu]": "\\[Mu]",
        "dNdLog10x[tauL]": "\\[Tau]L",
        "dNdLog10x[tauR]": "\\[Tau]R",
        "dNdLog10x[tau]": "\\[Tau]",
        "dNdLog10x[nue]": "\\[Nu]e",
        "dNdLog10x[numu]": "\\[Nu]\\[Mu]",
        "dNdLog10x[nutau]": "\\[Nu]\\[Tau]",
        "dNdLog10x[u]": "u",  # no PPPC4 equivalent; maps to q
        "dNdLog10x[d]": "d",  # no PPPC4 equivalent; maps to q
        "dNdLog10x[s]": "s",  # no PPPC4 equivalent; maps to q
        "dNdLog10x[c]": "c",
        "dNdLog10x[b]": "b",
        "dNdLog10x[t]": "t",
        "dNdLog10x[a]": "\\[Gamma]",
        "dNdLog10x[g]": "g",
        "dNdLog10x[W]": "W",
        "dNdLog10x[WL]": "WL",
        "dNdLog10x[WT]": "WT",
        "dNdLog10x[Z]": "Z",
        "dNdLog10x[ZL]": "ZL",
        "dNdLog10x[ZT]": "ZT",
        "dNdLog10x[H]": "h",
        "dNdLog10x[aZ]": None,  # does not exist in PPPC4
        "dNdLog10x[HZ]": None,  # does not exist in PPPC4
    }

    tag = ["ContinuumPrimaryFlux", "dm-pf"]

    def __init__(self, mDM, channel, source=None, mapping_dict=None):
        self.source = source

        if self._source_type == "custom_file":
            self.mapping_dict = mapping_dict

            if isinstance(self.source, Table):
                self.table = self.source
            else:
                table_filename = self.source
                self.table_path = make_path(table_filename)

                if self.table_path is None or not self.table_path.exists():
                    raise FileNotFoundError(  # pragma: no cover
                        f"\n\nFile not found: {table_filename}\n"
                        "You may download the dataset needed with the \
                            following command:\n"
                        "gammapy download datasets --src dark_matter_spectra"
                    )
                self.table = Table.read(self.table_path)

        else:
            base_data_path = "$GAMMAPY_DATA/dark_matter_spectra"
            if self.source == "pppc4":
                table_filename = f"{base_data_path}/PPPC4DMID/AtProduction_gammas.dat"
            elif self.source == "cosmixs":
                table_filename = f"{base_data_path}/cosmixs/AtProduction-Gamma.dat"

            self.table_path = make_path(table_filename)

            if self.table_path is None or not self.table_path.exists():
                raise FileNotFoundError(
                    f"\n\nFile not found: {table_filename}\n"
                    "You may download the dataset needed with the following command:\n"
                    "gammapy download datasets --src dark_matter_spectra"
                )

            ascii_format = (
                "ascii.commented_header"
                if self.source == "cosmixs"
                else "ascii.fast_basic"
            )
            self.table = Table.read(
                self.table_path,
                format=ascii_format,
                guess=False,
                delimiter=" ",
            )

        if self._source_type == "custom_file" and self.mapping_dict:
            self.table = table_map_columns(self.table, self.mapping_dict)
        if self.source == "cosmixs":
            self.table = table_map_columns(
                self.table, self.mapping_dict_PPPC4_to_CosmiXs
            )

        self.channel = channel
        masses = np.unique(self.table["mDM"])
        log10x = np.unique(self.table["Log[10,x]"])

        mass_axis = MapAxis.from_nodes(masses, name="mass", interp="log", unit="GeV")
        log10x_axis = MapAxis.from_nodes(log10x, name="energy_true")

        channel_name = self.channel_registry[self.channel]
        geom = RegionGeom(region=None, axes=[log10x_axis, mass_axis])
        region_map = Map.from_geom(
            geom=geom, data=self.table[channel_name].reshape(geom.data_shape)
        )

        interp_kwargs = {"extrapolate": True, "fill_value": 0, "values_scale": "lin"}
        super().__init__(region_map, interp_kwargs=interp_kwargs)
        self.mDM = mDM
        self.mass.frozen = True

    def _resolve_table_path(self):
        """Resolve and validate the path to the spectra data file."""
        base = "$GAMMAPY_DATA/dark_matter_spectra"
        if self._source_type == "custom_file":
            table_filename = self.source
        elif self.source == "pppc4":
            table_filename = f"{base}/PPPC4DMID/AtProduction_gammas.dat"
        elif self.source == "cosmixs":
            table_filename = f"{base}/cosmixs/AtProduction-Gamma.dat"
        else:
            raise ValueError(f"Unknown source: '{self.source}'")

        path = make_path(table_filename)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"\n\nFile not found: {table_filename}\n"
                "You may download the required dataset with:\n"
                "    gammapy download datasets --src dark_matter_spectra"
            )
        return path

    def evaluate(self, energy, *args):
        """Evaluate the continuum primary flux spectrum dN/dE.

        Converts the requested ``energy`` to ``log10(energy / mDM)``,
        evaluates the underlying
        `~gammapy.modeling.models.TemplateNDSpectralModel` (interpolated
        over ``Log[10,x]`` and ``mDM``) to obtain ``dN/dlog10(x)``, and
        converts this to ``dN/dE`` via the Jacobian
        ``dN/dE = dN/dlog10(x) / (E * ln(10))``.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the spectrum.
        *args
            Additional template axis coordinates required by the parent
            `~gammapy.modeling.models.TemplateNDSpectralModel.evaluate`
            (the current ``mDM`` value is appended automatically).

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy.
        """
        args = list(args)
        args.append(self.mDM)

        log10x = np.log10(energy / self.mDM)

        dN_dlogx = super().evaluate(log10x, *args)
        dN_dE = dN_dlogx / (energy * np.log(10))
        return dN_dE

    @property
    def mDM(self):
        """Dark matter mass."""
        return u.Quantity(self.mass.value, self.mass.unit)

    @mDM.setter
    def mDM(self, mDM):
        unit = self.mass.unit
        _mDM = u.Quantity(mDM).to(unit)
        _mDM_val = _mDM.to_value(unit)

        min_mass = u.Quantity(self.mass.min, unit)
        max_mass = u.Quantity(self.mass.max, unit)

        if _mDM_val < self.mass.min or _mDM_val > self.mass.max:
            raise ValueError(
                f"The mass {_mDM} is out of the bounds of the model. Please choose a \
                mass between {min_mass} < `mDM` < {max_mass}"
            )

        self.mass.value = _mDM_val

    @property
    def allowed_channels(self):
        """List of allowed annihilation channels."""
        return list(self.channel_registry.keys())

    @property
    def source(self):
        """Data source for the spectra."""
        return self._source

    @source.setter
    def source(self, source):
        if source is None:
            self._source = "cosmixs"

            log.info(
                "\nSince no spectra source has been chosen, CosmiXs will be \
                    used by default.\n",
            )
            log.warning(
                "\nSince Gammapy v2.2, the default source has changed from 'pppc4' to 'cosmixs'.\n",
            )
        elif isinstance(source, Table):
            self._source = source
        elif isinstance(source, str):
            if source.lower() in ("pppc4", "cosmixs"):
                self._source = source.lower()
            else:
                path = Path(make_path(source))
                if path.exists() and path.is_file():
                    if path.stat().st_size == 0:
                        raise KeyError("Source file is empty.")
                    self._source = source

                else:
                    raise ValueError(
                        f"Invalid source: {source}\nAvailable options: 'pppc4', \
                        'cosmixs' or a valid file path.\n"
                    )
        else:
            raise TypeError(
                f"source must be None, a string ('pppc4', 'cosmixs', or a file "
                f"path), or an astropy.table.Table instance, got {type(source)}"
            )

    @property
    def _source_type(self):
        """Return source type (predefined or custom)."""
        return "predefined" if self.source in ("pppc4", "cosmixs") else "custom_file"

    @property
    def channel(self):
        """Annihilation channel as a string."""
        return self._channel

    @channel.setter
    def channel(self, channel):
        if channel not in self.allowed_channels:
            raise ValueError(
                f"Invalid channel: '{channel}'\n"
                f"Available channels: {self.allowed_channels}"
            )
        else:
            if self._source_type == "custom_file":
                channel_translation = self.channel_registry[channel]

                if self.mapping_dict is not None:
                    if channel_translation not in self.mapping_dict.values():
                        raise ValueError(
                            f"The channel {channel_translation} is not available \
                            in the provided mapping dictionary. Please choose another \
                            channel or check the mapping_dict provided.\n"
                        )

                if channel_translation not in self.table.colnames:
                    raise ValueError(
                        f"\n\nThe channel {channel_translation} is not available \
                        in the provided source file. Please choose another channel \
                        or check the column names in the file.\n"
                    )
            elif self.source == "pppc4":
                if channel in ("aZ", "HZ"):
                    raise ValueError(
                        f"\n\nThe channel {channel} is not available in PPPC4, please \
                           choose another channel or use CosmiXs (cosmixs) as source\n"
                    )
                elif channel in ("d", "u", "s"):
                    raise ValueError(
                        f"\n\nThe channel {channel} is not available in PPPC4, \
                         please choose the equivalent channel q \
                         or CosmiXs (cosmixs) as source\n"
                    )

            elif self.source == "cosmixs":
                if channel in ("V->e", "V->mu", "V->tau"):
                    raise ValueError(
                        f"\n\nThe channel {channel} is not available in CosmiXs, \
                        please choose another channel or use PPPC4 as source\n"
                    )
                elif channel == "q":
                    raise ValueError(
                        "\n\nThe channel q is not available in cosmixs, please \
                        choose an equivalent channel such as d, u or s or \
                        use PPPC4 as source\n"
                    )

            self._channel = channel

    @property
    def mapping_dict(self):
        """Mapping dictionary for the spectra file."""
        return self._mapping_dict

    @mapping_dict.setter
    def mapping_dict(self, mapping_dict):
        if mapping_dict is not None:
            if not isinstance(mapping_dict, dict):
                raise TypeError("mapping_dict must be a dictionary.")

            for key in self.mandatory_keys:
                if key not in mapping_dict.values():
                    raise KeyError(
                        f"Mandatory column {key} not found in file. \
                        Please check the mapping_dict provided or the column \
                        names in the file.\n"
                    )
        self._mapping_dict = mapping_dict

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Parameters
        ----------
        full_output : bool, optional
            Unused; present for interface compatibility. Default is False.

        Returns
        -------
        data : dict
            Dictionary representation containing the model type, ``mDM``,
            ``channel``, and ``source``, suitable for round-tripping via
            `from_dict`.
        """
        return {
            "type": "ContinuumPrimaryFlux",
            "mDM": self.mDM.to_string(),
            "channel": self.channel,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a `ContinuumPrimaryFlux` from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by `to_dict`, containing ``mDM`` and
            ``channel``, and optionally ``source`` (defaults to
            ``"pppc4"`` if absent).

        Returns
        -------
        flux : `ContinuumPrimaryFlux`
            New instance reconstructed from ``data``.
        """
        return cls(
            mDM=u.Quantity(data["mDM"]),
            channel=data["channel"],
            source=data.get("source", "pppc4"),
        )


@deprecated("2.2", alternative="ContinuumPrimaryFlux")
class PrimaryFlux(ContinuumPrimaryFlux):
    tag = ["PrimaryFlux", "dm-pf"]
    pass


PRIMARY_FLUX_REGISTRY = {
    cls.tag[0]: cls
    for cls in (
        ContinuumPrimaryFlux,
        PrimaryFlux,
        MonochromaticPrimaryFlux,
        VIBPrimaryFlux,
        BoxPrimaryFlux,
    )
}

# ---------------------------------------------------------------------------
# Spectral model
# ---------------------------------------------------------------------------


class DarkMatterSpectralModel(SpectralModel):
    r"""Dark matter spectral model.

    Computes the differential gamma-ray flux expected from dark matter
    annihilation or decay in a region with a given  astrophysical factor, combining the
    thermally averaged annihilation cross-section or particle decay lifetime, the chosen primary
    particle-physics spectrum, and a fit-time normalization scale.

    For annihilation, the gamma-ray flux is computed as:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\langle \sigma\nu \rangle}{4\pi k m^2_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    where :math:`\langle \sigma\nu \rangle` is the thermal relic
    cross-section, :math:`k` accounts for the dark matter particle type
    (Majorana or Dirac), :math:`m_{\mathrm{DM}}` is the dark matter mass,
    :math:`\mathrm dN/\mathrm dE` is the primary photon spectrum per
    annihilation, and :math:`J(\Delta\Omega)` is the astrophysical
    J-factor.

    For decay, the flux is computed as:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\Gamma}{4\pi m_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    where :math:`\Gamma = 1/\tau` is the decay rate (inverse lifetime),
    :math:`m_{\mathrm{DM}}` is the dark matter mass,
    :math:`\mathrm dN/\mathrm dE` is the primary photon spectrum per decay,
    and :math:`J(\Delta\Omega)` is the astrophysical D-factor (here denoted
    generically as the astrophysical factor).

    .. note::

        Since Gammapy v2.2, the default source has changed from "pppc4" to "cosmixs".

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass.
    channel : str or None
        Annihilation/Decay channel for the primary flux spectrum, e.g. ``"b"`` for bb̄.
        See `ContinuumPrimaryFlux.channel_registry` for available channels. Can be
        ``None`` when ``primary_flux`` is given and does not depend on a channel, as
        for `MonochromaticPrimaryFlux`, `VIBPrimaryFlux` and `BoxPrimaryFlux`.
    scale : float, optional
        Dimensionless normalization parameter applied multiplicatively to
        the predicted flux, intended to be left free in spectral fits.
        Default is 1.
    factor : `~astropy.units.Quantity`, optional
        Astrophysical factor (integrated squared dark matter density
        along the line of sight and over the solid angle), needed when a
        `~gammapy.modeling.models.PointSpatialModel` is used for the
        spatial component. Default is 1 (dimensionless) and in a frozen form.
        If a unit is given it should be convertible to GeV2 cm-5 for annihilation
        and GeV cm-2 in the case of decay
    z : float, optional
        Redshift of the source. The primary flux is evaluated at the
        redshifted energy ``energy * (1 + z)``. Default is 0.
    k : int, optional
        Only used in the case of annihilation. Dark matter particle type: ``2`` for Majorana particles (which are
        their own antiparticles, giving a factor of 2 in the annihilation
        rate) or ``4`` for Dirac particles. Default is 2.
    primary_flux : `~gammapy.modeling.models.SpectralModel`, optional
        Primary photon spectrum dN/dE per annihilation or decay event, e.g.
        `ContinuumPrimaryFlux`, `MonochromaticPrimaryFlux`, `VIBPrimaryFlux` or
        `BoxPrimaryFlux`. If not provided, a default `ContinuumPrimaryFlux` is
        constructed using ``mDM`` and ``channel``.
    source : str, optional
        Data source for the spectral tables. Options are:

        * ``"cosmixs"`` (default): Arina et al. (2024) / CosmiXs tables.
        * ``"pppc4"``: Cirelli et al. (2011, 2016) PPPC4DMID tables.
        * A path to a custom file readable by `astropy.table.Table.read`
          (extensions ``.dat``, ``.txt``, ``.csv``, or ``.ecsv``).

            If a custom file path is provided, it must contain ``"mDM"`` and
            ``"Log[10,x]"`` columns (after applying ``mapping_dict`` if given),
            plus columns named after the requested annihilation channel(s)
            using the internal channel registry naming convention.
    mapping_dict : dict, optional
        Mapping dictionary used to rename the columns of a custom source
        file to the expected internal column names. Only used when
        ``source`` is a custom file path. Format is
        ``{actual_column_name: expected_column_name}``, and must cover the
        mandatory columns ``"mDM"`` and ``"Log[10,x]"``.
    annihilation : bool
        Boolean value indicating if the dark matter spectral model comes from an
        annihilation or decay scenario. Default is True, which indicates annihilation.

    Examples
    --------
    This is how to instantiate a `DarkMatterSpectralModel`::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterSpectralModel

        >>> channel = "b"
        >>> mDM = 5000*u.Unit("GeV")
        >>> factor = 3.41e19 * u.Unit("GeV2 cm-5")
        >>> modelDM = DarkMatterSpectralModel(mDM=mDM,
          channel=channel, factor=factor, annihilation=True)  # noqa: E501

    References
    ----------
    .. [1] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [2] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for
       indirect dark matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [3] `Di Mauro et al. (2025), "Nailing down the theoretical
       uncertainties of Dbar spectrum produced from dark matter"
       <https://arxiv.org/abs/2411.04815>`_ and available on
       `GitHub <https://github.com/ajueid/CosmiXs>`_
    """

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section."""

    LIFETIME_AGE_OF_UNIVERSE = 4.3e17 * u.Unit("s")
    """Use age of universe as lifetime"""

    scale = Parameter("scale", 1, unit="", interp="log")
    factor = Parameter("factor", 1, unit="", interp="log", frozen=True)
    tag = ["DarkMatterSpectralModel", "dm-spectralmodel"]

    @deprecated_renamed_argument("mass", "mDM", "2.2")
    @deprecated_renamed_argument("jfactor", "factor", "2.2")
    def __init__(
        self,
        mDM,
        channel,
        *,
        scale=scale.quantity,
        factor=factor.quantity,
        z=0,
        k=2,
        primary_flux=None,
        source=None,
        mapping_dict=None,
        annihilation=True,
    ):
        self._annihilation = annihilation

        if self._annihilation:
            if k not in (2, 4):
                raise ValueError(f"k must be 2 (Majorana) or 4 (Dirac), got: {k}.")
            self._k = k
        else:
            self._k = None

        if z < 0:
            raise ValueError(f"Redshift z must be >= 0, got: {z}.")

        self._z = z
        self._mDM = u.Quantity(mDM)
        self._channel = channel
        self.source = source
        self.mapping_dict = mapping_dict
        if primary_flux is not None:
            if hasattr(primary_flux, "channel"):
                primary_flux.channel = channel
            if hasattr(primary_flux, "mDM"):
                primary_flux.mDM = self._expected_primary_flux_mass
        else:
            primary_flux = ContinuumPrimaryFlux(
                self._expected_primary_flux_mass,
                channel=self.channel,
                source=self.source,
                mapping_dict=self.mapping_dict,
            )
        self._primary_flux = primary_flux

        factor = u.Quantity(factor)

        if np.any(factor.value <= 0):
            raise ValueError(f"factor must be positive, got {factor}.")

        if factor.unit != u.dimensionless_unscaled:
            expected = u.Unit("GeV2 cm-5") if annihilation else u.Unit("GeV cm-2")
            if not factor.unit.is_equivalent(expected):
                raise u.UnitConversionError(
                    f"factor must be convertible to {expected} "
                    f"({'annihilation' if annihilation else 'decay'}), got {factor.unit}."
                )
            factor = factor.to(expected)

        super().__init__(scale=scale, factor=factor.value)
        self.factor.unit = factor.unit

    @property
    def annihilation(self):
        """Annihilation/Decay flag"""
        return self._annihilation

    @property
    def mDM(self):
        """Dark matter mass."""
        return self._mDM

    @property
    def z(self):
        """Source redshift (must be >= 0)."""
        return self._z

    @property
    def primary_flux(self):
        """Primary flux model."""
        return self._primary_flux

    @property
    def channel(self):
        """Channel of the default primary flux spectrum."""
        return self._channel

    @property
    def k(self):
        """DM particle type (2: Majorana, 4: Dirac)."""
        return self._k

    @property
    def _expected_primary_flux_mass(self):
        """Expected primary flux mass for annihilation.

        Returns
        -------
        mass : `~astropy.units.Quantity`
            Equal to ``mDM``, since in annihilation each dark matter
            particle pair has a total rest energy of ``2 * mDM``, shared
            between two particles, and the primary flux tables are indexed
            by the per-particle mass ``mDM``.
        """
        return self._mDM if self._annihilation else self._mDM / 2

    def evaluate(self, energy, scale, factor):
        """Evaluate the dark matter annihilation differential flux.

        For annihilation, computes:
        ``flux = scale * factor * THERMAL_RELIC_CROSS_SECTION
        * primary_flux(energy * (1 + z)) / k / mDM**2 / (4 * pi)``.

        For decay, computes:
        ``flux = scale * factor * primary_flux(energy * (1 + z))
        / LIFETIME_AGE_OF_UNIVERSE / mDM / (4 * pi)``.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the flux.
        scale : float
            Current value of the ``scale`` normalization parameter.
        factor : `~astropy.units.Quantity`
            Current value of the ``factor`` parameter.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential gamma-ray flux per unit energy.
        """
        flux = scale * factor * self.primary_flux(energy=energy * (1 + self.z))
        if self.annihilation:
            flux = flux * self.THERMAL_RELIC_CROSS_SECTION / self.k / self.mDM**2
        else:
            flux = flux / self.LIFETIME_AGE_OF_UNIVERSE / self.mDM
        return flux / (4 * np.pi)

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Extends the base `~gammapy.modeling.models.SpectralModel.to_dict`
        output with ``channel``, ``mDM``, ``factor``, ``z``, ``k``, the
        serialized ``primary_flux`` (via its own `to_dict`), ``source``, ``mapping_dict``
        and ``annihilation``.

        Parameters
        ----------
        full_output : bool, optional
            Passed through to the parent class's `to_dict`. Default is
            False.

        Returns
        -------
        data : dict
            Dictionary representation suitable for round-tripping via
            `from_dict`.
        """
        data = super().to_dict(full_output=full_output)
        for p in data["spectral"]["parameters"]:
            if p["name"] == "factor":
                p["unit"] = self.factor.unit.to_string()
        data["spectral"]["channel"] = self.channel
        data["spectral"]["mDM"] = self.mDM.to_string()
        data["spectral"]["z"] = self.z
        data["spectral"]["k"] = self.k
        data["spectral"]["primary_flux"] = self.primary_flux.to_dict()
        data["spectral"]["source"] = self.source
        data["spectral"]["mapping_dict"] = self.mapping_dict
        data["spectral"]["annihilation"] = self.annihilation
        return data

    @classmethod
    def from_dict(cls, data):
        """Construct a `DarkMatterSpectralModel` from a dictionary.

        Reconstructs the ``primary_flux`` sub-model from the registry of
        known primary flux types, rebuilds ``scale`` and ``factor`` from the
        serialized parameter list (including unit, bounds, frozen state and
        prior), and passes the remaining fields to the constructor.

        .. note::

        Since Gammapy v2.2,  dictionaries using the previous format, in which
        the astrophysical factor was stored separately in a ``factor`` (or ``jfactor``)
        field, are still supported for backward compatibility.

        Parameters
        ----------
        data : dict
            Dictionary with a top-level ``"spectral"`` key, as produced by
            `to_dict`, containing ``mDM``, ``channel``, ``z``, ``k``,
            ``primary_flux``, ``annihilation`` and ``parameters``.

        Returns
        -------
        model : `DarkMatterSpectralModel`
            New instance reconstructed from ``data``.
        """
        data = copy.deepcopy(data["spectral"])
        model_type = data.pop("type", "")
        default_annihilation = "Decay" not in model_type

        _RENAMED_FIELDS = {"mass": "mDM", "jfactor": "factor"}
        for old_name, new_name in _RENAMED_FIELDS.items():
            if old_name in data:
                warnings.warn(
                    f"The '{old_name}' field is deprecated since v2.2 and will "
                    f"be removed in a future version. Use '{new_name}' instead. "
                    f"This serialized model appears to use the old format.",
                    GammapyDeprecationWarning,
                )
                data[new_name] = data.pop(old_name)

        pf_data = data.pop("primary_flux", None)
        if pf_data is None:
            # Old format: primary flux rebuilt from the flat fields
            primary_flux = ContinuumPrimaryFlux(
                data["mDM"],
                channel=data["channel"],
                source=data.get("source"),
                mapping_dict=data.get("mapping_dict"),
            )
        else:
            pf_cls = PRIMARY_FLUX_REGISTRY.get(pf_data["type"])
            if pf_cls is None:
                raise ValueError(
                    f"Unknown primary_flux type: '{pf_data['type']}'. "
                    f"Available: {list(PRIMARY_FLUX_REGISTRY.keys())}"
                )
            primary_flux = pf_cls.from_dict(pf_data)

        data.pop("source", None)
        data.pop("mapping_dict", None)
        annihilation = data.pop("annihilation", default_annihilation)
        pars = {p["name"]: p for p in data.pop("parameters")}

        if "factor" in pars:  # new format: factor is a Parameter
            p = pars["factor"]
            factor = u.Quantity(p["value"], p.get("unit", ""))
            data.pop("factor", None)
        else:  # old format: factor serialized as a string field
            factor = u.Quantity(data.pop("factor", 1))

        model = cls(
            scale=pars["scale"]["value"],
            factor=factor,
            primary_flux=primary_flux,
            annihilation=annihilation,
            **data,
        )

        # restore parameter state (bounds, frozen, error, prior)
        for name, p in pars.items():
            par = getattr(model, name)
            for key in ("min", "max", "error", "frozen"):
                if key in p:
                    setattr(par, key, p[key])
            if "prior" in p:
                prior_data = p["prior"]
                prior_type = prior_data.get("prior", prior_data)["type"]
                par.prior = PRIOR_REGISTRY.get_cls(prior_type).from_dict(prior_data)

        return model


@deprecated("2.2", alternative="DarkMatterSpectralModel")
class DarkMatterAnnihilationSpectralModel(DarkMatterSpectralModel):
    scale = Parameter("scale", 1, unit="", interp="log")
    factor = Parameter("factor", 1, unit="", interp="log", frozen=True)
    tag = ["DarkMatterAnnihilationSpectralModel", "dm-annihilation"]

    def __init__(self, *args, **kwargs):
        kwargs["annihilation"] = True
        super().__init__(*args, **kwargs)


@deprecated("2.2", alternative="DarkMatterSpectralModel")
class DarkMatterDecaySpectralModel(DarkMatterSpectralModel):
    scale = Parameter("scale", 1, unit="", interp="log")
    factor = Parameter("factor", 1, unit="", interp="log", frozen=True)
    tag = ["DarkMatterDecaySpectralModel", "dm-decay"]

    def __init__(self, *args, **kwargs):
        kwargs["annihilation"] = False
        super().__init__(*args, **kwargs)
