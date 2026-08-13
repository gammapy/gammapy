# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.modeling import Parameter
from gammapy.modeling.models import SpectralModel, TemplateNDSpectralModel
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_map_columns

__all__ = [
    "PrimaryFlux",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "MonochromaticPrimaryFlux",
    "VIBPrimaryFlux",
    "BoxPrimaryFlux",
]
log = logging.getLogger(__name__)


class _DarkMatterMassValidator:
    """DM mass validator for primary flux models."""

    @property
    def mDM(self):
        """Dark matter particle mass."""
        return self._mDM

    @mDM.setter
    def mDM(self, value):
        try:
            val = u.Quantity(value).to_value("GeV")
        except u.UnitConversionError:
            raise u.UnitConversionError(
                f"mDM must have energy units (e.g. GeV, TeV), "
                f"got: {u.Quantity(value).unit}"
            )
        if val <= 0:
            raise ValueError("mDM must be strictly positive.")
        self._mDM = u.Quantity(value)


class MonochromaticPrimaryFlux(_DarkMatterMassValidator, SpectralModel):
    tag = ["MonochromaticPrimaryFlux", "dm-mono"]

    """Monochromatic gamma-ray line from dark matter annihilation.

    Implements the spectral shape for χχ → γγ (two-photon final state) and
    χχ → γX (one photon plus a massive counterpart X, e.g. χχ → γZ or
    χχ → γh). The line is approximated as a narrow Gaussian centered at the
    kinematic line energy, to provide a smooth, differentiable spectral
    shape suitable for likelihood-based fitting.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be convertible to GeV and strictly
        positive.
    n_gamma_photons : {1, 2}
        Number of monochromatic gamma-ray photons produced per annihilation
        event. Use ``2`` for the two-photon channel (χχ → γγ) and ``1`` for
        the one-photon-plus-counterpart channel (χχ → γX).
    counterpart : str or None, optional
        Label identifying the massive counterpart particle X for the
        one-photon channel (e.g. ``"z"`` for the γZ channel, ``"h"`` for the
        γh channel). Required and validated when ``n_gamma_photons=1``.
        Must be ``None`` (or will be reset to ``None`` with a warning) when
        ``n_gamma_photons=2``, since no counterpart is involved.
    counterpart_mass : float or `~astropy.units.Quantity` or None, optional
        Mass of the counterpart particle, in GeV if given as a bare number.
        If ``counterpart`` is ``"z"`` or ``"h"``, the standard Z or Higgs
        boson mass is used automatically when this is ``None``. For any
        other counterpart label not present in the internal registry, this
        value must be provided explicitly.
    sigma_rel : float, optional
        Relative width of the Gaussian approximation to the spectral line,
        expressed as a fraction of the line energy E₀ (i.e.
        ``sigma = sigma_rel * E_0``). Default is 0.01 (1%).

    Warns
    -----
    UserWarning
        If ``counterpart`` is set while ``n_gamma_photons=2``; the
        counterpart is ignored and reset to ``None``.
    UserWarning
        If the line energy E₀ falls outside the energy range passed to
        `evaluate`, in which case the returned spectrum is effectively zero
        everywhere.

    Examples
    --------
    Two-photon line at mDM = 1 TeV::

        >>> import astropy.units as u
        >>> flux = MonochromaticPrimaryFlux(mDM=1*u.TeV, n_gamma_photons=2)

    One-photon line with a Z-boson counterpart::

        >>> flux = MonochromaticPrimaryFlux(
        ...     mDM=1*u.TeV, n_gamma_photons=1, counterpart="z"
        ... )
    References
    ----------
    `Bergström & Snellman (1988), "Observable monochromatic photons from cosmic photino
    annihilation"
    DOI:10.1103/PhysRevD.37.3737 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.37.3737>_
    """
    counterparts_particles = {
        "z": {"mass": 91.2},  # Channel Z-gamma
        "h": {"mass": 125.1},  # Channel Higgs-gamma
    }

    DEFAULT_SIGMA_REL = 0.01

    def __init__(
        self,
        mDM,
        n_gamma_photons,
        counterpart=None,
        counterpart_mass=None,
        sigma_rel=None,
    ):
        self.mDM = mDM
        self.n_gamma_photons = n_gamma_photons
        self.counterpart = counterpart
        self.counterpart_mass = counterpart_mass
        self.sigma_rel = sigma_rel if sigma_rel is not None else self.DEFAULT_SIGMA_REL
        super().__init__()

    def get_line_energy(self):
        """Compute the gamma-ray line energy E₀.

        For the two-photon channel (``n_gamma_photons=2``), the line
        energy equals the dark matter mass, ``E_0 = mDM``. For the
        one-photon channel (``n_gamma_photons=1``), the line energy is
        reduced by the kinematics of producing the massive counterpart X,
        ``E_0 = mDM * (1 - counterpart_mass**2 / (4 * mDM**2))``.

        Returns
        -------
        E_0 : `~astropy.units.Quantity`
            Energy of the monochromatic gamma-ray line.
        """

        if self.n_gamma_photons == 2:
            return self.mDM
        else:
            return self.mDM * (1 - (self.counterpart_mass**2) / (4 * self.mDM**2))

    def evaluate(self, energy):
        """Evaluate the monochromatic primary flux spectrum dN/dE.

        The spectral line is represented as a Gaussian of relative width
        ``sigma_rel`` centered on the line energy E₀ (as returned by
        `get_line_energy`), normalized so that its integral equals
        ``n_gamma_photons`` (i.e. the total photon yield per annihilation).

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the spectrum.

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy, in units of GeV⁻¹.

        Warns
        -----
        UserWarning
            If the line energy E₀ lies outside the range spanned by
            ``energy``, meaning the returned spectrum will be effectively
            zero over the requested energy range.
        """
        E_0 = self.get_line_energy()
        sigma = self.sigma_rel * E_0

        x = (energy - E_0) / sigma
        dN_dE = (self.n_gamma_photons / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * x**2
        )

        E_0_gev = E_0.to_value("GeV")
        e_min = energy.to_value("GeV").min()
        e_max = energy.to_value("GeV").max()
        if E_0_gev < e_min or E_0_gev > e_max:
            warnings.warn(
                f"The energy line E₀ = {E_0_gev:.3f} GeV is outside the "
                f"energy range [{e_min:.3f}, {e_max:.3f}] GeV. "
                "The returned spectrum will be effectively zero.",
                UserWarning,
                stacklevel=2,
            )

        return dN_dE.to("GeV-1")

    @property
    def n_gamma_photons(self):
        """Number of gamma photons produced."""
        return self._n_gamma_photons

    @n_gamma_photons.setter
    def n_gamma_photons(self, value):
        if value is None or value not in (1, 2):
            raise ValueError(
                "Number of produced photons must be 1 or 2 for this "
                "monochromatic line spectrum."
            )
        self._n_gamma_photons = value

    @property
    def counterpart(self):
        """Counterpart particle for the monochromatic line."""
        return self._counterpart

    @counterpart.setter
    def counterpart(self, value):
        if self.n_gamma_photons == 1 and value is None:
            raise ValueError(
                "Counterpart particle must be indicated for a "
                "monochromatic line with 1 photon."
            )

        self._counterpart = value

        if self.n_gamma_photons == 2 and value is not None:
            warnings.warn(
                "The counterpart parameter is ignored since you are "
                "studying a 2-photon channel.",
                UserWarning,
                stacklevel=2,
            )
            self._counterpart = None

    @property
    def counterpart_mass(self):
        """Counterpart particle mass."""
        return self._counterpart_mass

    @counterpart_mass.setter
    def counterpart_mass(self, value):
        if self.counterpart is not None:
            if self.counterpart not in self.counterparts_particles:
                if value is None:
                    raise ValueError(
                        f"Since the indicated counterpart particle "
                        f"'{self.counterpart}' is not in the registry, "
                        "its mass must be provided by the user."
                    )
                self._counterpart_mass = value * u.GeV
            else:
                mass_val = (
                    value
                    if value is not None
                    else self.counterparts_particles[self.counterpart]["mass"]
                )
                self._counterpart_mass = u.Quantity(mass_val, u.GeV).to(u.GeV)

            if 2 * self.mDM <= self._counterpart_mass:
                raise ValueError(
                    f"Kinematically forbidden: the available energy "
                    f"is not enough to produce the counterpart particle "
                    f"(mass = {self._counterpart_mass})."
                )
        else:
            self._counterpart_mass = None

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
            ``n_gamma_photons``, ``counterpart``, ``counterpart_mass``, and
            ``sigma_rel``, suitable for round-tripping via `from_dict`.
        """
        return {
            "type": "MonochromaticPrimaryFlux",
            "mDM": self.mDM.to_string(),
            "n_gamma_photons": self.n_gamma_photons,
            "counterpart": self.counterpart,
            "counterpart_mass": (
                self.counterpart_mass.to_string()
                if self.counterpart_mass is not None
                else None
            ),
            "sigma_rel": self.sigma_rel,
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a `MonochromaticPrimaryFlux` from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as produced by `to_dict`, containing ``mDM``,
            ``n_gamma_photons``, and optionally ``counterpart``,
            ``counterpart_mass``, and ``sigma_rel``.

        Returns
        -------
        flux : `MonochromaticPrimaryFlux`
            New instance reconstructed from ``data``.
        """
        return cls(
            mDM=u.Quantity(data["mDM"]),
            n_gamma_photons=data["n_gamma_photons"],
            counterpart=data.get("counterpart"),
            counterpart_mass=(
                u.Quantity(data["counterpart_mass"])
                if data.get("counterpart_mass")
                else None
            ),
            sigma_rel=data.get("sigma_rel"),
        )


class VIBPrimaryFlux(_DarkMatterMassValidator, SpectralModel):
    """Virtual Internal Bremsstrahlung (VIB) spectral shape.

    Describes the gamma-ray spectrum from internal bremsstrahlung in dark
    matter annihilation, valid for Majorana dark matter annihilating to
    Standard Model fermion pairs in the limit of large dark matter mass and
    nearly degenerate mediating sfermions.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be convertible to GeV and strictly
        positive.

    Notes
    -----
    The spectral shape is defined in terms of ``x = E / mDM`` and is only
    non-zero for ``0 < x < 1``; outside this range the returned flux is
    zero. The overall normalization constant ``A_VIB`` follows from the
    standard VIB spectral formula.


    References
    ----------
    `Bringmann et al. (2007), "New Gamma-Ray Contributions to Supersymmetric \n
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

        Computes the VIB spectral shape as a function of
        ``x = energy / mDM``, restricting the evaluation to the physically
        valid range ``0 < x < 1`` (the spectrum is zero outside this range)
        and clipping any small negative numerical artifacts to zero.

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


class BoxPrimaryFlux(_DarkMatterMassValidator, SpectralModel):
    """Box-shaped spectral signal from χχ → φφ → γγγγ.

    Describes the gamma-ray spectrum resulting from dark matter
    annihilation into a pair of long-lived neutral intermediate states φ,
    each of which subsequently decays into a pair of photons (φ → γγ). In
    the rest frame of each φ, the photons are monochromatic; boosting to
    the dark matter rest frame produces a flat ("box-shaped") spectrum
    between two energy edges for each intermediate state.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass. Must be convertible to GeV and strictly
        positive.
    mPhi : array_like
        Mass(es) of the intermediate state(s) φ, in units convertible to
        GeV. May be:

        - a single value or a one-element array (e.g. ``[100] * u.GeV``),
          in which case both intermediate states are assumed identical
          (χχ → φφ), producing a single box; or
        - a two-element array (e.g. ``[100, 50] * u.GeV``), to model two
          distinct intermediate states (χχ → φ₁φ₂), producing two
          (possibly overlapping) boxes.

    Warns
    -----
    UserWarning
        If the two spectral boxes (for ``mPhi1`` and ``mPhi2``) overlap in
        energy, in which case the spectrum in the overlap region is
        double-counted.

    References
    ----------
    `Ibarra et al. (2018), "Dark matter constraints from box-shaped gamma-ray features"
    DOI:10.1088/1475-7516/2012/07/043 <https://arxiv.org/abs/1205.0007>`_
    """

    tag = ["BoxPrimaryFlux", "dm-box"]

    def __init__(self, mDM, mPhi):
        self.mDM = u.Quantity(mDM)
        self.mPhi = mPhi
        super().__init__()

    def evaluate(self, energy):
        """Evaluate the box-shaped primary flux spectrum dN/dE.

        For each intermediate state φᵢ, computes the energy range
        ``[E_min_i, E_max_i]`` (centered on the boosted photon energy
        ``E_phi_i`` with half-width ``delta_E``) and adds a constant
        contribution ``2 / delta_E`` to ``dN/dE`` for energies falling
        inside that range, reflecting the two photons produced per φᵢ
        decay.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the spectrum.

        Returns
        -------
        dN_dE : `~astropy.units.Quantity`
            Differential photon yield per unit energy, in units of GeV⁻¹.

        Warns
        -----
        UserWarning
            If the energy ranges of the two boxes overlap.
        """

        E_phi1, E_phi2 = self.energies_phi
        delta_E = self.delta_E

        dN_dE = np.zeros(len(energy)) / u.GeV

        E_min1 = (E_phi1 - delta_E) / 2
        E_max1 = (E_phi1 + delta_E) / 2
        in_box1 = (energy >= E_min1) & (energy <= E_max1)

        E_min2 = (E_phi2 - delta_E) / 2
        E_max2 = (E_phi2 + delta_E) / 2
        in_box2 = (energy >= E_min2) & (energy <= E_max2)

        two_distinct_boxes = not u.allclose(E_phi1, E_phi2)
        if two_distinct_boxes and E_min2 < E_max1:
            warnings.warn(
                "The two spectral boxes overlap in energy. This can happen "
                "when the two intermediate-particle masses are very close. "
                "The spectrum in the overlap region will be double-counted.",
                UserWarning,
                stacklevel=2,
            )

        if two_distinct_boxes:
            dN_dE[in_box1] += 2.0 / delta_E
            dN_dE[in_box2] += 2.0 / delta_E
        else:
            # single box: avoid double-adding the same contribution twice
            dN_dE[in_box1] += 2.0 / delta_E

        return dN_dE.to("GeV-1")

    @property
    def energies_phi(self):
        """Energy of each intermediate particle in the centre-of-mass frame."""
        E_phi1 = self.mDM + (self.mPhi1**2 - self.mPhi2**2) / (4 * self.mDM)
        E_phi2 = self.mDM + (self.mPhi2**2 - self.mPhi1**2) / (4 * self.mDM)
        return E_phi1, E_phi2

    @property
    def delta_E(self):
        """Width of the boxes (momentum magnitude). Both boxes share this width."""
        E_phi1, _ = self.energies_phi
        return np.sqrt(E_phi1**2 - self.mPhi1**2)

    @property
    def mPhi(self):
        """Intermediate particle masses."""
        return self._mPhi

    @mPhi.setter
    def mPhi(self, value):
        if value is not None:
            if getattr(value, "ndim", 1) == 0 or not isinstance(
                value, (list, tuple, np.ndarray)
            ):
                mPhi_list = [value]
            else:
                mPhi_list = value

            if len(mPhi_list) == 1:
                self.mPhi1 = u.Quantity(mPhi_list[0])
                self.mPhi2 = self.mPhi1
            elif len(mPhi_list) == 2:
                self.mPhi1 = u.Quantity(mPhi_list[0])
                self.mPhi2 = u.Quantity(mPhi_list[1])
            else:
                raise ValueError(
                    f"The intermediate mass array must contain exactly 1 or 2 values. "
                    f"Received {len(mPhi_list)}."
                )

            for label, m in [("mPhi1", self.mPhi1), ("mPhi2", self.mPhi2)]:
                if u.Quantity(m).to_value("GeV") <= 0:
                    raise ValueError(f"{label} must be strictly positive.")

            if self.mPhi1 + self.mPhi2 >= 2 * self.mDM:
                raise ValueError(
                    f"Kinematically forbidden: the sum of intermediate masses "
                    f"({self.mPhi1 + self.mPhi2}) must be "
                    f"less than 2 * mDM ({2 * self.mDM})."
                )

            self._mPhi = value
        else:
            self._mPhi = None
            self.mPhi1 = None
            self.mPhi2 = None

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
            mPhi=[u.Quantity(m) for m in data["mPhi"]],
        )


class PrimaryFlux(TemplateNDSpectralModel):
    """DM-annihilation gamma-ray spectra.

    Based on the precomputed models of PPPC4 DM ID by [1]_, [2]_ and CosmiXs by [3]_, [4]_.
    All available annihilation channels can be found there. The dark matter mass will be set
    to the nearest available value. The spectra will be available as
    `~gammapy.modeling.models.TemplateNDSpectralModel` for a chosen dark matter mass and
    annihilation channel. Using a `~gammapy.modeling.models.TemplateNDSpectralModel`
    allows the interpolation between different dark matter masses.

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass as rest mass energy.
    channel : str
        Annihilation channel. List available channels with `~gammapy.astro.darkmatter.PrimaryFlux.allowed_channels`.
    source : str or Table, optional
        Data source for the spectra. Options are:

        * ``"pppc4"`` (default): Cirelli et al. 2011.
        * ``"cosmixs"``: Cirelli et al. 2024.
        * A path to a custom file: Any format readable by `astropy.table.Table.read`
        (e.g., .ecsv, .fits, .csv, .dat).
        * An Astropy Table that read the desired path

        If a custom file path is provided, it must contain 'mDM' (mass of dark
        matter particle) and 'Log[10,x]' (energy) columns, plus columns named after
        the requested annihilation/decay channels (see the documentation).
    mapping_dict : dict, optional
        Mapping dictionary to map the columns of the custom source file to the expected
        column names. This is only needed if a file as a spectra source is provided
        and the column names in the file do not match the expected names. The dictionary
        should have the format {actual_column_name_in_file:expected_column_name}.
        An example of the expected columns can be found in the documentation.


    References
    ----------
    .. [1] `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist cookbook
    for dark matter indirect detection" <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    .. [2] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist Cookbook
    for Dark Matter Indirect Detection" <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [3] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for indirect dark
    matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [4] `Di Mauro et al. (2025), "Nailing down the theoretical uncertainties of Dbar
     spectrum produced from dark matter" <https://arxiv.org/abs/2411.04815>`_

    """  # noqa: E501

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
        "dNdLog10x[u]": "u",  # Does not exist on PPPC4, it is equivalent to q
        "dNdLog10x[d]": "d",  # Does not exist on PPPC4, it is equivalent to q
        "dNdLog10x[s]": "s",  # Does not exist on PPPC4, it is equivalent to q
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
        "dNdLog10x[aZ]": None,  # Does not exist  on PPPC4
        "dNdLog10x[HZ]": None,  # Does not exist  on PPPC4
    }

    tag = ["PrimaryFlux", "dm-pf"]

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

    @property
    def mDM(self):
        """Dark matter mass."""
        return u.Quantity(self.mass.value, "GeV")

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
            self._source = "pppc4"

            log.info(
                "\nSince no spectra source has been chosen, PPPC4 will be \
                    used by default.\n",
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
                f"Invalid channel: {channel}\nAvailable: {self.allowed_channels}\n"
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

    def evaluate(self, energy, *args):
        """Evaluate the primary flux."""

        args = list(args)
        args.append(self.mDM)

        log10x = np.log10(energy / self.mDM)

        dN_dlogx = super().evaluate(log10x, *args)
        dN_dE = dN_dlogx / (energy * np.log(10))
        return dN_dE


class DarkMatterAnnihilationSpectralModel(SpectralModel):
    r"""Dark matter annihilation spectral model.

    The gamma-ray flux is computed as follows:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\langle \sigma\nu \rangle}{4\pi k m^2_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Dark matter mass.
    channel : str
        Annihilation channel for `~gammapy.astro.darkmatter.PrimaryFlux`, e.g. "b"
        for "bbar".
        See `PrimaryFlux.channel_registry` for more.
    scale : float
        Scale parameter for model fitting.
    jfactor : `~astropy.units.Quantity`, optional
        Integrated J-Factor needed when `~gammapy.modeling.models.PointSpatialModel`
        is used. Default is 1.
    z : float, optional
        Redshift value. Default is 0.
    k : int, optional
        Type of dark matter particle (k:2 Majorana, k:4 Dirac). Default is 2.
    source : str or Table, optional
        Data source for the spectra. Options are:

        * ``"pppc4"`` (default): Cirelli et al. 2011.
        * ``"cosmixs"``: Cirelli et al. 2024.
        * A path to a custom file: Any format readable by `astropy.table.Table.read`
        (e.g., .ecsv, .fits, .csv, .dat).
        * An Astropy Table that read the desired path


        If a custom file path is provided, it must contain 'mDM' (mass of dark
        matter particle) and 'Log[10,x]' (energy) columns, plus columns named after
        the requested annihilation/decay channels (see the documentation).
    mapping_dict : dict, optional
        Mapping dictionary to map the columns of the custom source file to the expected
        column names. This is only needed if a file as a spectra source is provided
        and the column names in the file do not match the expected names. The dictionary
        should have the format {actual_column_name_in_file:expected_column_name}.
        An example of the expected columns can be found in the documentation.

    Examples
    --------
    This is how to instantiate a `DarkMatterAnnihilationSpectralModel` model::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel

        >>> channel = "b"
        >>> massDM = 5000*u.Unit("GeV")
        >>> jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
        >>> modelDM = DarkMatterAnnihilationSpectralModel(mass=massDM,
          channel=channel, jfactor=jfactor)  # noqa: E501

    References
    ----------
    `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist cookbook for dark
      matter indirect detection" <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    """

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section"""

    scale = Parameter(
        "scale",
        1,
        unit="",
        interp="log",
    )
    tag = ["DarkMatterAnnihilationSpectralModel", "dm-annihilation"]

    def __init__(
        self,
        mass,
        channel,
        scale=scale.quantity,
        jfactor=1,
        z=0,
        k=2,
        source=None,
        mapping_dict=None,
    ):
        self.k = k
        self.z = z
        self.mass = u.Quantity(mass)
        self.channel = channel
        self.jfactor = u.Quantity(jfactor)
        self.source = source
        self.mapping_dict = mapping_dict
        self.primary_flux = PrimaryFlux(
            mass,
            channel=self.channel,
            source=self.source,
            mapping_dict=self.mapping_dict,
        )

        super().__init__(scale=scale)

    def evaluate(self, energy, scale):
        """Evaluate dark matter annihilation model."""
        flux = (
            scale
            * self.jfactor
            * self.THERMAL_RELIC_CROSS_SECTION
            * self.primary_flux(energy=energy * (1 + self.z))
            / self.k
            / self.mass
            / self.mass
            / (4 * np.pi)
        )
        return flux

    def to_dict(self, full_output=False):
        """Convert to dictionary."""
        data = super().to_dict(full_output=full_output)
        data["spectral"]["channel"] = self.channel
        data["spectral"]["mass"] = self.mass.to_string()
        data["spectral"]["jfactor"] = self.jfactor.to_string()
        data["spectral"]["z"] = self.z
        data["spectral"]["k"] = self.k
        data["spectral"]["source"] = self.source
        data["spectral"]["mapping_dict"] = self.mapping_dict
        return data

    @classmethod
    def from_dict(cls, data):
        """Create spectral model from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with model data.

        Returns
        -------
        model : `DarkMatterAnnihilationSpectralModel`
            Dark matter annihilation spectral model.
        """
        data = data["spectral"]
        data.pop("type")
        parameters = data.pop("parameters")
        scale = [p["value"] for p in parameters if p["name"] == "scale"][0]
        return cls(scale=scale, **data)


class DarkMatterDecaySpectralModel(SpectralModel):
    r"""Dark matter decay spectral model.

    The gamma-ray flux is computed as follows:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\Gamma}{4\pi m_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Dark matter mass.
    channel : str
        Decay channel for `~gammapy.astro.darkmatter.PrimaryFlux`, e.g. "b" for "bbar".
        See `PrimaryFlux.channel_registry` for more.
    scale : float
        Scale parameter for model fitting
    jfactor : `~astropy.units.Quantity`, optional
        Integrated J-Factor needed when `~gammapy.modeling.models.PointSpatialModel`
        is used. Default is 1.
    z : float, optional
        Redshift value. Default is 0.
    source : str or Table, optional
        Data source for the spectra. Options are:

        * ``"pppc4"`` (default): Cirelli et al. 2011.
        * ``"cosmixs"``: Cirelli et al. 2024.
        * A path to a custom file: Any format readable by `astropy.table.Table.read`
        (e.g., .ecsv, .fits, .csv, .dat).
        * An Astropy Table that read the desired path


        If a custom file path is provided, it must contain 'mDM' (mass of dark
        matter particle) and 'Log[10,x]' (energy) columns, plus columns named after
        the requested annihilation/decay channels (see the documentation).
    mapping_dict : dict, optional
        Mapping dictionary to map the columns of the custom source file to the expected
        column names. This is only needed if a file as a spectra source is provided
        and the column names in the file do not match the expected names. The dictionary
        should have the format {actual_column_name_in_file:expected_column_name}.
        An example of the expected columns can be found in the documentation.


    Examples
    --------
    This is how to instantiate a `DarkMatterDecaySpectralModel` model::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterDecaySpectralModel

        >>> channel = "b"
        >>> massDM = 5000*u.Unit("GeV")
        >>> jfactor = 3.41e19 * u.Unit("GeV cm-2")
        >>> modelDM = DarkMatterDecaySpectralModel(mass=massDM,
        channel=channel, jfactor=jfactor)

    References
    ----------
    `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist cookbook for dark
    matter indirect detection" <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    """

    LIFETIME_AGE_OF_UNIVERSE = 4.3e17 * u.Unit("s")
    """Use age of univserse as lifetime"""

    scale = Parameter(
        "scale",
        1,
        unit="",
        interp="log",
    )

    tag = ["DarkMatterDecaySpectralModel", "dm-decay"]

    def __init__(
        self,
        mass,
        channel,
        scale=scale.quantity,
        jfactor=1,
        z=0,
        source=None,
        mapping_dict=None,
    ):
        self.z = z
        self.mass = u.Quantity(mass)
        self.channel = channel
        self.jfactor = u.Quantity(jfactor)
        self.source = source
        self.mapping_dict = mapping_dict
        self.primary_flux = PrimaryFlux(
            self.mass / 2,
            channel=self.channel,
            source=self.source,
            mapping_dict=self.mapping_dict,
        )

        super().__init__(scale=scale)

    def evaluate(self, energy, scale):
        """Evaluate dark matter decay model."""
        flux = (
            scale
            * self.jfactor
            * self.primary_flux(energy=energy * (1 + self.z))
            / self.LIFETIME_AGE_OF_UNIVERSE
            / self.mass
            / (4 * np.pi)
        )
        return flux

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output=full_output)
        data["spectral"]["channel"] = self.channel
        data["spectral"]["mass"] = self.mass.to_string()
        data["spectral"]["jfactor"] = self.jfactor.to_string()
        data["spectral"]["z"] = self.z
        data["spectral"]["source"] = self.source
        data["spectral"]["mapping_dict"] = self.mapping_dict
        return data

    @classmethod
    def from_dict(cls, data):
        """Create spectral model from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with model data.

        Returns
        -------
        model : `DarkMatterDecaySpectralModel`
            Dark matter decay spectral model.
        """
        data = data["spectral"]
        data.pop("type")
        parameters = data.pop("parameters")
        scale = [p["value"] for p in parameters if p["name"] == "scale"][0]
        return cls(scale=scale, **data)
