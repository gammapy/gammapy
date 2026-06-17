# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""

import copy
import warnings
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
    "ContinuumPrimaryFlux",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "MonochromaticPrimaryFlux",
    "VIBPrimaryFlux",
    "BoxPrimaryFlux",
]


# ---------------------------------------------------------------------------
# Validator mixins
# ---------------------------------------------------------------------------


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


class _PrimaryFluxValidator:
    """Primary flux validator for DM spectral models.

    Concrete subclasses must implement ``_expected_primary_flux_mass``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        has_impl = any(
            "_expected_primary_flux_mass" in base.__dict__
            for base in cls.__mro__
            if base is not _PrimaryFluxValidator
        )
        if not has_impl:
            raise TypeError(
                f"Class '{cls.__name__}' must implement '_expected_primary_flux_mass'."
            )

    @property
    def _valid_primary_flux_types(self):
        return (
            ContinuumPrimaryFlux,
            MonochromaticPrimaryFlux,
            VIBPrimaryFlux,
            BoxPrimaryFlux,
        )

    @property
    def primary_flux(self):
        """Primary flux model."""
        return self._primary_flux

    @primary_flux.setter
    def primary_flux(self, value):
        if not isinstance(value, self._valid_primary_flux_types):
            raise TypeError(
                f"primary_flux must be one of "
                f"{[t.__name__ for t in self._valid_primary_flux_types]}, "
                f"got: {type(value).__name__}"
            )

        actual = u.Quantity(value.mDM)
        expected = self._expected_primary_flux_mass()
        rel_diff = abs((actual - expected) / expected).decompose().value
        if rel_diff > 0.01:
            warnings.warn(
                f"primary_flux.mDM ({actual}) does not match the expected "
                f"mass ({expected}). Results may be inconsistent.",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(value, ContinuumPrimaryFlux) and hasattr(self, "channel"):
            if value.channel != self.channel:
                warnings.warn(
                    f"primary_flux.channel ('{value.channel}') does not match "
                    f"the model channel ('{self.channel}'). "
                    "Make sure this is intentional.",
                    UserWarning,
                    stacklevel=2,
                )

        self._primary_flux = value

    def _expected_primary_flux_mass(self):
        """Expected mass for the primary flux.

        Returns mDM for annihilation, mDM/2 for decay.
        Must be implemented in every concrete subclass.
        """
        raise NotImplementedError(
            "_expected_primary_flux_mass must be implemented in subclasses."
        )


class _AstrophysicalFactorValidator:
    """Astrophysical factor validator for primary flux models."""

    @property
    def factor(self):
        """Astrophysical Factor."""
        return self._factor

    @factor.setter
    def factor(self, value):
        if u.Quantity(value).value <= 0:
            raise ValueError("The astrophysical factor must be strictly positive.")
        self._factor = u.Quantity(value)


class _RedshiftValidator:
    """Redshift validator for DM spectral models."""

    @property
    def z(self):
        """Source redshift (must be >= 0)."""
        return self._z

    @z.setter
    def z(self, value):
        try:
            z_val = float(value)
        except (TypeError, ValueError, u.UnitConversionError):
            raise TypeError(f"z must be a dimensionless scalar, got: {type(value)!r}")
        if z_val < 0:
            raise ValueError(f"Redshift z must be >= 0, got: {z_val}.")
        self._z = z_val


# ---------------------------------------------------------------------------
# Primary flux models
# ---------------------------------------------------------------------------


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

    Raises
    ------
    ValueError
        If ``n_gamma_photons`` is not 1 or 2.
    ValueError
        If ``n_gamma_photons=1`` and ``counterpart`` is ``None``.
    ValueError
        If the counterpart mass is such that the process is kinematically
        forbidden (i.e. ``2 * mDM <= counterpart_mass``).
    ValueError
        If ``counterpart`` is not in the internal registry and
        ``counterpart_mass`` is not provided.

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

    Raises
    ------
    ValueError
        If ``mPhi`` does not contain exactly 1 or 2 values.
    ValueError
        If any element of ``mPhi`` is not strictly positive.
    ValueError
        If the process is kinematically forbidden, i.e.
        ``mPhi1 + mPhi2 >= 2 * mDM``.

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


class ContinuumPrimaryFlux(TemplateNDSpectralModel):
    """Continuum gamma-ray spectrum from dark matter annihilation.

    Based on the precomputed tables of PPPC4 DM ID and CosmiXs. All
    available annihilation channels can be found in those tables. For a
    requested dark matter mass and channel, this class builds a
    `~gammapy.modeling.models.TemplateNDSpectralModel` over a 2D grid of
    (``Log[10,x]``, ``mDM``), enabling interpolation in dark matter mass
    while keeping ``mDM`` itself frozen as a model parameter.

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

        * ``"pppc4"`` (default): Cirelli et al. (2011, 2016) PPPC4DMID
          tables.
        * ``"cosmixs"``: Cirelli et al. (2024) / CosmiXs tables.
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

    Raises
    ------
    ValueError
        If ``source`` is not ``"pppc4"``, ``"cosmixs"``, or a valid
        existing file path.
    KeyError
        If a custom ``source`` file has an unsupported extension.
    ValueError
        If a custom ``source`` file is empty.
    ValueError
        If ``channel`` is not one of `allowed_channels`.
    ValueError
        If ``channel`` is listed as unavailable for the selected
        ``source``.
    KeyError
        If ``mapping_dict`` is provided for a custom file but does not
        contain the mandatory columns ``"mDM"`` and ``"Log[10,x]"``.
    ValueError
        If the requested channel's translated column name is not present
        in ``mapping_dict`` (when provided) or in the custom file's
        columns.
    FileNotFoundError
        If the resolved table file for the chosen ``source`` does not
        exist on disk (with a suggestion to run
        ``gammapy download datasets --src dark_matter_spectra``).
    ValueError
        If ``mDM`` (set via the `mDM` property) lies outside the mass
        range tabulated for the chosen ``source`` and ``channel``.

    Warns
    -----
    UserWarning
        If ``source`` is ``None``; defaults to ``"pppc4"``.

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
    .. [1] `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist
       cookbook for dark matter indirect detection"
       <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    .. [2] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [3] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for
       indirect dark matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [4] `Di Mauro et al. (2025), "Nailing down the theoretical
       uncertainties of Dbar spectrum produced from dark matter"
       <https://arxiv.org/abs/2411.04815>`_
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

        self.table_path = self._resolve_table_path()

        if self._source_type == "custom_file":
            self.table = Table.read(self.table_path, guess=True)
        else:
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
                f"The mass {_mDM} is out of bounds. "
                f"Please choose a mass between {min_mass} < mDM < {max_mass}."
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
            self._source_type = "predefined"

            warnings.warn(
                "No spectra source provided; PPPC4 will be used by default.",
                UserWarning,
            )
        elif isinstance(source, str):
            if source.lower() in ("pppc4", "cosmixs"):
                self._source = source.lower()
                self._source_type = "predefined"
            else:
                path = Path(make_path(source))
                if not (path.exists() and path.is_file()):
                    raise ValueError(
                        f"Invalid source: '{source}'. "
                        "Available options: 'pppc4', 'cosmixs', "
                        "or a valid file path."
                    )
                valid_extensions = {".dat", ".txt", ".csv", ".ecsv"}
                if path.suffix.lower() not in valid_extensions:
                    raise KeyError(
                        f"Source file extension '{path.suffix}' is not supported. "
                        f"Supported: {sorted(valid_extensions)}"
                    )
                if path.stat().st_size == 0:
                    raise ValueError("Source file is empty.")
                self._source = source
                self._source_type = "custom_file"
        else:
            raise TypeError(f"source must be a string, got: {type(source).__name__}")

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

        source_restrictions = self._unavailable_channels.get(self.source, {})
        if channel in source_restrictions:
            raise ValueError(
                f"The channel '{channel}' is not available in {self.source}: "
                f"please {source_restrictions[channel]}."
            )

        if self._source_type == "custom_file":
            channel_translation = self.channel_registry[channel]
            if self.mapping_dict is not None:
                if channel_translation not in self.mapping_dict.values():
                    raise ValueError(
                        f"The channel '{channel_translation}' is not present "
                        "in the provided mapping_dict. Please choose another "
                        "channel or check the mapping_dict provided."
                    )
            if channel_translation not in self.table.colnames:
                raise ValueError(
                    f"The channel '{channel_translation}' is not present in "
                    "the provided source file. Please choose another channel "
                    "or check the column names in the file."
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
                        f"Mandatory column '{key}' not found in mapping_dict. "
                        "Please check the mapping_dict or the column names "
                        "in the file."
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


PRIMARY_FLUX_REGISTRY = {
    cls.tag[0]: cls
    for cls in (
        ContinuumPrimaryFlux,
        MonochromaticPrimaryFlux,
        VIBPrimaryFlux,
        BoxPrimaryFlux,
    )
}


# ---------------------------------------------------------------------------
# Spectral models
# ---------------------------------------------------------------------------


class DarkMatterAnnihilationSpectralModel(
    _AstrophysicalFactorValidator,
    _RedshiftValidator,
    _PrimaryFluxValidator,
    SpectralModel,
):
    r"""Dark matter annihilation spectral model.

    Computes the differential gamma-ray flux expected from dark matter
    annihilation in a region with a given J-factor, combining the
    thermally averaged annihilation cross-section, the chosen primary
    particle-physics spectrum, and a fit-time normalization scale.

    The gamma-ray flux is computed as:

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

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass.
    channel : str
        Annihilation channel for the primary flux spectrum (used to build
        the default `ContinuumPrimaryFlux` if ``primary_flux`` is not
        given), e.g. ``"b"`` for bb̄. See
        `ContinuumPrimaryFlux.channel_registry` for available channels.
    scale : float, optional
        Dimensionless normalization parameter applied multiplicatively to
        the predicted flux, intended to be left free in spectral fits.
        Default is 1.
    factor : `~astropy.units.Quantity`, optional
        Astrophysical J-factor (integrated squared dark matter density
        along the line of sight and over the solid angle), needed when a
        `~gammapy.modeling.models.PointSpatialModel` is used for the
        spatial component. Default is 1 (dimensionless).
    z : float, optional
        Redshift of the source. The primary flux is evaluated at the
        redshifted energy ``energy * (1 + z)``. Default is 0.
    k : int, optional
        Dark matter particle type: ``2`` for Majorana particles (which are
        their own antiparticles, giving a factor of 2 in the annihilation
        rate) or ``4`` for Dirac particles. Default is 2.
    primary_flux : primary flux model, optional
        Primary photon spectrum per annihilation event. Must be an
        instance of `ContinuumPrimaryFlux`, `MonochromaticPrimaryFlux`,
        `VIBPrimaryFlux`, or `BoxPrimaryFlux`. If not provided, a default
        `ContinuumPrimaryFlux` is constructed using ``mDM`` and ``channel``.

    Raises
    ------
    ValueError
        If ``k`` is not 2 or 4.
    ValueError
        If ``z`` is negative or not a dimensionless scalar.
    ValueError
        If ``factor`` is not strictly positive.
    TypeError
        If ``primary_flux`` is provided but is not one of the supported
        primary flux types.

    Warns
    -----
    UserWarning
        If ``primary_flux.mDM`` does not match ``mDM`` within 1% relative
        tolerance.
    UserWarning
        If ``primary_flux`` is a `ContinuumPrimaryFlux` whose ``channel``
        does not match ``channel``.

    Examples
    --------
    This is how to instantiate a `DarkMatterAnnihilationSpectralModel`::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel

        >>> channel = "b"
        >>> mDM = 5000 * u.Unit("GeV")
        >>> factor = 3.41e19 * u.Unit("GeV2 cm-5")
        >>> modelDM = DarkMatterAnnihilationSpectralModel(
        ...     mDM=mDM, channel=channel, factor=factor
        ... )

    References
    ----------
    `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist cookbook
    for dark matter indirect detection"
    <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    """

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section."""

    scale = Parameter("scale", 1, unit="", interp="log")
    tag = ["DarkMatterAnnihilationSpectralModel", "dm-annihilation"]

    def __init__(
        self,
        mDM,
        channel,
        scale=scale.quantity,
        factor=1,
        z=0,
        k=2,
        primary_flux=None,
    ):
        self.k = k
        self.z = z
        self.mDM = u.Quantity(mDM)
        self.channel = channel
        self.factor = u.Quantity(factor)
        self.primary_flux = primary_flux or ContinuumPrimaryFlux(
            self._expected_primary_flux_mass(), channel=self.channel
        )
        super().__init__(scale=scale)

    def evaluate(self, energy, scale):
        """Evaluate the dark matter annihilation differential flux.

        Computes
        ``flux = scale * factor * THERMAL_RELIC_CROSS_SECTION
        * primary_flux(energy * (1 + z)) / k / mDM**2 / (4 * pi)``.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the flux.
        scale : float
            Current value of the ``scale`` normalization parameter.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential gamma-ray flux per unit energy.
        """
        flux = (
            scale
            * self.factor
            * self.THERMAL_RELIC_CROSS_SECTION
            * self.primary_flux(energy=energy * (1 + self.z))
            / self.k
            / self.mDM
            / self.mDM
            / (4 * np.pi)
        )
        return flux

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Extends the base `~gammapy.modeling.models.SpectralModel.to_dict`
        output with ``channel``, ``mDM``, ``factor``, ``z``, ``k``, and the
        serialized ``primary_flux`` (via its own `to_dict`).

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
        data["spectral"]["channel"] = self.channel
        data["spectral"]["mDM"] = self.mDM.to_string()
        data["spectral"]["factor"] = self.factor.to_string()
        data["spectral"]["z"] = self.z
        data["spectral"]["k"] = self.k
        data["spectral"]["primary_flux"] = self.primary_flux.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        """Construct a `DarkMatterAnnihilationSpectralModel` from a dictionary.

        Reconstructs the ``primary_flux`` sub-model using the registry of
        known primary flux types, extracts the ``scale`` parameter value
        from the serialized parameter list, and passes the remaining
        fields through to the constructor.

        Parameters
        ----------
        data : dict
            Dictionary with a top-level ``"spectral"`` key, as produced by
            `to_dict`, containing ``mDM``, ``channel``, ``factor``, ``z``,
            ``k``, ``primary_flux``, and ``parameters`` (including
            ``scale``).

        Returns
        -------
        model : `DarkMatterAnnihilationSpectralModel`
            New instance reconstructed from ``data``.

        Raises
        ------
        ValueError
            If the ``primary_flux`` type recorded in ``data`` is not found
            in the primary flux registry.
        """
        data = copy.deepcopy(data["spectral"])
        data.pop("type")

        pf_data = data.pop("primary_flux")
        pf_cls = PRIMARY_FLUX_REGISTRY.get(pf_data["type"])
        if pf_cls is None:
            raise ValueError(
                f"Unknown primary_flux type: '{pf_data['type']}'. "
                f"Available: {list(PRIMARY_FLUX_REGISTRY.keys())}"
            )
        primary_flux = pf_cls.from_dict(pf_data)

        parameters = data.pop("parameters")
        scale = next(p["value"] for p in parameters if p["name"] == "scale")

        return cls(scale=scale, primary_flux=primary_flux, **data)

    @property
    def k(self):
        """DM particle type (2: Majorana, 4: Dirac)."""
        return self._k

    @k.setter
    def k(self, value):
        if value not in (2, 4):
            raise ValueError(f"k must be 2 (Majorana) or 4 (Dirac), got: {value}.")
        self._k = value

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
        return self.mDM


class DarkMatterDecaySpectralModel(
    _AstrophysicalFactorValidator,
    _RedshiftValidator,
    _PrimaryFluxValidator,
    SpectralModel,
):
    r"""Dark matter decay spectral model.

    Computes the differential gamma-ray flux expected from the decay of
    dark matter particles in a region with a given D-factor, combining the
    decay lifetime, the chosen primary particle-physics spectrum, and a
    fit-time normalization scale.

    The gamma-ray flux is computed as:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\Gamma}{4\pi m_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times J(\Delta\Omega)

    where :math:`\Gamma = 1/\tau` is the decay rate (inverse lifetime),
    :math:`m_{\mathrm{DM}}` is the dark matter mass,
    :math:`\mathrm dN/\mathrm dE` is the primary photon spectrum per decay,
    and :math:`J(\Delta\Omega)` is the astrophysical D-factor (here denoted
    generically as the astrophysical factor).

    Parameters
    ----------
    mDM : `~astropy.units.Quantity`
        Dark matter particle mass.
    channel : str
        Decay channel for the primary flux spectrum (used to build the
        default `ContinuumPrimaryFlux` if ``primary_flux`` is not given),
        e.g. ``"b"`` for bb̄. See `ContinuumPrimaryFlux.channel_registry`
        for available channels.
    scale : float, optional
        Dimensionless normalization parameter applied multiplicatively to
        the predicted flux, intended to be left free in spectral fits.
        Default is 1.
    lifetime : `~astropy.units.Quantity`, optional
        Characteristic dark matter decay lifetime :math:`\tau`, in units
        convertible to seconds, with a minimum allowed value of
        :math:`10^{10}` s. Default is the age of the universe
        (:math:`4.3 \times 10^{17}` s).
    factor : `~astropy.units.Quantity`, optional
        Astrophysical D-factor (integrated dark matter density along the
        line of sight and over the solid angle), needed when a
        `~gammapy.modeling.models.PointSpatialModel` is used for the
        spatial component. Default is 1 (dimensionless).
    z : float, optional
        Redshift of the source. The primary flux is evaluated at the
        redshifted energy ``energy * (1 + z)``. Default is 0.
    primary_flux : primary flux model, optional
        Primary photon spectrum per decay event. Must be an instance of
        `ContinuumPrimaryFlux`, `MonochromaticPrimaryFlux`,
        `VIBPrimaryFlux`, or `BoxPrimaryFlux`. If not provided, a default
        `ContinuumPrimaryFlux` is constructed using ``mDM / 2`` (the energy
        scale relevant for two-body decay products) and ``channel``.

    Raises
    ------
    ValueError
        If ``z`` is negative or not a dimensionless scalar.
    ValueError
        If ``factor`` is not strictly positive.
    TypeError
        If ``primary_flux`` is provided but is not one of the supported
        primary flux types.

    Warns
    -----
    UserWarning
        If ``primary_flux.mDM`` does not match the expected mass
        (``mDM / 2``) within 1% relative tolerance.
    UserWarning
        If ``primary_flux`` is a `ContinuumPrimaryFlux` whose ``channel``
        does not match ``channel``.

    Examples
    --------
    This is how to instantiate a `DarkMatterDecaySpectralModel`::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterDecaySpectralModel

        >>> channel = "b"
        >>> mDM = 5000 * u.Unit("GeV")
        >>> factor = 3.41e19 * u.Unit("GeV cm-2")
        >>> modelDM = DarkMatterDecaySpectralModel(
        ...     mDM=mDM, channel=channel, factor=factor
        ... )

    References
    ----------
    `Marco et al. (2011), "PPPC 4 DM ID: a poor particle physicist cookbook
    for dark matter indirect detection"
    <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_

    """

    scale = Parameter("scale", 1, unit="", interp="log")
    lifetime = Parameter("lifetime", 4.3e17, unit="s", min=1e10)

    tag = ["DarkMatterDecaySpectralModel", "dm-decay"]

    def __init__(
        self,
        mDM,
        channel,
        scale=scale.quantity,
        lifetime=lifetime.quantity,
        factor=1,
        z=0,
        primary_flux=None,
    ):
        self.z = z
        self.mDM = u.Quantity(mDM)
        self.channel = channel
        self.factor = u.Quantity(factor)
        self.primary_flux = primary_flux or ContinuumPrimaryFlux(
            self._expected_primary_flux_mass(), channel=self.channel
        )
        super().__init__(scale=scale, lifetime=lifetime)

    def evaluate(self, energy, scale, lifetime):
        """Evaluate the dark matter decay differential flux.

        Computes
        ``flux = scale * factor * primary_flux(energy * (1 + z))
        / lifetime / mDM / (4 * pi)``.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy values (array-like) at which to evaluate the flux.
        scale : float
            Current value of the ``scale`` normalization parameter.
        lifetime : `~astropy.units.Quantity`
            Current value of the ``lifetime`` parameter.

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential gamma-ray flux per unit energy.
        """
        flux = (
            scale
            * self.factor
            * self.primary_flux(energy=energy * (1 + self.z))
            / lifetime
            / self.mDM
            / (4 * np.pi)
        )
        return flux

    def to_dict(self, full_output=False):
        """Serialize the model to a dictionary.

        Extends the base `~gammapy.modeling.models.SpectralModel.to_dict`
        output with ``channel``, ``mDM``, ``factor``, ``z``, and the
        serialized ``primary_flux`` (via its own `to_dict`). The ``scale``
        and ``lifetime`` parameters are included via the base class's
        ``parameters`` serialization.

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
        data["spectral"]["channel"] = self.channel
        data["spectral"]["mDM"] = self.mDM.to_string()
        data["spectral"]["factor"] = self.factor.to_string()
        data["spectral"]["z"] = self.z
        data["spectral"]["primary_flux"] = self.primary_flux.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        """Construct a `DarkMatterDecaySpectralModel` from a dictionary.

        Reconstructs the ``primary_flux`` sub-model using the registry of
        known primary flux types, extracts the ``scale`` and ``lifetime``
        parameter values (with units) from the serialized parameter list,
        and passes the remaining fields through to the constructor.

        Parameters
        ----------
        data : dict
            Dictionary with a top-level ``"spectral"`` key, as produced by
            `to_dict`, containing ``mDM``, ``channel``, ``factor``, ``z``,
            ``primary_flux``, and ``parameters`` (including ``scale`` and
            ``lifetime``).

        Returns
        -------
        model : `DarkMatterDecaySpectralModel`
            New instance reconstructed from ``data``.

        Raises
        ------
        ValueError
            If the ``primary_flux`` type recorded in ``data`` is not found
            in the primary flux registry.
        """
        data = copy.deepcopy(data["spectral"])
        data.pop("type")

        pf_data = data.pop("primary_flux")
        pf_cls = PRIMARY_FLUX_REGISTRY.get(pf_data["type"])
        if pf_cls is None:
            raise ValueError(
                f"Unknown primary_flux type: '{pf_data['type']}'. "
                f"Available: {list(PRIMARY_FLUX_REGISTRY.keys())}"
            )
        primary_flux = pf_cls.from_dict(pf_data)

        parameters = data.pop("parameters")
        scale = next(p["value"] for p in parameters if p["name"] == "scale")
        lifetime_entry = next(p for p in parameters if p["name"] == "lifetime")
        lifetime = u.Quantity(lifetime_entry["value"], lifetime_entry.get("unit", "s"))

        return cls(scale=scale, lifetime=lifetime, primary_flux=primary_flux, **data)

    def _expected_primary_flux_mass(self):
        """Expected primary flux mass for decay.

        Returns
        -------
        mass : `~astropy.units.Quantity`
            Equal to ``mDM / 2``, since a decaying dark matter particle of
            mass ``mDM`` produces two-body final states each carrying
            roughly half the rest energy, and the primary flux tables are
            indexed by this per-product mass scale.
        """
        return self.mDM / 2
