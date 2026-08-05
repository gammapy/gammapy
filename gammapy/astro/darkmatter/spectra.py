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
from gammapy.modeling.models import SpectralModel, TemplateNDSpectralModel
from gammapy.utils.deprecation import (
    GammapyDeprecationWarning,
    deprecated,
    deprecated_renamed_argument,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_map_columns

__all__ = [
    "ContinuumPrimaryFlux",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "PrimaryFlux",
]
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validator mixins
# ---------------------------------------------------------------------------


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
        return (ContinuumPrimaryFlux,)

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

        if hasattr(self, "channel") and value.channel != self.channel:
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
    .. [1] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [2] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for
       indirect dark matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [3] `Di Mauro et al. (2025), "Nailing down the theoretical
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


PRIMARY_FLUX_REGISTRY = {cls.tag[0]: cls for cls in (ContinuumPrimaryFlux, PrimaryFlux)}

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
        instance of `ContinuumPrimaryFlux`. If not provided, a default
        `ContinuumPrimaryFlux` is constructed using ``mDM`` and ``channel``.

    Warns
    -----
    UserWarning
        If ``primary_flux.mDM`` does not match ``mDM`` within 1% relative
        tolerance.
    UserWarning
        If ``primary_flux`` is a `ContinuumPrimaryFlux` whose ``channel``
        does not match ``channel``.
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
    This is how to instantiate a `DarkMatterAnnihilationSpectralModel`::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel

        >>> channel = "b"
        >>> mDM = 5000*u.Unit("GeV")
        >>> factor = 3.41e19 * u.Unit("GeV2 cm-5")
        >>> modelDM = DarkMatterAnnihilationSpectralModel(mDM=mDM,
          channel=channel, factor=factor)  # noqa: E501

    References
    ----------
    .. [1] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    """

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section."""

    scale = Parameter("scale", 1, unit="", interp="log")
    tag = ["DarkMatterAnnihilationSpectralModel", "dm-annihilation"]

    @deprecated_renamed_argument("mass", "mDM", "2.2")
    @deprecated_renamed_argument("jfactor", "factor", "2.2")
    def __init__(
        self,
        mDM,
        channel,
        scale=scale.quantity,
        factor=1,
        z=0,
        k=2,
        primary_flux=None,
        source=None,
        mapping_dict=None,
    ):
        self.k = k
        self.z = z
        self.mDM = u.Quantity(mDM)
        self.channel = channel
        self.factor = u.Quantity(factor)
        self.primary_flux = primary_flux or ContinuumPrimaryFlux(
            self._expected_primary_flux_mass(), channel=self.channel
        )
        self.source = source
        self.mapping_dict = mapping_dict
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
        data["spectral"]["source"] = self.source
        data["spectral"]["mapping_dict"] = self.mapping_dict
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
        """
        data = copy.deepcopy(data["spectral"])
        data.pop("type")

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
            # Old format
            primary_flux = ContinuumPrimaryFlux(
                data.get("mDM", data.get("mass")),
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
        `ContinuumPrimaryFlux`. If not provided, a default
        `ContinuumPrimaryFlux` is constructed using ``mDM / 2`` (the energy
        scale relevant for two-body decay products) and ``channel``.

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
        >>> mDM = 5000*u.Unit("GeV")
        >>> factor = 3.41e19 * u.Unit("GeV cm-2")
        >>> modelDM = DarkMatterDecaySpectralModel(mDM=mDM,
        channel=channel, factor=factor)

    References
    ----------
    .. [1] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    """

    LIFETIME_AGE_OF_UNIVERSE = 4.3e17 * u.Unit("s")
    """Use age of univserse as lifetime"""

    scale = Parameter("scale", 1, unit="", interp="log")

    tag = ["DarkMatterDecaySpectralModel", "dm-decay"]

    @deprecated_renamed_argument("mass", "mDM", "2.2")
    @deprecated_renamed_argument("jfactor", "factor", "2.2")
    def __init__(
        self,
        mDM,
        channel,
        scale=scale.quantity,
        factor=1,
        z=0,
        primary_flux=None,
        source=None,
        mapping_dict=None,
    ):
        self.z = z
        self.mDM = u.Quantity(mDM)
        self.channel = channel
        self.factor = u.Quantity(factor)
        self.primary_flux = primary_flux or ContinuumPrimaryFlux(
            self._expected_primary_flux_mass(), channel=self.channel
        )
        self.source = source
        self.mapping_dict = mapping_dict
        super().__init__(scale=scale)

    def evaluate(self, energy, scale):
        """Evaluate the dark matter decay differential flux.

        Computes
        ``flux = scale * factor * primary_flux(energy * (1 + z))
        / LIFETIME_AGE_OF_UNIVERSE / mDM / (4 * pi)``.

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
            * self.primary_flux(energy=energy * (1 + self.z))
            / self.LIFETIME_AGE_OF_UNIVERSE
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
        data["spectral"]["source"] = self.source
        data["spectral"]["mapping_dict"] = self.mapping_dict
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
        """
        data = copy.deepcopy(data["spectral"])
        data.pop("type")

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
            # Old format
            primary_flux = ContinuumPrimaryFlux(
                data.get("mDM", data.get("mass")),
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

        parameters = data.pop("parameters")
        scale = next(p["value"] for p in parameters if p["name"] == "scale")

        return cls(scale=scale, primary_flux=primary_flux, **data)

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
