# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spectra."""

import warnings
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.modeling import Parameter
from gammapy.modeling.models import SpectralModel, TemplateNDSpectralModel
from gammapy.modeling.models.prior import (
    GaussianPrior,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_map_columns

__all__ = [
    "PrimaryFlux",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
]
log = logging.getLogger(__name__)


class _SigmaValidator:
    """Mixin that validates and stores the uncertainty ``sigma``."""

    @property
    def sigma(self):
        """Uncertainty on log10(factor) [dex]."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value is None:
            self._sigma = 0.0
            return
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"The sigma must be a number or None, got {type(value)}")
        if value < 0:
            raise ValueError(f"The sigma must be non-negative, got {value}")
        self._sigma = float(value)


class PrimaryFlux(TemplateNDSpectralModel):
    """
    DM-annihilation gamma-ray spectra.

    Based on the precomputed models of PPPC4 DM ID by [Cirelli2016]_ and CosmiXs by [Arina2024]_, [DiMauro2025]_.
    All available annihilation channels can be found there. The dark matter mass will be set to the nearest available value. The spectra will be available as `~gammapy.modeling.models.TemplateNDSpectralModel` for a chosen dark matter mass and annihilation channel.
    Using a `~gammapy.modeling.models.TemplateNDSpectralModel` allows the interpolation between different dark matter masses.

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
    .. [Cirelli2016] `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist
       Cookbook for Dark Matter Indirect Detection"
       <http://www.marcocirelli.net/PPPC4DMID.html>`_
    .. [Arina2024] `Arina et al. (2024), "CosmiXs: Cosmic messenger spectra for
       indirect dark matter searches" <https://arxiv.org/abs/2312.01153>`_
    .. [DiMauro2025] `Di Mauro et al. (2025), "Nailing down the theoretical
       uncertainties of Dbar spectrum produced from dark matter"
       <https://arxiv.org/abs/2411.04815>`_
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
        "dNdLog10x[u]": "u",  # Does not exist explicitly on PPPC4, equivalent to q
        "dNdLog10x[d]": "d",  # Does not exist explicitly on PPPC4, equivalent to q
        "dNdLog10x[s]": "s",  # Does not exist explicitly on PPPC4, equivalent to q
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
        "dNdLog10x[aZ]": None,  # Does not exist on PPPC4
        "dNdLog10x[HZ]": None,  # Does not exist on PPPC4
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
            if source is None:
                source = "pppc4"
                warnings.warn(
                    "Since no spectra source has been chosen, PPPC4 will be used by "
                    "default.",
                    UserWarning,
                )
            self.source = source.lower()

            base_data_path = "$GAMMAPY_DATA/dark_matter_spectra"
            if self.source == "pppc4":
                table_filename = f"{base_data_path}/PPPC4DMID/AtProduction_gammas.dat"
            elif self.source == "cosmixs":
                table_filename = f"{base_data_path}/cosmixs/AtProduction-Gamma.dat"
            else:
                raise ValueError(
                    "\n\nData source is not valid, please choose between PPPC4 or cosmixs\n"
                )
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
                f"The mass {_mDM} is out of the bounds of the model. Please choose "
                f"a mass between {min_mass} < `mDM` < {max_mass}"
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
                        f"\n\nThe channel {channel} is not available in PPPC4, please "
                        "choose another channel or use CosmiXs (cosmixs) as source\n"
                    )
                elif channel in ("d", "u", "s"):
                    raise ValueError(
                        f"\n\nThe channel {channel} is not available in PPPC4, please "
                        "choose the equivalent channel q or use CosmiXs (cosmixs)\
                              as source\n"
                    )

            elif self.source == "cosmixs":
                if channel in ("V->e", "V->mu", "V->tau"):
                    raise ValueError(
                        f"\n\nThe channel {channel} is not available in CosmiXs, "
                        "please choose another channel or use PPPC4 as source\n"
                    )
                elif channel == "q":
                    raise ValueError(
                        "\n\nThe channel q is not available in cosmixs, please "
                        "choose an equivalent channel such as d, u or s or "
                        "use PPPC4 as source\n"
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


class DarkMatterAnnihilationSpectralModel(SpectralModel, _SigmaValidator):
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
        Annihilation channel for `~gammapy.astro.darkmatter.PrimaryFlux`,
        e.g. "b" for "bbar". See `PrimaryFlux.channel_registry` for more.
    scale : float
        Scale parameter for model fitting.
    jfactor : `~astropy.units.Quantity`, optional
        Integrated J-Factor over the region of interest. Default is
        ``1 GeV² cm⁻⁵``, which makes the model return the flux per unit
        J-factor. This allows the user to multiply the result externally
        by a J-factor map to produce spatial flux maps::

            flux_map = jfact * model.integral(...) / model.jfactor

        Pass a scalar J-factor to include it directly in the model,
        e.g. for spectral fitting with nuisance parameters.
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
    sigma : float or None, optional
        Statistical and systematic uncertainty on log10(J) in dex. Default is None (no prior).
        Passing ``sigma=0.0`` (or ``None``) is equivalent: no nuisance prior is attached and ``log10_jfactor`` remains frozen at the observed value. A prior is only created when at least one sigma is strictly positive.

    Examples
    --------
    This is how to instantiate a `DarkMatterAnnihilationSpectralModel` model::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel

        >>> channel = "b"
        >>> massDM = 5000 * u.Unit("GeV")
        >>> jfactor = 3.41e19 * u.Unit("GeV2 cm-5")
        >>> modelDM = DarkMatterAnnihilationSpectralModel(mass=massDM, channel=channel,jfactor=jfactor)  # noqa: E501

    References
    ----------
    `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist Cookbook
    for Dark Matter Indirect Detection" <http://www.marcocirelli.net/PPPC4DMID.html>`_
    `Ackermann et al. (2015), "Searching for Dark Matter Annihilation from Milky Way
    Dwarf Spheroidal Galaxies with Six Years of Fermi-LAT Data"
    <https://doi.org/10.1103/PhysRevLett.115.231301>`_
    """

    THERMAL_RELIC_CROSS_SECTION = 3e-26 * u.Unit("cm3 s-1")
    """Thermally averaged annihilation cross-section"""

    scale = Parameter("scale", 1, unit="", interp="log")
    log10_jfactor = Parameter("log10_jfactor", 1.0, unit="", frozen=True, prior=None)

    tag = ["DarkMatterAnnihilationSpectralModel", "dm-annihilation"]

    def __init__(
        self,
        mass,
        channel,
        scale=scale.quantity,
        jfactor=1 * u.Unit("GeV2 cm-5"),
        z=0,
        k=2,
        source="pppc4",
        mapping_dict=None,
        sigma=None,
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

        self.jfactor = u.Quantity(jfactor)
        jfactor_val = self.jfactor.to("GeV2 cm-5").value
        if np.ndim(jfactor_val) > 0:
            raise ValueError(
                "jfactor must be a scalar Quantity. "
                "Pass the J-factor integrated over the region of interest, not a map. "
                "For spatial flux maps, use the default jfactor and multiply the "
                "result externally: \
                    flux_map = jfact * model.integral(...) / model.jfactor"
            )
        self._log10_j_obs = np.log10(float(jfactor_val))

        super().__init__(scale=scale, log10_jfactor=self._log10_j_obs)

        self.sigma = sigma
        if self._sigma > 0.0:
            self._create_prior()

    def _create_prior(self):
        """Create and attach a Gaussian prior on ``log10_jfactor``.

        Equivalent to a log-normal prior on the J-factor in linear space.
        The Jacobian of the log10(J) -> J transformation is absorbed into
        the parametrisation, so a Gaussian in log10 space is the correct
        implementation.
        """
        self.log10_jfactor.frozen = False
        prior = GaussianPrior(mu=self._log10_j_obs, sigma=self._sigma)
        self.log10_jfactor.prior = prior
        self.log10_jfactor.min = self._log10_j_obs - 5 * self._sigma
        self.log10_jfactor.max = self._log10_j_obs + 5 * self._sigma

    def evaluate(self, energy, scale, log10_jfactor=None):
        """Evaluate dark matter annihilation model."""
        if log10_jfactor is None:
            log10_jfactor = self._log10_j_obs
        jfactor = 10**log10_jfactor * u.Unit("GeV2 cm-5")

        flux = (
            scale
            * jfactor
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
        data["spectral"]["sigma"] = self._sigma
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
        data.setdefault("sigma", 0.0)
        return cls(scale=scale, **data)


class DarkMatterDecaySpectralModel(SpectralModel, _SigmaValidator):
    r"""Dark matter decay spectral model.

    The gamma-ray flux is computed as follows:

    .. math::
        \frac{\mathrm d \phi}{\mathrm d E} =
        \frac{\Gamma}{4\pi m_{\mathrm{DM}}}
        \frac{\mathrm d N}{\mathrm dE} \times D(\Delta\Omega)

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Dark matter mass.
    channel : str
        Decay channel for `~gammapy.astro.darkmatter.PrimaryFlux`, e.g. "b" for "bbar".
        See `PrimaryFlux.channel_registry` for more.
    scale : float
        Scale parameter for model fitting.
    jfactor : `~astropy.units.Quantity`, optional
        Integrated D-Factor over the region of interest. Default is
        ``1 GeV cm⁻²``, which makes the model return the flux per unit
        D-factor. This allows the user to multiply the result externally
        by a D-factor map to produce spatial flux maps::

            flux_map = jfact_decay * model.integral(...) / model.jfactor

        Pass a scalar D-factor to include it directly in the model,
        e.g. for spectral fitting with nuisance parameters.
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
    sigma : float or None, optional
        Statistical and systematic uncertainty on log10(D) in dex. Default is None (no prior).
        Passing ``sigma=0.0`` (or ``None``) is equivalent: no nuisance prior is attached and ``log10_jfactor`` remains frozen at the observed value. A prior is only created when at least one sigma is strictly positive.


    Examples
    --------
    This is how to instantiate a `DarkMatterDecaySpectralModel` model::

        >>> import astropy.units as u
        >>> from gammapy.astro.darkmatter import DarkMatterDecaySpectralModel

        >>> channel = "b"
        >>> massDM = 5000 * u.Unit("GeV")
        >>> jfactor = 3.41e19 * u.Unit("GeV cm-2")
        >>> modelDM = DarkMatterDecaySpectralModel(mass=massDM, channel=channel, jfactor=jfactor)  # noqa: E501

    References
    ----------
    `Cirelli et al. (2016), "PPPC 4 DM ID: A Poor Particle Physicist Cookbook
    for Dark Matter Indirect Detection" <http://www.marcocirelli.net/PPPC4DMID.html>`_
    `Ackermann et al. (2015), "Searching for Dark Matter Annihilation from Milky Way
    Dwarf Spheroidal Galaxies with Six Years of Fermi-LAT Data"
    <https://doi.org/10.1103/PhysRevLett.115.231301>`_
    """

    LIFETIME_AGE_OF_UNIVERSE = 4.3e17 * u.Unit("s")
    """Use age of universe as lifetime"""

    scale = Parameter("scale", 1, unit="", interp="log")
    log10_jfactor = Parameter("log10_jfactor", 1.0, unit="", frozen=True, prior=None)

    tag = ["DarkMatterDecaySpectralModel", "dm-decay"]

    def __init__(
        self,
        mass,
        channel,
        scale=scale.quantity,
        jfactor=1 * u.Unit("GeV cm-2"),
        z=0,
        source="pppc4",
        mapping_dict=None,
        sigma=None,
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

        self.jfactor = u.Quantity(jfactor)
        jfactor_val = self.jfactor.to("GeV cm-2").value
        if np.ndim(jfactor_val) > 0:
            raise ValueError(
                "jfactor must be a scalar Quantity. "
                "Pass the D-factor integrated over the region of interest, not a map. "
                "For spatial flux maps, use the default jfactor and multiply the "
                "result externally:\
                     flux_map = jfact * model.integral(...) / model.jfactor"
            )
        self._log10_j_obs = np.log10(float(jfactor_val))

        super().__init__(scale=scale, log10_jfactor=self._log10_j_obs)

        self.sigma = sigma
        if self._sigma > 0.0:
            self._create_prior()

    def _create_prior(self):
        """Create and attach a Gaussian prior on ``log10_jfactor``.

        Equivalent to a log-normal prior on the J-factor in linear space.
        The Jacobian of the log10(J) -> J transformation is absorbed into
        the parametrisation, so a Gaussian in log10 space is the correct
        implementation.
        """
        self.log10_jfactor.frozen = False
        prior = GaussianPrior(mu=self._log10_j_obs, sigma=self._sigma)
        self.log10_jfactor.prior = prior
        self.log10_jfactor.min = self._log10_j_obs - 5 * self._sigma
        self.log10_jfactor.max = self._log10_j_obs + 5 * self._sigma

    def evaluate(self, energy, scale, log10_jfactor=None):
        """Evaluate dark matter decay model."""
        if log10_jfactor is None:
            log10_jfactor = self._log10_j_obs
        jfactor = 10**log10_jfactor * u.Unit("GeV cm-2")

        flux = (
            scale
            * jfactor
            * self.primary_flux(energy=energy * (1 + self.z))
            / self.LIFETIME_AGE_OF_UNIVERSE
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
        data["spectral"]["source"] = self.source
        data["spectral"]["sigma"] = self._sigma
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
        data.setdefault("sigma", 0.0)
        return cls(scale=scale, **data)
