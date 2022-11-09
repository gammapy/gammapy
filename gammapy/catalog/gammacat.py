# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammacat open TeV source catalog.

https://github.com/gammapy/gamma-cat
"""
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Model, SkyModel
from gammapy.utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject, format_flux_points_table

__all__ = ["SourceCatalogGammaCat", "SourceCatalogObjectGammaCat"]

log = logging.getLogger(__name__)


class SourceCatalogObjectGammaCat(SourceCatalogObject):
    """One object from the gamma-cat source catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalogGammaCat`.
    """

    _source_name_key = "common_name"

    def __str__(self):
        return self.info()

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        return FluxPoints.from_table(
            table=self.flux_points_table,
            reference_model=self.sky_model(),
            format="gadf-sed",
        )

    def info(self, info="all"):
        """Info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position, 'model'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,model"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "model" in ops:
            ss += self._info_morph()
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        return (
            f"\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based): {self.row_index}\n"
            f"Common name: {self.name}\n"
            f"Gamma names: {d.gamma_names}\n"
            f"Fermi names: {d.fermi_names}\n"
            f"Other names: {d.other_names}\n"
            f"Location: {d.where}\n"
            f"Class: {d.classes}\n\n"
            f"TeVCat ID: {d.tevcat_id}\n"
            f"TeVCat 2 ID: {d.tevcat2_id}\n"
            f"TeVCat name: {d.tevcat_name}\n\n"
            f"TGeVCat ID: {d.tgevcat_id}\n"
            f"TGeVCat name: {d.tgevcat_name}\n\n"
            f"Discoverer: {d.discoverer}\n"
            f"Discovery date: {d.discovery_date}\n"
            f"Seen by: {d.seen_by}\n"
            f"Reference: {d.reference_id}\n"
        )

    def _info_position(self):
        """Print position info."""
        d = self.data
        return (
            f"\n*** Position info ***\n\n"
            f"SIMBAD:\n"
            f"RA: {d.ra:.3f}\n"
            f"DEC: {d.dec:.3f}\n"
            f"GLON: {d.glon:.3f}\n"
            f"GLAT: {d.glat:.3f}\n"
            f"\nMeasurement:\n"
            f"RA: {d.pos_ra:.3f}\n"
            f"DEC: {d.pos_dec:.3f}\n"
            f"GLON: {d.pos_glon:.3f}\n"
            f"GLAT: {d.pos_glat:.3f}\n"
            f"Position error: {d.pos_err:.3f}\n"
        )

    def _info_morph(self):
        """Print morphology info."""
        d = self.data
        return (
            f"\n*** Morphology info ***\n\n"
            f"Morphology model type: {d.morph_type}\n"
            f"Sigma: {d.morph_sigma:.3f}\n"
            f"Sigma error: {d.morph_sigma_err:.3f}\n"
            f"Sigma2: {d.morph_sigma2:.3f}\n"
            f"Sigma2 error: {d.morph_sigma2_err:.3f}\n"
            f"Position angle: {d.morph_pa:.3f}\n"
            f"Position angle error: {d.morph_pa_err:.3f}\n"
            f"Position angle frame: {d.morph_pa_frame}\n"
        )

    def _info_spectral_fit(self):
        """Print spectral info."""
        d = self.data
        ss = "\n*** Spectral info ***\n\n"
        ss += f"Significance: {d.significance:.3f}\n"
        ss += f"Livetime: {d.livetime:.3f}\n"

        spec_type = d["spec_type"]
        ss += f"\nSpectrum type: {spec_type}\n"
        if spec_type == "pl":
            ss += f"norm: {d.spec_pl_norm:.3} +- {d.spec_pl_norm_err:.3} (stat) +- {d.spec_pl_norm_err_sys:.3} (sys) cm-2 s-1 TeV-1\n"  # noqa: E501
            ss += f"index: {d.spec_pl_index:.3} +- {d.spec_pl_index_err:.3} (stat) +- {d.spec_pl_index_err_sys:.3} (sys)\n"  # noqa: E501
            ss += f"reference: {d.spec_pl_e_ref:.3}\n"
        elif spec_type == "pl2":
            ss += f"flux: {d.spec_pl2_flux.value:.3} +- {d.spec_pl2_flux_err.value:.3} (stat) +- {d.spec_pl2_flux_err_sys.value:.3} (sys) cm-2 s-1\n"  # noqa: E501
            ss += f"index: {d.spec_pl2_index:.3} +- {d.spec_pl2_index_err:.3} (stat) +- {d.spec_pl2_index_err_sys:.3} (sys)\n"  # noqa: E501
            ss += f"e_min: {d.spec_pl2_e_min:.3}\n"
            ss += f"e_max: {d.spec_pl2_e_max:.3}\n"
        elif spec_type == "ecpl":
            ss += f"norm: {d.spec_ecpl_norm.value:.3g} +- {d.spec_ecpl_norm_err.value:.3g} (stat) +- {d.spec_ecpl_norm_err_sys.value:.03g} (sys) cm-2 s-1 TeV-1\n"  # noqa: E501
            ss += f"index: {d.spec_ecpl_index:.3} +- {d.spec_ecpl_index_err:.3} (stat) +- {d.spec_ecpl_index_err_sys:.3} (sys)\n"  # noqa: E501
            ss += f"e_cut: {d.spec_ecpl_e_cut.value:.3} +- {d.spec_ecpl_e_cut_err.value:.3} (stat) +- {d.spec_ecpl_e_cut_err_sys.value:.3} (sys) TeV\n"  # noqa: E501
            ss += f"reference: {d.spec_ecpl_e_ref:.3}\n"
        elif spec_type == "none":
            pass
        else:
            raise ValueError(f"Invalid spec_type: {spec_type}")

        ss += f"\nEnergy range: ({d.spec_erange_min.value:.3}, {d.spec_erange_max.value:.3}) TeV\n"
        ss += f"theta: {d.spec_theta:.3}\n"

        ss += "\n\nDerived fluxes:\n"

        ss += f"Spectral model norm (1 TeV): {d.spec_dnde_1TeV:.3} +- {d.spec_dnde_1TeV_err:.3} (stat) cm-2 s-1 TeV-1\n"  # noqa: E501
        ss += f"Integrated flux (>1 TeV): {d.spec_flux_1TeV.value:.3} +- {d.spec_flux_1TeV_err.value:.3} (stat) cm-2 s-1\n"  # noqa: E501
        ss += f"Integrated flux (>1 TeV): {d.spec_flux_1TeV_crab:.3f} +- {d.spec_flux_1TeV_crab_err:.3f} (% Crab)\n"  # noqa: E501
        ss += f"Integrated flux (1-10 TeV): {d.spec_eflux_1TeV_10TeV.value:.3} +- {d.spec_eflux_1TeV_10TeV_err.value:.3} (stat) erg cm-2 s-1\n"  # noqa: E501

        return ss

    def _info_spectral_points(self):
        """Print spectral points info."""
        d = self.data
        ss = "\n*** Spectral points ***\n\n"
        ss += f"SED reference ID: {d.sed_reference_id}\n"
        ss += f"Number of spectral points: {d.sed_n_points}\n"
        ss += f"Number of upper limits: {d.sed_n_ul}\n\n"

        table = self.flux_points_table
        if table is None:
            ss += "\nNo spectral points available for this source."
        else:
            lines = format_flux_points_table(table).pformat(max_width=-1, max_lines=-1)
            ss += "\n".join(lines)

        return ss + "\n"

    def spectral_model(self):
        """Source spectral model (`~gammapy.modeling.models.SpectralModel`).

        Parameter errors are statistical errors only.
        """
        data = self.data
        spec_type = data["spec_type"]

        if spec_type == "pl":
            tag = "PowerLawSpectralModel"
            pars = {
                "amplitude": data["spec_pl_norm"],
                "index": data["spec_pl_index"],
                "reference": data["spec_pl_e_ref"],
            }
            errs = {
                "amplitude": data["spec_pl_norm_err"],
                "index": data["spec_pl_index_err"],
            }
        elif spec_type == "pl2":
            e_max = data["spec_pl2_e_max"]
            DEFAULT_E_MAX = u.Quantity(1e5, "TeV")

            if np.isnan(e_max.value) or e_max.value == 0:
                e_max = DEFAULT_E_MAX

            tag = "PowerLaw2SpectralModel"
            pars = {
                "amplitude": data["spec_pl2_flux"],
                "index": data["spec_pl2_index"],
                "emin": data["spec_pl2_e_min"],
                "emax": e_max,
            }
            errs = {
                "amplitude": data["spec_pl2_flux_err"],
                "index": data["spec_pl2_index_err"],
            }
        elif spec_type == "ecpl":
            tag = "ExpCutoffPowerLawSpectralModel"
            pars = {
                "amplitude": data["spec_ecpl_norm"],
                "index": data["spec_ecpl_index"],
                "lambda_": 1.0 / data["spec_ecpl_e_cut"],
                "reference": data["spec_ecpl_e_ref"],
            }
            errs = {
                "amplitude": data["spec_ecpl_norm_err"],
                "index": data["spec_ecpl_index_err"],
                "lambda_": data["spec_ecpl_e_cut_err"] / data["spec_ecpl_e_cut"] ** 2,
            }
        elif spec_type == "none":
            return None
        else:
            raise ValueError(f"Invalid spec_type: {spec_type}")

        model = Model.create(tag, "spectral", **pars)
        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self):
        """Source spatial model (`~gammapy.modeling.models.SpatialModel`).

        TODO: add parameter errors!
        """
        d = self.data
        morph_type = d["morph_type"]

        pars = {"lon_0": d["glon"], "lat_0": d["glat"], "frame": "galactic"}
        errs = {
            "lat_0": self.data["pos_err"],
            "lon_0": self.data["pos_err"] / np.cos(self.data["glat"]),
        }

        if morph_type == "point":
            tag = "PointSpatialModel"
        elif morph_type == "gauss":
            # TODO: add infos back once model support elongation
            # pars['x_stddev'] = d['morph_sigma']
            # pars['y_stddev'] = d['morph_sigma']
            # if not np.isnan(d['morph_sigma2']):
            #     pars['y_stddev'] = d['morph_sigma2']
            # if not np.isnan(d['morph_pa']):
            #     # TODO: handle reference frame for rotation angle
            #     pars['theta'] = Angle(d['morph_pa'], 'deg').rad
            tag = "GaussianSpatialModel"
            pars["sigma"] = d["morph_sigma"]
        elif morph_type == "shell":
            tag = "ShellSpatialModel"
            # TODO: probably we shouldn't guess a shell width here!
            pars["radius"] = 0.8 * d["morph_sigma"]
            pars["width"] = 0.2 * d["morph_sigma"]
        elif morph_type == "none":
            return None
        else:
            raise ValueError(f"Invalid morph_type: {morph_type!r}")

        model = Model.create(tag, "spatial", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def sky_model(self):
        """Source sky model (`~gammapy.modeling.models.SkyModel`)."""
        return SkyModel(
            spatial_model=self.spatial_model(),
            spectral_model=self.spectral_model(),
            name=self.name,
        )

    def _add_source_meta(self, table):
        """Copy over some info to table.meta"""
        d = self.data
        m = table.meta
        m["origin"] = "Data from gamma-cat"
        m["source_id"] = d["source_id"]
        m["common_name"] = d["common_name"]
        m["reference_id"] = d["reference_id"]

    @property
    def flux_points_table(self):
        """Differential flux points (`~gammapy.estimators.FluxPoints`)."""
        d = self.data
        table = Table()
        table.meta["SED_TYPE"] = "dnde"
        self._add_source_meta(table)

        valid = np.isfinite(d["sed_e_ref"].value)

        if valid.sum() == 0:
            return None

        table["e_ref"] = d["sed_e_ref"]
        table["e_min"] = d["sed_e_min"]
        table["e_max"] = d["sed_e_max"]

        table["dnde"] = d["sed_dnde"]
        table["dnde_err"] = d["sed_dnde_err"]
        table["dnde_errn"] = d["sed_dnde_errn"]
        table["dnde_errp"] = d["sed_dnde_errp"]
        table["dnde_ul"] = d["sed_dnde_ul"]

        # Only keep rows that actually contain information
        table = table[valid]

        for colname in table.colnames:
            if not np.isfinite(table[colname]).any():
                table.remove_column(colname)

        return table


class SourceCatalogGammaCat(SourceCatalog):
    """Gammacat open TeV source catalog.

    See: https://github.com/gammapy/gamma-cat

    One source is represented by `~gammapy.catalog.SourceCatalogObjectGammaCat`.

    Parameters
    ----------
    filename : str
        Path to the gamma-cat fits file.

    Examples
    --------
    Load the catalog data:

    >>> import matplotlib.pyplot as plt
    >>> import astropy.units as u
    >>> from gammapy.catalog import SourceCatalogGammaCat
    >>> cat = SourceCatalogGammaCat()

    Access a source by name:

    >>> source = cat['Vela Junior']

    Access source spectral data and plot it:

    >>> ax= plt.subplot()
    >>> energy_range = [1, 10] * u.TeV
    >>> source.spectral_model().plot(energy_range, ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>
    >>> source.spectral_model().plot_error(energy_range, ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>
    >>> source.flux_points.plot(ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>
    """

    tag = "gamma-cat"
    description = "An open catalog of gamma-ray sources"
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/gammacat/gammacat.fits.gz"):
        filename = make_path(filename)
        table = Table.read(filename, hdu=1)

        source_name_key = "common_name"
        source_name_alias = ("other_names", "gamma_names")
        super().__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )
