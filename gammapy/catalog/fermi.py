# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes."""
import abc
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
from gammapy.estimators import FluxPoints, LightCurve
from gammapy.modeling.models import (
    DiskSpatialModel,
    GaussianSpatialModel,
    Model,
    PointSpatialModel,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_standardise_units_inplace
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalogObject4FGL",
    "SourceCatalogObject3FGL",
    "SourceCatalogObject2FHL",
    "SourceCatalogObject3FHL",
    "SourceCatalog4FGL",
    "SourceCatalog3FGL",
    "SourceCatalog2FHL",
    "SourceCatalog3FHL",
]


def compute_flux_points_ul(quantity, quantity_errp):
    """Compute UL value for fermi flux points.

    See https://arxiv.org/pdf/1501.02003.pdf (page 30)
    """
    return 2 * quantity_errp + quantity


class SourceCatalogObjectFermiBase(SourceCatalogObject, abc.ABC):
    """Base class for Fermi-LAT catalogs."""

    asso = ["ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM1", "ASSOC_GAM2", "ASSOC_GAM3"]

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'more', 'position', 'spectral','lightcurve'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,more,position,spectral,lightcurve"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "more" in ops:
            ss += self._info_more()
        if "position" in ops:
            ss += self._info_position()
            if not self.is_pointlike:
                ss += self._info_morphology()
        if "spectral" in ops:
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()
        if "lightcurve" in ops:
            ss += self._info_lightcurve()
        return ss

    def _info_basic(self):
        d = self.data
        keys = self.asso
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(self.row_index)
        ss += "{:<20s} : {}\n".format("Source name", self.name)
        if "Extended_Source_Name" in d:
            ss += "{:<20s} : {}\n".format("Extended name", d["Extended_Source_Name"])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ", ".join([_ for _ in vals if _ != ""])

        associations = get_nonentry_keys(keys)
        ss += "{:<16s} : {}\n".format("Associations", associations)
        try:
            ss += "{:<16s} : {:.3f}\n".format("ASSOC_PROB_BAY", d["ASSOC_PROB_BAY"])
            ss += "{:<16s} : {:.3f}\n".format("ASSOC_PROB_LR", d["ASSOC_PROB_LR"])
        except (KeyError):
            pass
        try:
            ss += "{:<16s} : {}\n".format("Class1", d["CLASS1"])
        except (KeyError):
            ss += "{:<16s} : {}\n".format("Class", d["CLASS"])
        try:
            ss += "{:<16s} : {}\n".format("Class2", d["CLASS2"])
        except (KeyError):
            pass
        ss += "{:<16s} : {}\n".format("TeVCat flag", d.get("TEVCAT_FLAG", "N/A"))
        return ss

    @abc.abstractmethod
    def _info_more(self):
        pass

    def _info_position(self):
        d = self.data
        ss = "\n*** Position info ***\n\n"
        ss += "{:<20s} : {:.3f}\n".format("RA", d["RAJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("DEC", d["DEJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("GLON", d["GLON"])
        ss += "{:<20s} : {:.3f}\n".format("GLAT", d["GLAT"])

        ss += "\n"
        ss += "{:<20s} : {:.4f}\n".format("Semimajor (68%)", d["Conf_68_SemiMajor"])
        ss += "{:<20s} : {:.4f}\n".format("Semiminor (68%)", d["Conf_68_SemiMinor"])
        ss += "{:<20s} : {:.2f}\n".format("Position angle (68%)", d["Conf_68_PosAng"])
        ss += "{:<20s} : {:.4f}\n".format("Semimajor (95%)", d["Conf_95_SemiMajor"])
        ss += "{:<20s} : {:.4f}\n".format("Semiminor (95%)", d["Conf_95_SemiMinor"])
        ss += "{:<20s} : {:.2f}\n".format("Position angle (95%)", d["Conf_95_PosAng"])
        ss += "{:<20s} : {:.0f}\n".format("ROI number", d["ROI_num"])
        return ss

    def _info_morphology(self):
        e = self.data_extended
        ss = "\n*** Extended source information ***\n\n"
        ss += "{:<16s} : {}\n".format("Model form", e["Model_Form"])
        ss += "{:<16s} : {:.4f}\n".format("Model semimajor", e["Model_SemiMajor"])
        ss += "{:<16s} : {:.4f}\n".format("Model semiminor", e["Model_SemiMinor"])
        ss += "{:<16s} : {:.4f}\n".format("Position angle", e["Model_PosAng"])
        try:
            ss += "{:<16s} : {}\n".format("Spatial function", e["Spatial_Function"])
        except KeyError:
            pass
        ss += "{:<16s} : {}\n\n".format("Spatial filename", e["Spatial_Filename"])
        return ss

    def _info_spectral_fit(self):
        return "\n"

    def _info_spectral_points(self):
        ss = "\n*** Spectral points ***\n\n"
        lines = self.flux_points.table_formatted.pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)
        return ss

    def _info_lightcurve(self):
        return "\n"

    @property
    def is_pointlike(self):
        return self.data["Extended_Source_Name"].strip() == ""

    # FIXME: this should be renamed `set_position_error`,
    # and `phi_0` isn't filled correctly, other parameters missing
    # see https://github.com/gammapy/gammapy/pull/2533#issuecomment-553329049
    def _set_spatial_errors(self, model):
        d = self.data

        if "Pos_err_68" in d:
            percent = 0.68
            semi_minor = d["Pos_err_68"]
            semi_major = d["Pos_err_68"]
            phi_0 = 0.0
        else:
            percent = 0.95
            semi_minor = d["Conf_95_SemiMinor"]
            semi_major = d["Conf_95_SemiMajor"]
            phi_0 = d["Conf_95_PosAng"]

        if np.isnan(phi_0):
            phi_0 = 0.0 * u.deg

        scale_1sigma = Gauss2DPDF().containment_radius(percent)
        lat_err = semi_major / scale_1sigma
        lon_err = semi_minor / scale_1sigma / np.cos(d["DEJ2000"])

        if "TemplateSpatialModel" not in model.tag:
            model.parameters["lon_0"].error = lon_err
            model.parameters["lat_0"].error = lat_err
            model.phi_0 = phi_0

    def sky_model(self, name=None):
        """Sky model (`~gammapy.modeling.models.SkyModel`)."""
        if name is None:
            name = self.name

        return SkyModel(
            spatial_model=self.spatial_model(),
            spectral_model=self.spectral_model(),
            name=name,
        )


class SourceCatalogObject4FGL(SourceCatalogObjectFermiBase):
    """One source from the Fermi-LAT 4FGL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog4FGL`.
    """

    asso = [
        "ASSOC1",
        "ASSOC2",
        "ASSOC_TEV",
        "ASSOC_FGL",
        "ASSOC_FHL",
        "ASSOC_GAM1",
        "ASSOC_GAM2",
        "ASSOC_GAM3",
    ]
    _energy_edges = u.Quantity([50, 100, 300, 1000, 3000, 10000, 30000, 300000], "MeV")

    def _info_more(self):
        d = self.data
        ss = "\n*** Other info ***\n\n"
        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Significance (100 MeV - 1 TeV)", d["Signif_Avg"])
        ss += "{:<32s} : {:.1f}\n".format("Npred", d["Npred"])
        ss += "\n{:<20s} : {}\n".format("Other flags", d["Flags"])
        return ss

    def _info_spectral_fit(self):
        d = self.data
        spec_type = d["SpectrumType"].strip()

        ss = "\n*** Spectral info ***\n\n"

        ss += "{:<45s} : {}\n".format("Spectrum type", d["SpectrumType"])
        fmt = "{:<45s} : {:.3f}\n"
        ss += fmt.format("Detection significance (100 MeV - 1 TeV)", d["Signif_Avg"])

        if spec_type == "PowerLaw":
            tag = "PL"
        elif spec_type == "LogParabola":
            tag = "LP"
            ss += "{:<45s} : {:.4f} +- {:.5f}\n".format(
                "beta", d["LP_beta"], d["Unc_LP_beta"]
            )
            ss += "{:<45s} : {:.1f}\n".format("Significance curvature", d["LP_SigCurv"])

        elif spec_type == "PLSuperExpCutoff":
            tag = "PLEC"
            fmt = "{:<45s} : {} +- {}\n"
            ss += fmt.format(
                "Exponential factor", d["PLEC_Expfactor"], d["Unc_PLEC_Expfactor"]
            )
            ss += "{:<45s} : {} +- {}\n".format(
                "Super-exponential cutoff index",
                d["PLEC_Exp_Index"],
                d["Unc_PLEC_Exp_Index"],
            )
            ss += "{:<45s} : {:.1f}\n".format(
                "Significance curvature", d["PLEC_SigCurv"]
            )

        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        ss += "{:<45s} : {:.0f} {}\n".format(
            "Pivot energy", d["Pivot_Energy"].value, d["Pivot_Energy"].unit
        )

        fmt = "{:<45s} : {:.3f} +- {:.3f}\n"
        ss += fmt.format(
            "Spectral index", d[tag + "_Index"], d["Unc_" + tag + "_Index"]
        )

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Flux Density at pivot energy",
            d[tag + "_Flux_Density"].value,
            d["Unc_" + tag + "_Flux_Density"].value,
            "cm-2 MeV-1 s-1",
        )

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Integral flux (1 - 100 GeV)",
            d["Flux1000"].value,
            d["Unc_Flux1000"].value,
            "cm-2 s-1",
        )

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Energy flux (100 MeV - 100 GeV)",
            d["Energy_Flux100"].value,
            d["Unc_Energy_Flux100"].value,
            "erg cm-2 s-1",
        )

        return ss

    def _info_lightcurve(self):
        d = self.data
        ss = "\n*** Lightcurve info ***\n\n"
        ss += "Lightcurve measured in the energy band: 100 MeV - 100 GeV\n\n"

        ss += "{:<15s} : {:.3f}\n".format("Variability index", d["Variability_Index"])

        if np.isfinite(d["Flux_Peak"]):
            ss += "{:<40s} : {:.3f}\n".format(
                "Significance peak (100 MeV - 100 GeV)", d["Signif_Peak"]
            )

            fmt = "{:<40s} : {:.3} +- {:.3} cm^-2 s^-1\n"
            ss += fmt.format(
                "Integral flux peak (100 MeV - 100 GeV)",
                d["Flux_Peak"].value,
                d["Unc_Flux_Peak"].value,
            )

            # TODO: give time as UTC string, not MET
            ss += "{:<40s} : {:.3} s (Mission elapsed time)\n".format(
                "Time peak", d["Time_Peak"].value
            )
            peak_interval = d["Peak_Interval"].to_value("day")
            ss += "{:<40s} : {:.3} day\n".format("Peak interval", peak_interval)
        else:
            ss += "\nNo peak measured for this source.\n"

        # TODO: Add a lightcurve table with d['Flux_History'] and d['Unc_Flux_History']

        return ss

    def spatial_model(self):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`)."""
        d = self.data
        ra = d["RAJ2000"]
        dec = d["DEJ2000"]

        if self.is_pointlike:
            model = PointSpatialModel(lon_0=ra, lat_0=dec, frame="icrs")
        else:
            de = self.data_extended
            morph_type = de["Model_Form"].strip()
            e = (1 - (de["Model_SemiMinor"] / de["Model_SemiMajor"]) ** 2.0) ** 0.5
            sigma = de["Model_SemiMajor"]
            phi = de["Model_PosAng"]
            if morph_type == "Disk":
                r_0 = de["Model_SemiMajor"]
                model = DiskSpatialModel(
                    lon_0=ra, lat_0=dec, r_0=r_0, e=e, phi=phi, frame="icrs"
                )
            elif morph_type in ["Map", "Ring", "2D Gaussian x2"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/LAT_extended_sources_8years/Templates/"
                )
                with warnings.catch_warnings():  # ignore FITS units warnings
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    model = TemplateSpatialModel.read(path / filename)
            elif morph_type == "2D Gaussian":
                model = GaussianSpatialModel(
                    lon_0=ra, lat_0=dec, sigma=sigma, e=e, phi=phi, frame="icrs"
                )
            else:
                raise ValueError(f"Invalid spatial model: {morph_type!r}")
        self._set_spatial_errors(model)
        return model

    def spectral_model(self):
        """Best fit spectral model (`~gammapy.modeling.models.SpectralModel`)."""
        spec_type = self.data["SpectrumType"].strip()

        if spec_type == "PowerLaw":
            tag = "PowerLawSpectralModel"
            pars = {
                "reference": self.data["Pivot_Energy"],
                "amplitude": self.data["PL_Flux_Density"],
                "index": self.data["PL_Index"],
            }
            errs = {
                "amplitude": self.data["Unc_PL_Flux_Density"],
                "index": self.data["Unc_PL_Index"],
            }
        elif spec_type == "LogParabola":
            tag = "LogParabolaSpectralModel"
            pars = {
                "reference": self.data["Pivot_Energy"],
                "amplitude": self.data["LP_Flux_Density"],
                "alpha": self.data["LP_Index"],
                "beta": self.data["LP_beta"],
            }
            errs = {
                "amplitude": self.data["Unc_LP_Flux_Density"],
                "alpha": self.data["Unc_LP_Index"],
                "beta": self.data["Unc_LP_beta"],
            }
        elif spec_type == "PLSuperExpCutoff":
            tag = "SuperExpCutoffPowerLaw4FGLSpectralModel"
            pars = {
                "reference": self.data["Pivot_Energy"],
                "amplitude": self.data["PLEC_Flux_Density"],
                "index_1": self.data["PLEC_Index"],
                "index_2": self.data["PLEC_Exp_Index"],
                "expfactor": self.data["PLEC_Expfactor"],
            }
            errs = {
                "amplitude": self.data["Unc_PLEC_Flux_Density"],
                "index_1": self.data["Unc_PLEC_Index"],
                "index_2": np.nan_to_num(self.data["Unc_PLEC_Exp_Index"]),
                "expfactor": self.data["Unc_PLEC_Expfactor"],
            }
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model = Model.create(tag, "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"

        table["e_min"] = self._energy_edges[:-1]
        table["e_max"] = self._energy_edges[1:]

        flux = self._get_flux_values("Flux_Band")
        flux_err = self._get_flux_values("Unc_Flux_Band")
        table["flux"] = flux
        table["flux_errn"] = np.abs(flux_err[:, 0])
        table["flux_errp"] = flux_err[:, 1]

        nuFnu = self._get_flux_values("nuFnu_Band", "erg cm-2 s-1")
        table["e2dnde"] = nuFnu
        table["e2dnde_errn"] = np.abs(nuFnu * flux_err[:, 0] / flux)
        table["e2dnde_errp"] = nuFnu * flux_err[:, 1] / flux

        is_ul = np.isnan(table["flux_errn"])
        table["is_ul"] = is_ul

        # handle upper limits
        table["flux_ul"] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table["flux"], table["flux_errp"])
        table["flux_ul"][is_ul] = flux_ul[is_ul]

        # handle upper limits
        table["e2dnde_ul"] = np.nan * nuFnu.unit
        e2dnde_ul = compute_flux_points_ul(table["e2dnde"], table["e2dnde_errp"])
        table["e2dnde_ul"][is_ul] = e2dnde_ul[is_ul]

        # Square root of test statistic
        table["sqrt_TS"] = self.data["Sqrt_TS_Band"]
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = self.data[prefix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.estimators.LightCurve`)."""
        flux = self.data["Flux_History"]

        # Flux error is given as asymmetric high/low
        flux_errn = -self.data["Unc_Flux_History"][:, 0]
        flux_errp = self.data["Unc_Flux_History"][:, 1]

        # Really the time binning is stored in a separate HDU in the FITS
        # catalog file called `Hist_Start`, with a single column `Hist_Start`
        # giving the time binning in MET (mission elapsed time)
        # This is not available here for now.
        # TODO: read that info in `SourceCatalog3FGL` and pass it down to the
        # `SourceCatalogObject3FGL` object somehow.

        # For now, we just hard-code the start and stop time and assume
        # equally-spaced time intervals. This is roughly correct,
        # for plotting the difference doesn't matter, only for analysis
        time_start = Time("2008-08-04T15:43:36.0000")
        time_end = Time("2016-08-02T05:44:11.9999")
        n_points = len(flux)
        time_step = (time_end - time_start) / n_points
        time_bounds = time_start + np.arange(n_points + 1) * time_step

        table = Table(
            [
                Column(time_bounds[:-1].utc.mjd, "time_min"),
                Column(time_bounds[1:].utc.mjd, "time_max"),
                Column(flux, "flux"),
                Column(flux_errp, "flux_errp"),
                Column(flux_errn, "flux_errn"),
            ]
        )
        return LightCurve(table)


class SourceCatalogObject3FGL(SourceCatalogObjectFermiBase):
    """One source from the Fermi-LAT 3FGL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FGL`.
    """

    _energy_edges = u.Quantity([100, 300, 1000, 3000, 10000, 100000], "MeV")
    _energy_edges_suffix = [
        "100_300",
        "300_1000",
        "1000_3000",
        "3000_10000",
        "10000_100000",
    ]
    energy_range = u.Quantity([100, 100000], "MeV")
    """Energy range used for the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def _info_more(self):
        d = self.data
        ss = "\n*** Other info ***\n\n"
        ss += "{:<20s} : {}\n".format("Other flags", d["Flags"])
        return ss

    def _info_spectral_fit(self):
        d = self.data
        spec_type = d["SpectrumType"].strip()

        ss = "\n*** Spectral info ***\n\n"

        ss += "{:<45s} : {}\n".format("Spectrum type", d["SpectrumType"])
        fmt = "{:<45s} : {:.3f}\n"
        ss += fmt.format("Detection significance (100 MeV - 300 GeV)", d["Signif_Avg"])
        ss += "{:<45s} : {:.1f}\n".format("Significance curvature", d["Signif_Curve"])

        if spec_type == "PowerLaw":
            pass
        elif spec_type == "LogParabola":
            ss += "{:<45s} : {} +- {}\n".format("beta", d["beta"], d["Unc_beta"])
        elif spec_type in ["PLExpCutoff", "PlSuperExpCutoff"]:
            fmt = "{:<45s} : {:.0f} +- {:.0f} {}\n"
            ss += fmt.format(
                "Cutoff energy",
                d["Cutoff"].value,
                d["Unc_Cutoff"].value,
                d["Cutoff"].unit,
            )
        elif spec_type == "PLSuperExpCutoff":
            ss += "{:<45s} : {} +- {}\n".format(
                "Super-exponential cutoff index", d["Exp_Index"], d["Unc_Exp_Index"]
            )
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        ss += "{:<45s} : {:.0f} {}\n".format(
            "Pivot energy", d["Pivot_Energy"].value, d["Pivot_Energy"].unit
        )

        ss += "{:<45s} : {:.3f}\n".format(
            "Power law spectral index", d["PowerLaw_Index"]
        )

        fmt = "{:<45s} : {:.3f} +- {:.3f}\n"
        ss += fmt.format("Spectral index", d["Spectral_Index"], d["Unc_Spectral_Index"])

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Flux Density at pivot energy",
            d["Flux_Density"].value,
            d["Unc_Flux_Density"].value,
            "cm-2 MeV-1 s-1",
        )

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Integral flux (1 - 100 GeV)",
            d["Flux1000"].value,
            d["Unc_Flux1000"].value,
            "cm-2 s-1",
        )

        fmt = "{:<45s} : {:.3} +- {:.3} {}\n"
        ss += fmt.format(
            "Energy flux (100 MeV - 100 GeV)",
            d["Energy_Flux100"].value,
            d["Unc_Energy_Flux100"].value,
            "erg cm-2 s-1",
        )

        return ss

    def _info_lightcurve(self):
        d = self.data
        ss = "\n*** Lightcurve info ***\n\n"
        ss += "Lightcurve measured in the energy band: 100 MeV - 100 GeV\n\n"

        ss += "{:<15s} : {:.3f}\n".format("Variability index", d["Variability_Index"])

        if np.isfinite(d["Flux_Peak"]):
            ss += "{:<40s} : {:.3f}\n".format(
                "Significance peak (100 MeV - 100 GeV)", d["Signif_Peak"]
            )

            fmt = "{:<40s} : {:.3} +- {:.3} cm^-2 s^-1\n"
            ss += fmt.format(
                "Integral flux peak (100 MeV - 100 GeV)",
                d["Flux_Peak"].value,
                d["Unc_Flux_Peak"].value,
            )

            # TODO: give time as UTC string, not MET
            ss += "{:<40s} : {:.3} s (Mission elapsed time)\n".format(
                "Time peak", d["Time_Peak"].value
            )
            peak_interval = d["Peak_Interval"].to_value("day")
            ss += "{:<40s} : {:.3} day\n".format("Peak interval", peak_interval)
        else:
            ss += "\nNo peak measured for this source.\n"

        # TODO: Add a lightcurve table with d['Flux_History'] and d['Unc_Flux_History']

        return ss

    def spectral_model(self):
        """Best fit spectral model (`~gammapy.modeling.models.SpectralModel`)."""
        spec_type = self.data["SpectrumType"].strip()

        if spec_type == "PowerLaw":
            tag = "PowerLawSpectralModel"
            pars = {
                "amplitude": self.data["Flux_Density"],
                "reference": self.data["Pivot_Energy"],
                "index": self.data["Spectral_Index"],
            }
            errs = {
                "amplitude": self.data["Unc_Flux_Density"],
                "index": self.data["Unc_Spectral_Index"],
            }
        elif spec_type == "PLExpCutoff":
            tag = "ExpCutoffPowerLaw3FGLSpectralModel"
            pars = {
                "amplitude": self.data["Flux_Density"],
                "reference": self.data["Pivot_Energy"],
                "index": self.data["Spectral_Index"],
                "ecut": self.data["Cutoff"],
            }
            errs = {
                "amplitude": self.data["Unc_Flux_Density"],
                "index": self.data["Unc_Spectral_Index"],
                "ecut": self.data["Unc_Cutoff"],
            }
        elif spec_type == "LogParabola":
            tag = "LogParabolaSpectralModel"
            pars = {
                "amplitude": self.data["Flux_Density"],
                "reference": self.data["Pivot_Energy"],
                "alpha": self.data["Spectral_Index"],
                "beta": self.data["beta"],
            }
            errs = {
                "amplitude": self.data["Unc_Flux_Density"],
                "alpha": self.data["Unc_Spectral_Index"],
                "beta": self.data["Unc_beta"],
            }
        elif spec_type == "PLSuperExpCutoff":
            tag = "SuperExpCutoffPowerLaw3FGLSpectralModel"
            pars = {
                "amplitude": self.data["Flux_Density"],
                "reference": self.data["Pivot_Energy"],
                "index_1": self.data["Spectral_Index"],
                "index_2": self.data["Exp_Index"],
                "ecut": self.data["Cutoff"],
            }
            errs = {
                "amplitude": self.data["Unc_Flux_Density"],
                "index_1": self.data["Unc_Spectral_Index"],
                "index_2": self.data["Unc_Exp_Index"],
                "ecut": self.data["Unc_Cutoff"],
            }
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model = Model.create(tag, "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`)."""
        d = self.data
        ra = d["RAJ2000"]
        dec = d["DEJ2000"]

        if self.is_pointlike:
            model = PointSpatialModel(lon_0=ra, lat_0=dec, frame="icrs")
        else:
            de = self.data_extended
            morph_type = de["Model_Form"].strip()
            e = (1 - (de["Model_SemiMinor"] / de["Model_SemiMajor"]) ** 2.0) ** 0.5
            sigma = de["Model_SemiMajor"]
            phi = de["Model_PosAng"]
            if morph_type == "Disk":
                r_0 = de["Model_SemiMajor"]
                model = DiskSpatialModel(
                    lon_0=ra, lat_0=dec, r_0=r_0, e=e, phi=phi, frame="icrs"
                )
            elif morph_type in ["Map", "Ring", "2D Gaussian x2"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v15/Templates/"
                )
                model = TemplateSpatialModel.read(path / filename)
            elif morph_type == "2D Gaussian":
                model = GaussianSpatialModel(
                    lon_0=ra, lat_0=dec, sigma=sigma, e=e, phi=phi, frame="icrs"
                )
            else:
                raise ValueError(f"Invalid spatial model: {morph_type!r}")
        self._set_spatial_errors(model)
        return model

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"

        table["e_min"] = self._energy_edges[:-1]
        table["e_max"] = self._energy_edges[1:]

        flux = self._get_flux_values("Flux")
        flux_err = self._get_flux_values("Unc_Flux")
        table["flux"] = flux
        table["flux_errn"] = np.abs(flux_err[:, 0])
        table["flux_errp"] = flux_err[:, 1]

        nuFnu = self._get_flux_values("nuFnu", "erg cm-2 s-1")
        table["e2dnde"] = nuFnu
        table["e2dnde_errn"] = np.abs(nuFnu * flux_err[:, 0] / flux)
        table["e2dnde_errp"] = nuFnu * flux_err[:, 1] / flux

        is_ul = np.isnan(table["flux_errn"])
        table["is_ul"] = is_ul

        # handle upper limits
        table["flux_ul"] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table["flux"], table["flux_errp"])
        table["flux_ul"][is_ul] = flux_ul[is_ul]

        # handle upper limits
        table["e2dnde_ul"] = np.nan * nuFnu.unit
        e2dnde_ul = compute_flux_points_ul(table["e2dnde"], table["e2dnde_errp"])
        table["e2dnde_ul"][is_ul] = e2dnde_ul[is_ul]

        # Square root of test statistic
        table["sqrt_TS"] = [self.data["Sqrt_TS" + _] for _ in self._energy_edges_suffix]
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _] for _ in self._energy_edges_suffix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.estimators.LightCurve`)."""
        flux = self.data["Flux_History"]

        # Flux error is given as asymmetric high/low
        flux_errn = -self.data["Unc_Flux_History"][:, 0]
        flux_errp = self.data["Unc_Flux_History"][:, 1]

        # Really the time binning is stored in a separate HDU in the FITS
        # catalog file called `Hist_Start`, with a single column `Hist_Start`
        # giving the time binning in MET (mission elapsed time)
        # This is not available here for now.
        # TODO: read that info in `SourceCatalog3FGL` and pass it down to the
        # `SourceCatalogObject3FGL` object somehow.

        # For now, we just hard-code the start and stop time and assume
        # equally-spaced time intervals. This is roughly correct,
        # for plotting the difference doesn't matter, only for analysis
        time_start = Time("2008-08-02T00:33:19")
        time_end = Time("2012-07-31T22:45:47")
        n_points = len(flux)
        time_step = (time_end - time_start) / n_points
        time_bounds = time_start + np.arange(n_points + 1) * time_step

        table = Table(
            [
                Column(time_bounds[:-1].utc.mjd, "time_min"),
                Column(time_bounds[1:].utc.mjd, "time_max"),
                Column(flux, "flux"),
                Column(flux_errp, "flux_errp"),
                Column(flux_errn, "flux_errn"),
            ]
        )
        return LightCurve(table)


class SourceCatalogObject2FHL(SourceCatalogObjectFermiBase):
    """One source from the Fermi-LAT 2FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2FHL`.
    """

    asso = ["ASSOC", "3FGL_Name", "1FHL_Name", "TeVCat_Name"]
    _energy_edges = u.Quantity([50, 171, 585, 2000], "GeV")
    _energy_edges_suffix = ["50_171", "171_585", "585_2000"]
    energy_range = u.Quantity([0.05, 2], "TeV")
    """Energy range used for the catalog."""

    def _info_more(self):
        d = self.data
        ss = "\n*** Other info ***\n\n"
        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Test statistic (50 GeV - 2 TeV)", d["TS"])
        return ss

    def _info_position(self):
        d = self.data
        ss = "\n*** Position info ***\n\n"
        ss += "{:<20s} : {:.3f}\n".format("RA", d["RAJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("DEC", d["DEJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("GLON", d["GLON"])
        ss += "{:<20s} : {:.3f}\n".format("GLAT", d["GLAT"])

        ss += "\n"
        ss += "{:<20s} : {:.4f}\n".format("Error on position (68%)", d["Pos_err_68"])
        ss += "{:<20s} : {:.0f}\n".format("ROI number", d["ROI"])
        return ss

    def _info_spectral_fit(self):
        d = self.data

        ss = "\n*** Spectral fit info ***\n\n"

        fmt = "{:<32s} : {:.3f} +- {:.3f}\n"
        ss += fmt.format(
            "Power-law spectral index", d["Spectral_Index"], d["Unc_Spectral_Index"]
        )

        ss += "{:<32s} : {:.3} +- {:.3} {}\n".format(
            "Integral flux (50 GeV - 2 TeV)",
            d["Flux50"].value,
            d["Unc_Flux50"].value,
            "cm-2 s-1",
        )

        ss += "{:<32s} : {:.3} +- {:.3} {}\n".format(
            "Energy flux (50 GeV - 2 TeV)",
            d["Energy_Flux50"].value,
            d["Unc_Energy_Flux50"].value,
            "erg cm-2 s-1",
        )

        return ss

    @property
    def is_pointlike(self):
        return self.data["Source_Name"].strip()[-1] != "e"

    def spatial_model(self):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`)."""
        d = self.data
        ra = d["RAJ2000"]
        dec = d["DEJ2000"]

        if self.is_pointlike:
            model = PointSpatialModel(lon_0=ra, lat_0=dec, frame="icrs")
        else:
            de = self.data_extended
            morph_type = de["Model_Form"].strip()
            e = (1 - (de["Model_SemiMinor"] / de["Model_SemiMajor"]) ** 2.0) ** 0.5
            sigma = de["Model_SemiMajor"]
            phi = de["Model_PosAng"]
            if morph_type in ["Disk", "Elliptical Disk"]:
                r_0 = de["Model_SemiMajor"]
                model = DiskSpatialModel(
                    lon_0=ra, lat_0=dec, r_0=r_0, e=e, phi=phi, frame="icrs"
                )
            elif morph_type in ["Map", "Ring", "2D Gaussian x2"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v15/Templates/"
                )
                return TemplateSpatialModel.read(path / filename)
            elif morph_type in ["2D Gaussian", "Elliptical 2D Gaussian"]:
                model = GaussianSpatialModel(
                    lon_0=ra, lat_0=dec, sigma=sigma, e=e, phi=phi, frame="icrs"
                )
            else:
                raise ValueError(f"Invalid spatial model: {morph_type!r}")

        self._set_spatial_errors(model)
        return model

    def spectral_model(self):
        """Best fit spectral model (`~gammapy.modeling.models.SpectralModel`)."""
        tag = "PowerLaw2SpectralModel"
        pars = {
            "amplitude": self.data["Flux50"],
            "emin": self.energy_range[0],
            "emax": self.energy_range[1],
            "index": self.data["Spectral_Index"],
        }
        errs = {
            "amplitude": self.data["Unc_Flux50"],
            "index": self.data["Unc_Spectral_Index"],
        }

        model = Model.create(tag, "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.estimators.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._energy_edges[:-1]
        table["e_max"] = self._energy_edges[1:]
        table["flux"] = self._get_flux_values("Flux")
        flux_err = self._get_flux_values("Unc_Flux")
        table["flux_errn"] = np.abs(flux_err[:, 0])
        table["flux_errp"] = flux_err[:, 1]

        # handle upper limits
        is_ul = np.isnan(table["flux_errn"])
        table["is_ul"] = is_ul
        table["flux_ul"] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table["flux"], table["flux_errp"])
        table["flux_ul"][is_ul] = flux_ul[is_ul]
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _ + "GeV"] for _ in self._energy_edges_suffix]
        return u.Quantity(values, unit)


class SourceCatalogObject3FHL(SourceCatalogObjectFermiBase):
    """One source from the Fermi-LAT 3FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FHL`.
    """

    asso = ["ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM"]
    energy_range = u.Quantity([0.01, 2], "TeV")
    """Energy range used for the catalog."""

    _energy_edges = u.Quantity([10, 20, 50, 150, 500, 2000], "GeV")

    def _info_position(self):
        d = self.data
        ss = "\n*** Position info ***\n\n"
        ss += "{:<20s} : {:.3f}\n".format("RA", d["RAJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("DEC", d["DEJ2000"])
        ss += "{:<20s} : {:.3f}\n".format("GLON", d["GLON"])
        ss += "{:<20s} : {:.3f}\n".format("GLAT", d["GLAT"])

        # TODO: All sources are non-elliptical; just give one number for radius?
        ss += "\n"
        ss += "{:<20s} : {:.4f}\n".format("Semimajor (95%)", d["Conf_95_SemiMajor"])
        ss += "{:<20s} : {:.4f}\n".format("Semiminor (95%)", d["Conf_95_SemiMinor"])
        ss += "{:<20s} : {:.2f}\n".format("Position angle (95%)", d["Conf_95_PosAng"])
        ss += "{:<20s} : {:.0f}\n".format("ROI number", d["ROI_num"])

        return ss

    def _info_spectral_fit(self):
        d = self.data
        spec_type = d["SpectrumType"].strip()

        ss = "\n*** Spectral fit info ***\n\n"

        ss += "{:<32s} : {}\n".format("Spectrum type", d["SpectrumType"])
        ss += "{:<32s} : {:.1f}\n".format("Significance curvature", d["Signif_Curve"])

        # Power-law parameters are always given; give in any case
        fmt = "{:<32s} : {:.3f} +- {:.3f}\n"
        ss += fmt.format(
            "Power-law spectral index", d["PowerLaw_Index"], d["Unc_PowerLaw_Index"]
        )

        if spec_type == "PowerLaw":
            pass
        elif spec_type == "LogParabola":
            fmt = "{:<32s} : {:.3f} +- {:.3f}\n"
            ss += fmt.format(
                "LogParabolaSpectralModel spectral index",
                d["Spectral_Index"],
                d["Unc_Spectral_Index"],
            )

            ss += "{:<32s} : {:.3f} +- {:.3f}\n".format(
                "LogParabolaSpectralModel beta", d["beta"], d["Unc_beta"]
            )
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        ss += "{:<32s} : {:.1f} {}\n".format(
            "Pivot energy", d["Pivot_Energy"].value, d["Pivot_Energy"].unit
        )

        ss += "{:<32s} : {:.3} +- {:.3} {}\n".format(
            "Flux Density at pivot energy",
            d["Flux_Density"].value,
            d["Unc_Flux_Density"].value,
            "cm-2 GeV-1 s-1",
        )

        ss += "{:<32s} : {:.3} +- {:.3} {}\n".format(
            "Integral flux (10 GeV - 1 TeV)",
            d["Flux"].value,
            d["Unc_Flux"].value,
            "cm-2 s-1",
        )

        ss += "{:<32s} : {:.3} +- {:.3} {}\n".format(
            "Energy flux (10 GeV - TeV)",
            d["Energy_Flux"].value,
            d["Unc_Energy_Flux"].value,
            "erg cm-2 s-1",
        )

        return ss

    def _info_more(self):
        d = self.data
        ss = "\n*** Other info ***\n\n"

        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Significance (10 GeV - 2 TeV)", d["Signif_Avg"])
        ss += "{:<32s} : {:.1f}\n".format("Npred", d["Npred"])

        ss += "\n{:<16s} : {:.3f} {}\n".format(
            "HEP Energy", d["HEP_Energy"].value, d["HEP_Energy"].unit
        )
        ss += "{:<16s} : {:.3f}\n".format("HEP Probability", d["HEP_Prob"])

        ss += "{:<16s} : {}\n".format("Bayesian Blocks", d["Variability_BayesBlocks"])

        ss += "{:<16s} : {:.3f}\n".format("Redshift", d["Redshift"])
        ss += "{:<16s} : {:.3} {}\n".format(
            "NuPeak_obs", d["NuPeak_obs"].value, d["NuPeak_obs"].unit
        )

        return ss

    def spectral_model(self):
        """Best fit spectral model (`~gammapy.modeling.models.SpectralModel`)."""
        d = self.data
        spec_type = self.data["SpectrumType"].strip()

        if spec_type == "PowerLaw":
            tag = "PowerLawSpectralModel"
            pars = {
                "reference": d["Pivot_Energy"],
                "amplitude": d["Flux_Density"],
                "index": d["PowerLaw_Index"],
            }
            errs = {
                "amplitude": d["Unc_Flux_Density"],
                "index": d["Unc_PowerLaw_Index"],
            }
        elif spec_type == "LogParabola":
            tag = "LogParabolaSpectralModel"
            pars = {
                "reference": d["Pivot_Energy"],
                "amplitude": d["Flux_Density"],
                "alpha": d["Spectral_Index"],
                "beta": d["beta"],
            }
            errs = {
                "amplitude": d["Unc_Flux_Density"],
                "alpha": d["Unc_Spectral_Index"],
                "beta": d["Unc_beta"],
            }
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model = Model.create(tag, "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._energy_edges[:-1]
        table["e_max"] = self._energy_edges[1:]

        flux = self.data["Flux_Band"]
        flux_err = self.data["Unc_Flux_Band"]
        e2dnde = self.data["nuFnu"]

        table["flux"] = flux
        table["flux_errn"] = np.abs(flux_err[:, 0])
        table["flux_errp"] = flux_err[:, 1]

        table["e2dnde"] = e2dnde
        table["e2dnde_errn"] = np.abs(e2dnde * flux_err[:, 0] / flux)
        table["e2dnde_errp"] = e2dnde * flux_err[:, 1] / flux

        is_ul = np.isnan(table["flux_errn"])
        table["is_ul"] = is_ul

        # handle upper limits
        table["flux_ul"] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table["flux"], table["flux_errp"])
        table["flux_ul"][is_ul] = flux_ul[is_ul]

        table["e2dnde_ul"] = np.nan * e2dnde.unit
        e2dnde_ul = compute_flux_points_ul(table["e2dnde"], table["e2dnde_errp"])
        table["e2dnde_ul"][is_ul] = e2dnde_ul[is_ul]

        # Square root of test statistic
        table["sqrt_ts"] = self.data["Sqrt_TS_Band"]
        return FluxPoints(table)

    def spatial_model(self):
        """Source spatial model (`~gammapy.modeling.models.SpatialModel`)."""
        d = self.data
        ra = d["RAJ2000"]
        dec = d["DEJ2000"]

        if self.is_pointlike:
            model = PointSpatialModel(lon_0=ra, lat_0=dec, frame="icrs")
        else:
            de = self.data_extended
            morph_type = de["Spatial_Function"].strip()
            e = (1 - (de["Model_SemiMinor"] / de["Model_SemiMajor"]) ** 2.0) ** 0.5
            sigma = de["Model_SemiMajor"]
            phi = de["Model_PosAng"]
            if morph_type == "RadialDisk":
                r_0 = de["Model_SemiMajor"]
                model = DiskSpatialModel(
                    lon_0=ra, lat_0=dec, r_0=r_0, e=e, phi=phi, frame="icrs"
                )
            elif morph_type in ["SpatialMap"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/"
                )
                model = TemplateSpatialModel.read(path / filename)
            elif morph_type == "RadialGauss":
                model = GaussianSpatialModel(
                    lon_0=ra, lat_0=dec, sigma=sigma, e=e, phi=phi, frame="icrs"
                )
            else:
                raise ValueError(f"Invalid morph_type: {morph_type!r}")
        self._set_spatial_errors(model)
        return model


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.

    - https://ui.adsabs.harvard.edu/#abs/2015ApJS..218...23A
    - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FGL`.
    """

    tag = "3fgl"
    description = "LAT 4-year point source catalog"
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psc_v16.fit.gz"):
        filename = make_path(filename)

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="LAT_Point_Source_Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = (
            "Extended_Source_Name",
            "0FGL_Name",
            "1FGL_Name",
            "2FGL_Name",
            "1FHL_Name",
            "ASSOC_TEV",
            "ASSOC1",
            "ASSOC2",
        )
        super().__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="ExtendedSources")


class SourceCatalog4FGL(SourceCatalog):
    """Fermi-LAT 4FGL source catalog.

    - https://arxiv.org/abs/1902.10045
    - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/8yr_catalog/

    One source is represented by `~gammapy.catalog.SourceCatalogObject4FGL`.
    """

    tag = "4fgl"
    description = "LAT 8-year point source catalog"
    source_object_class = SourceCatalogObject4FGL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz"):
        filename = make_path(filename)
        table = Table.read(filename, hdu="LAT_Point_Source_Catalog")
        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = (
            "Extended_Source_Name",
            "ASSOC_FGL",
            "ASSOC_FHL",
            "ASSOC_GAM1",
            "ASSOC_GAM2",
            "ASSOC_GAM3",
            "ASSOC_TEV",
            "ASSOC1",
            "ASSOC2",
        )
        super().__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="ExtendedSources")


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.

    - https://ui.adsabs.harvard.edu/abs/2016ApJS..222....5A
    - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/2FHL/

    One source is represented by `~gammapy.catalog.SourceCatalogObject2FHL`.
    """

    tag = "2fhl"
    description = "LAT second high-energy source catalog"
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psch_v09.fit.gz"):
        filename = make_path(filename)

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="2FHL Source Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = ("ASSOC", "3FGL_Name", "1FHL_Name", "TeVCat_Name")
        super().__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="Extended Sources")
        self.rois = Table.read(filename, hdu="ROIs")


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.

    - https://ui.adsabs.harvard.edu/abs/2017ApJS..232...18A
    - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/3FHL/

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FHL`.
    """

    tag = "3fhl"
    description = "LAT third high-energy source catalog"
    source_object_class = SourceCatalogObject3FHL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psch_v13.fit.gz"):
        filename = make_path(filename)

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="LAT_Point_Source_Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = ("ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM")
        super().__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="ExtendedSources")
        self.rois = Table.read(filename, hdu="ROIs")
        self.energy_bounds_table = Table.read(filename, hdu="EnergyBounds")
