# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes."""
import abc
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Column, Table
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
from gammapy.maps import Map
from gammapy.modeling.models import (
    DiskSpatialModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    GaussianSpatialModel,
    LogParabolaSpectralModel,
    PointSpatialModel,
    PowerLaw2SpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    TemplateSpatialModel,
)
from gammapy.spectrum import FluxPoints
from gammapy.time import LightCurve
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_standardise_units_inplace
from gammapy.utils.gauss import Gauss2DPDF
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


class SourceCatalogObjectFermiBase(SourceCatalogObject):
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
        """Print basic info."""
        d = self.data
        keys = self.asso
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(d["catalog_row_index"])
        ss += "{:<20s} : {}\n".format("Source name", d["Source_Name"])
        try:
            ss += "{:<20s} : {}\n".format("Extended name", d["Extended_Source_Name"])
        except (KeyError):
            pass

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
        try:
            tevcat_flag = d["TEVCAT_FLAG"]
            if tevcat_flag == "N":
                tevcat_message = "No TeV association"
            elif tevcat_flag == "P":
                tevcat_message = "Small TeV source"
            elif tevcat_flag == "E":
                tevcat_message = "Extended TeV source (diameter > 40 arcmins)"
            else:
                tevcat_message = "N/A"
            ss += "{:<16s} : {}\n".format("TeVCat flag", tevcat_message)
        except (KeyError):
            pass
        return ss

    @abc.abstractmethod
    def _info_more(self):
        return "\n"

    def _info_position(self):
        """Print position info."""
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

    @abc.abstractmethod
    def _info_spectral_fit(self):
        pass

    def _info_spectral_points(self):
        """Print spectral points."""
        ss = "\n*** Spectral points ***\n\n"
        lines = self.flux_points.table_formatted.pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)
        return ss

    @abc.abstractmethod
    def _info_lightcurve(self):
        pass

    @property
    def is_pointlike(self):
        return self.data["Extended_Source_Name"].strip() == ""

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
            phi_0 = d["Conf_95_PosAng"].to("deg")

        if np.isnan(phi_0):
            phi_0 = 0.0 * u.deg

        scale_1sigma = Gauss2DPDF().containment_radius(percent)
        lat_err = semi_major.to("deg") / scale_1sigma
        lon_err = semi_minor.to("deg") / scale_1sigma / np.cos(d["DEJ2000"].to("rad"))
        model.parameters.set_parameter_errors(dict(lon_0=lon_err, lat_0=lat_err))
        model.phi_0 = phi_0

    def sky_model(self):
        """Sky model (`~gammapy.modeling.models.SkyModel`)."""
        return SkyModel(self.spatial_model(), self.spectral_model(), name=self.name)


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
    _ebounds = u.Quantity([50, 100, 300, 1000, 3000, 10000, 30000, 300000], "MeV")

    def _info_more(self):
        """Print other info."""
        d = self.data
        ss = "\n*** Other info ***\n\n"
        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Significance (100 MeV - 1 TeV)", d["Signif_Avg"])
        ss += "{:<32s} : {:.1f}\n".format("Npred", d["Npred"])

        flag_message = {
            0: "None",
            1: "Source with T S > 35 which went to T S < 25 when changing the diffuse model "
            "(see Sec. 3.7.1 in catalog paper) or the analysis method (see Sec. 3.7.2 in catalog paper). "
            "Sources with T S ≤ 35 are not flagged with this bit because normal statistical fluctuations can push them to T S < 25.",
            2: "Moved beyond its 95% error ellipse when changing the diffuse model. ",
            3: "Flux (> 1 GeV) or energy flux (> 100 MeV) changed by more than 3σ when "
            "changing the diffuse model or the analysis method. Requires also that the flux "
            "change by more than 35% (to not flag strong sources).",
            4: "Source-to-background ratio less than 10% in highest band in which TS > 25. Background is integrated "
            "over the 68%-confidence area (pi*r_682) or 1 square degree, whichever is smaller.",
            5: "Closer than theta_ref from a brighter neighbor, where theta_ref is defined in the highest band in which "
            " source TS > 25, or the band with highest TS if all are < 25. theta_ref is set to 3.77 degrees (FWHM) below 100 MeV, "
            "1.68 degrees between 100 and 300 MeV , 1.03 degrees between 300 MeV and 1 GeV, "
            "0. 76 degree between 1 and 3 GeV (in-between FWHM and 2*r_68), "
            "0.49 degree between 3 and 10 GeV and 0.25 degree above 10 GeV (2*r_68).",
            6: "On top of an interstellar gas clump or small-scale defect in the model of diffuse emission. This flag "
            'is equivalent to the "c" suffix in the source name (see Sec. 3.7.1 in catalog paper).',
            9: "Localization Quality > 8 in pointlike (see Section 3.1 in catalog paper) or long axis of 95% ellipse > 0.25.",
            10: "Total Spectral Fit Quality > 20  or Spectral Fit Quality > 9 in any band (see Equation 5 in catalog paper).",
            12: "Highly curved spectrum; LogParabolaSpectralModel beta fixed to 1 or PLExpCutoff Spectral Index fixed to 0 (see "
            "Section 3.3 in catalog paper).",
        }
        ss += "\n{:<20s} : {}\n".format(
            "Other flags", flag_message.get(d["Flags"], "N/A")
        )

        return ss

    def _info_spectral_fit(self):
        """Print spectral info."""
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
        """Print lightcurve info."""
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
            sigma = de["Model_SemiMajor"].to("deg")
            phi = de["Model_PosAng"].to("deg")
            if morph_type == "Disk":
                r_0 = de["Model_SemiMajor"].to("deg")
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

        pars, errs = {}, {}
        pars["reference"] = self.data["Pivot_Energy"]

        if spec_type == "PowerLaw":
            pars["amplitude"] = self.data["PL_Flux_Density"]
            pars["index"] = self.data["PL_Index"]
            errs["amplitude"] = self.data["Unc_PL_Flux_Density"]
            errs["index"] = self.data["Unc_PL_Index"]
            model = PowerLawSpectralModel(**pars)
        elif spec_type == "LogParabola":
            pars["amplitude"] = self.data["LP_Flux_Density"]
            pars["alpha"] = self.data["LP_Index"]
            pars["beta"] = self.data["LP_beta"]
            errs["amplitude"] = self.data["Unc_LP_Flux_Density"]
            errs["alpha"] = self.data["Unc_LP_Index"]
            errs["beta"] = self.data["Unc_LP_beta"]
            model = LogParabolaSpectralModel(**pars)
        elif spec_type == "PLSuperExpCutoff":
            pars["amplitude"] = self.data["PLEC_Flux_Density"]
            pars["index_1"] = self.data["PLEC_Index"]
            pars["index_2"] = self.data["PLEC_Exp_Index"]
            pars["expfactor"] = self.data["PLEC_Expfactor"]
            errs["amplitude"] = self.data["Unc_PLEC_Flux_Density"]
            errs["index_1"] = self.data["Unc_PLEC_Index"]
            errs["index_2"] = np.nan_to_num(self.data["Unc_PLEC_Exp_Index"])
            errs["expfactor"] = self.data["Unc_PLEC_Expfactor"]
            model = SuperExpCutoffPowerLaw4FGLSpectralModel(**pars)
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"

        table["e_min"] = self._ebounds[:-1]
        table["e_max"] = self._ebounds[1:]

        flux = self._get_flux_values("Flux_Band")
        flux_err = self._get_flux_values("Unc_Flux_Band")
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
        table["sqrt_TS"] = self.data["Sqrt_TS_Band"]
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = self.data[prefix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.time.LightCurve`)."""
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

    _ebounds = u.Quantity([100, 300, 1000, 3000, 10000, 100000], "MeV")
    _ebounds_suffix = ["100_300", "300_1000", "1000_3000", "3000_10000", "10000_100000"]
    energy_range = u.Quantity([100, 100000], "MeV")
    """Energy range used for the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def _info_more(self):
        """Print other info."""
        d = self.data
        ss = "\n*** Other info ***\n\n"
        flag_message = {
            0: "None",
            1: "Source with TS > 35 which went to TS < 25 when changing the diffuse model. Note that sources with TS < "
            "35 are not flagged with this bit because normal statistical fluctuations can push them to TS < 25.",
            3: "Flux (> 1 GeV) or energy flux (> 100 MeV) changed by more than 3 sigma when changing the diffuse model."
            " Requires also that the flux change by more than 35% (to not flag strong sources).",
            4: "Source-to-background ratio less than 10% in highest band in which TS > 25. Background is integrated "
            "over the 68%-confidence area (pi*r_682) or 1 square degree, whichever is smaller.",
            5: "Closer than theta_ref from a brighter neighbor, where theta_ref is defined in the highest band in which"
            " source TS > 25, or the band with highest TS if all are < 25. theta_ref is set to 2.17 degrees (FWHM)"
            " below 300 MeV, 1.38 degrees between 300 MeV and 1 GeV, 0.87 degrees between 1 GeV and 3 GeV, 0.67"
            " degrees between 3 and 10 GeV and 0.45 degrees about 10 GeV (2*r_68).",
            6: "On top of an interstellar gas clump or small-scale defect in the model of diffuse emission. This flag "
            'is equivalent to the "c" suffix in the source name.',
            7: "Unstable position determination; result from gtfindsrc outside the 95% ellipse from pointlike.",
            9: "Localization Quality > 8 in pointlike (see Section 3.1 in catalog paper) or long axis of 95% ellipse >"
            " 0.25.",
            10: "Spectral Fit Quality > 16.3 (see Equation 3 in 2FGL catalog paper).",
            11: "Possibly due to the Sun (see Section 3.6 in catalog paper).",
            12: "Highly curved spectrum; LogParabolaSpectralModel beta fixed to 1 or PLExpCutoff Spectral Index fixed to 0 (see "
            "Section 3.3 in catalog paper).",
        }
        ss += "{:<20s} : {}\n".format(
            "Other flags", flag_message.get(d["Flags"], "N/A")
        )

        return ss

    def _info_spectral_fit(self):
        """Print spectral info."""
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
        """Print lightcurve info."""
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

        pars, errs = {}, {}
        pars["amplitude"] = self.data["Flux_Density"]
        errs["amplitude"] = self.data["Unc_Flux_Density"]
        pars["reference"] = self.data["Pivot_Energy"]

        if spec_type == "PowerLaw":
            pars["index"] = self.data["Spectral_Index"]
            errs["index"] = self.data["Unc_Spectral_Index"]
            model = PowerLawSpectralModel(**pars)
        elif spec_type == "PLExpCutoff":
            pars["index"] = self.data["Spectral_Index"]
            pars["ecut"] = self.data["Cutoff"]
            errs["index"] = self.data["Unc_Spectral_Index"]
            errs["ecut"] = self.data["Unc_Cutoff"]
            model = ExpCutoffPowerLaw3FGLSpectralModel(**pars)
        elif spec_type == "LogParabola":
            pars["alpha"] = self.data["Spectral_Index"]
            pars["beta"] = self.data["beta"]
            errs["alpha"] = self.data["Unc_Spectral_Index"]
            errs["beta"] = self.data["Unc_beta"]
            model = LogParabolaSpectralModel(**pars)
        elif spec_type == "PLSuperExpCutoff":
            # TODO: why convert to GeV here? Remove?
            pars["reference"] = pars["reference"].to("GeV")
            pars["index_1"] = self.data["Spectral_Index"]
            pars["index_2"] = self.data["Exp_Index"]
            pars["ecut"] = self.data["Cutoff"].to("GeV")
            errs["index_1"] = self.data["Unc_Spectral_Index"]
            errs["index_2"] = self.data["Unc_Exp_Index"]
            errs["ecut"] = self.data["Unc_Cutoff"].to("GeV")
            model = SuperExpCutoffPowerLaw3FGLSpectralModel(**pars)
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model.parameters.set_parameter_errors(errs)
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
            sigma = de["Model_SemiMajor"].to("deg")
            phi = de["Model_PosAng"].to("deg")
            if morph_type == "Disk":
                r_0 = de["Model_SemiMajor"].to("deg")
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
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"

        table["e_min"] = self._ebounds[:-1]
        table["e_max"] = self._ebounds[1:]

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
        table["sqrt_TS"] = [self.data["Sqrt_TS" + _] for _ in self._ebounds_suffix]
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.time.LightCurve`)."""
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
    _ebounds = u.Quantity([50, 171, 585, 2000], "GeV")
    _ebounds_suffix = ["50_171", "171_585", "585_2000"]
    energy_range = u.Quantity([0.05, 2], "TeV")
    """Energy range used for the catalog."""

    def _info_more(self):
        """Print other info."""
        d = self.data
        ss = "\n*** Other info ***\n\n"
        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Test statistic (50 GeV - 2 TeV)", d["TS"])
        return ss

    def _info_position(self):
        """Print position info."""
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
        """Print model data."""
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
            sigma = de["Model_SemiMajor"].to("deg")
            phi = de["Model_PosAng"].to("deg")
            if morph_type in ["Disk", "Elliptical Disk"]:
                r_0 = de["Model_SemiMajor"].to("deg")
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
        pars, errs = {}, {}
        pars["amplitude"] = self.data["Flux50"]
        pars["emin"], pars["emax"] = self.energy_range
        pars["index"] = self.data["Spectral_Index"]

        errs["amplitude"] = self.data["Unc_Flux50"]
        errs["index"] = self.data["Unc_Spectral_Index"]

        model = PowerLaw2SpectralModel(**pars)
        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._ebounds[:-1]
        table["e_max"] = self._ebounds[1:]
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
        values = [self.data[prefix + _ + "GeV"] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    def _info_lightcurve(self):
        return "\n"


class SourceCatalogObject3FHL(SourceCatalogObjectFermiBase):
    """One source from the Fermi-LAT 3FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FHL`.
    """

    asso = ["ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM"]
    energy_range = u.Quantity([0.01, 2], "TeV")
    """Energy range used for the catalog."""

    _ebounds = u.Quantity([10, 20, 50, 150, 500, 2000], "GeV")

    def _info_position(self):
        """Print position info."""
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
        """Print model data."""
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
        """Print other info."""
        d = self.data
        ss = "\n*** Other info ***\n\n"

        fmt = "{:<32s} : {:.3f}\n"
        ss += fmt.format("Significance (10 GeV - 2 TeV)", d["Signif_Avg"])
        ss += "{:<32s} : {:.1f}\n".format("Npred", d["Npred"])

        ss += "\n{:<16s} : {:.3f} {}\n".format(
            "HEP Energy", d["HEP_Energy"].value, d["HEP_Energy"].unit
        )
        ss += "{:<16s} : {:.3f}\n".format("HEP Probability", d["HEP_Prob"])

        # This is the number of Bayesian blocks for most sources,
        # except -1 means "could not be tested"
        msg = d["Variability_BayesBlocks"]
        if msg == 1:
            msg = "1 (not variable)"
        elif msg == -1:
            msg = "Could not be tested"
        ss += "{:<16s} : {}\n".format("Bayesian Blocks", msg)

        ss += "{:<16s} : {:.3f}\n".format("Redshift", d["Redshift"])
        ss += "{:<16s} : {:.3} {}\n".format(
            "NuPeak_obs", d["NuPeak_obs"].value, d["NuPeak_obs"].unit
        )

        return ss

    def spectral_model(self):
        """Best fit spectral model (`~gammapy.modeling.models.SpectralModel`)."""
        d = self.data
        spec_type = self.data["SpectrumType"].strip()

        pars, errs = {}, {}
        pars["amplitude"] = d["Flux_Density"]
        errs["amplitude"] = d["Unc_Flux_Density"]
        pars["reference"] = d["Pivot_Energy"]

        if spec_type == "PowerLaw":
            pars["index"] = d["PowerLaw_Index"]
            errs["index"] = d["Unc_PowerLaw_Index"]
            model = PowerLawSpectralModel(**pars)
        elif spec_type == "LogParabola":
            pars["alpha"] = d["Spectral_Index"]
            pars["beta"] = d["beta"]
            errs["alpha"] = d["Unc_Spectral_Index"]
            errs["beta"] = d["Unc_beta"]
            model = LogParabolaSpectralModel(**pars)
        else:
            raise ValueError(f"Invalid spec_type: {spec_type!r}")

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._ebounds[:-1]
        table["e_max"] = self._ebounds[1:]

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
            sigma = de["Model_SemiMajor"].to("deg")
            phi = de["Model_PosAng"].to("deg")
            if morph_type == "RadialDisk":
                r_0 = de["Model_SemiMajor"].to("deg")
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

    def _info_lightcurve(self):
        return "\n"


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.

    Reference: https://ui.adsabs.harvard.edu/#abs/2015ApJS..218...23A

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FGL`.
    """

    name = "3fgl"
    description = "LAT 4-year point source catalog"
    source_object_class = SourceCatalogObject3FGL
    source_categories = {
        "galactic": ["psr", "pwn", "snr", "spp", "glc"],
        "extra-galactic": [
            "css",
            "bll",
            "fsrq",
            "agn",
            "nlsy1",
            "rdg",
            "sey",
            "bcu",
            "gal",
            "sbg",
            "ssrq",
        ],
        "GALACTIC": ["PSR", "PWN", "SNR", "HMB", "BIN", "NOV", "SFR"],
        "EXTRA-GALACTIC": [
            "CSS",
            "BLL",
            "FSRQ",
            "AGN",
            "NLSY1",
            "RDG",
            "SEY",
            "BCU",
            "GAL",
            "SBG",
            "SSRQ",
        ],
        "unassociated": [""],
    }

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

    def is_source_class(self, source_class):
        """
        Check if source belongs to a given source class.

        The classes are described in Table 3 of the 3FGL paper:

        https://ui.adsabs.harvard.edu/abs/2015ApJS..218...23A

        Parameters
        ----------
        source_class : str
            Source class designator as defined in Table 3. There are a few extra
            selections available:

            - 'ALL': all identified objects
            - 'all': all objects with associations
            - 'galactic': all sources with an associated galactic object
            - 'GALACTIC': all identified galactic sources
            - 'extra-galactic': all sources with an associated extra-galactic object
            - 'EXTRA-GALACTIC': all identified extra-galactic sources
            - 'unassociated': all unassociated objects

        Returns
        -------
        selection : `~numpy.ndarray`
            Selection mask.
        """
        source_class_info = np.array([_.strip() for _ in self.table["CLASS1"]])

        cats = self.source_categories
        if source_class in cats:
            category = set(cats[source_class])
        elif source_class == "ALL":
            category = set(cats["EXTRA-GALACTIC"] + cats["GALACTIC"])
        elif source_class == "all":
            category = set(cats["extra-galactic"] + cats["galactic"])
        elif source_class in np.unique(source_class_info):
            category = {source_class}
        else:
            raise ValueError(f"Invalid source_class: {source_class!r}")

        return np.array([_ in category for _ in source_class_info])

    def select_source_class(self, source_class):
        """
        Select all sources of a given source class.

        See `SourceCatalog3FHL.is_source_class` for further documentation

        Parameters
        ----------
        source_class : str
            Source class designator.

        Returns
        -------
        selection : `SourceCatalog3FHL`
            Subset of the 3FHL catalog containing only the selected source class.
        """
        catalog = self.copy()
        selection = self.is_source_class(source_class)
        catalog.table = catalog.table[selection]
        return catalog


class SourceCatalog4FGL(SourceCatalog):
    """Fermi-LAT 4FGL source catalog.

    References:

    - https://arxiv.org/abs/1902.10045
    - https://fermi.gsfc.nasa.gov/ssc/data/access/lat/8yr_catalog/

    One source is represented by `~gammapy.catalog.SourceCatalogObject4FGL`.
    """

    name = "4fgl"
    description = "LAT 8-year point source catalog"
    source_object_class = SourceCatalogObject4FGL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psc_v19.fit.gz"):
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

    Reference: https://ui.adsabs.harvard.edu/abs/2016ApJS..222....5A

    One source is represented by `~gammapy.catalog.SourceCatalogObject2FHL`.
    """

    name = "2fhl"
    description = "LAT second high-energy source catalog"
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psch_v08.fit.gz"):
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

        self.counts_image = Map.read(filename, hdu="Count Map")
        self.extended_sources_table = Table.read(filename, hdu="Extended Sources")
        self.rois = Table.read(filename, hdu="ROIs")


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.

    Reference: https://ui.adsabs.harvard.edu/abs/2017ApJS..232...18A

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FHL`.
    """

    name = "3fhl"
    description = "LAT third high-energy source catalog"
    source_object_class = SourceCatalogObject3FHL
    source_categories = {
        "galactic": ["glc", "hmb", "psr", "pwn", "sfr", "snr", "spp"],
        "extra-galactic": ["agn", "bcu", "bll", "fsrq", "rdg", "sbg"],
        "GALACTIC": ["BIN", "HMB", "PSR", "PWN", "SFR", "SNR"],
        "EXTRA-GALACTIC": ["BLL", "FSRQ", "NLSY1", "RDG"],
        "unassociated": [""],
    }

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

    def is_source_class(self, source_class):
        """
        Check if source belongs to a given source class.

        The classes are described in Table 3 of the 3FGL paper:

        https://ui.adsabs.harvard.edu/abs/2015ApJS..218...23A

        Parameters
        ----------
        source_class : str
            Source class designator as defined in Table 3. There are a few extra
            selections available:

            - 'ALL': all identified objects
            - 'all': all objects with associations
            - 'galactic': all sources with an associated galactic object
            - 'GALACTIC': all identified galactic sources
            - 'extra-galactic': all sources with an associated extra-galactic object
            - 'EXTRA-GALACTIC': all identified extra-galactic sources
            - 'unassociated': all unassociated objects

        Returns
        -------
        selection : `~numpy.ndarray`
            Selection mask.
        """
        source_class_info = np.array([_.strip() for _ in self.table["CLASS"]])

        cats = self.source_categories
        if source_class in cats:
            category = set(cats[source_class])
        elif source_class == "ALL":
            category = set(cats["EXTRA-GALACTIC"] + cats["GALACTIC"])
        elif source_class == "all":
            category = set(cats["extra-galactic"] + cats["galactic"])
        elif source_class in np.unique(source_class_info):
            category = {source_class}
        else:
            raise ValueError(f"Invalid source_class: {source_class!r}")

        return np.array([_ in category for _ in source_class_info])

    def select_source_class(self, source_class):
        """
        Select all sources of a given source class.

        See `SourceCatalog3FHL.is_source_class` for further documentation

        Parameters
        ----------
        source_class : str
            Source class designator.

        Returns
        -------
        selection : `SourceCatalog3FHL`
            Subset of the 3FHL catalog containing only the selected source class.
        """
        catalog = self.copy()
        selection = self.is_source_class(source_class)
        catalog.table = catalog.table[selection]
        return catalog
