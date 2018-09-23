# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
import numpy as np
import astropy.units as u
from astropy.table import Table, Column
from astropy.time import Time
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds
from ..utils.table import table_standardise_units_inplace
from ..maps import Map
from ..spectrum import FluxPoints
from ..spectrum.models import (
    PowerLaw,
    PowerLaw2,
    ExponentialCutoffPowerLaw3FGL,
    PLSuperExpCutoff3FGL,
    LogParabola,
)
from ..image.models import SkyPointSource, SkyGaussian, SkyDisk, SkyDiffuseMap
from ..cube.models import SkyModel
from ..time import LightCurve
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalogObject3FGL",
    "SourceCatalogObject1FHL",
    "SourceCatalogObject2FHL",
    "SourceCatalogObject3FHL",
    "SourceCatalog3FGL",
    "SourceCatalog1FHL",
    "SourceCatalog2FHL",
    "SourceCatalog3FHL",
]


def compute_flux_points_ul(quantity, quantity_errp):
    """Compute UL value for fermi flux points.

    See https://arxiv.org/pdf/1501.02003.pdf (page 30)
    """
    return 2 * quantity_errp + quantity


class SourceCatalogObject3FGL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FGL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FGL`.
    """

    _ebounds = EnergyBounds([100, 300, 1000, 3000, 10000, 100000], "MeV")
    _ebounds_suffix = ["100_300", "300_1000", "1000_3000", "3000_10000", "10000_100000"]
    energy_range = u.Quantity([100, 100000], "MeV")
    """Energy range of the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral', 'lightcurve'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,spectral,lightcurve"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "spectral" in ops:
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()
        if "lightcurve" in ops:
            ss += self._info_lightcurve()
        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(d["catalog_row_index"])
        ss += "{:<20s} : {}\n".format("Source name", d["Source_Name"])
        ss += "{:<20s} : {}\n".format("Extended name", d["Extended_Source_Name"])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ", ".join([_ for _ in vals if _ != ""])

        keys = [
            "ASSOC1",
            "ASSOC2",
            "ASSOC_TEV",
            "ASSOC_GAM1",
            "ASSOC_GAM2",
            "ASSOC_GAM3",
        ]
        associations = get_nonentry_keys(keys)
        ss += "{:<20s} : {}\n".format("Associations", associations)

        keys = ["0FGL_Name", "1FGL_Name", "2FGL_Name", "1FHL_Name"]
        other_names = get_nonentry_keys(keys)
        ss += "{:<20s} : {}\n".format("Other names", other_names)

        ss += "{:<20s} : {}\n".format("Class", d["CLASS1"])

        tevcat_flag = d["TEVCAT_FLAG"]
        if tevcat_flag == "N":
            tevcat_message = "No TeV association"
        elif tevcat_flag == "P":
            tevcat_message = "Small TeV source"
        elif tevcat_flag == "E":
            tevcat_message = "Extended TeV source (diameter > 40 arcmins)"
        else:
            tevcat_message = "N/A"
        ss += "{:<20s} : {}\n".format("TeVCat flag", tevcat_message)

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
            12: "Highly curved spectrum; LogParabola beta fixed to 1 or PLExpCutoff Spectral Index fixed to 0 (see "
            "Section 3.3 in catalog paper).",
        }
        ss += "{:<20s} : {}\n".format(
            "Other flags", flag_message.get(d["Flags"], "N/A")
        )

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
        ss += "{:<20s} : {:.4f}\n".format("Semimajor (68%)", d["Conf_68_SemiMajor"])
        ss += "{:<20s} : {:.4f}\n".format("Semiminor (68%)", d["Conf_68_SemiMinor"])
        ss += "{:<20s} : {:.2f}\n".format("Position angle (68%)", d["Conf_68_PosAng"])
        ss += "{:<20s} : {:.4f}\n".format("Semimajor (95%)", d["Conf_95_SemiMajor"])
        ss += "{:<20s} : {:.4f}\n".format("Semiminor (95%)", d["Conf_95_SemiMinor"])
        ss += "{:<20s} : {:.2f}\n".format("Position angle (95%)", d["Conf_95_PosAng"])
        ss += "{:<20s} : {:.0f}\n".format("ROI number", d["ROI_num"])

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
            raise ValueError("Invalid spec_type")

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

    def _info_spectral_points(self):
        """Print spectral points."""
        ss = "\n*** Spectral points ***\n\n"
        lines = self._flux_points_table_formatted.pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)

        return ss + "\n"

    def _info_lightcurve(self):
        """Print lightcurve info."""
        d = self.data
        ss = "\n*** Lightcurve info ***\n\n"
        ss += "Lightcurve measured in the energy band: 100 MeV - 100 GeV\n\n"

        ss += "{:<15s} : {:.3f}\n".format("Variability index", d["Variability_Index"])

        if d["Signif_Peak"] == np.nan:
            ss += "{:<40s} : {:.3f}\n".format(
                "Significance peak (100 MeV - 100 GeV)", d["Signif_Peak"]
            )

            fmt = "{:<40s} : {:.3} +- {:.3} cm^-2 s^-1\n"
            ss += fmt.format(
                "Integral flux peak (100 MeV - 100 GeV)",
                d["Flux_Peak"],
                d["Unc_Flux_Peak"],
            )

            # TODO: give time as UTC string, not MET
            ss += "{:<40s} : {:.3} s (Mission elapsed time)\n".format(
                "Time peak", d["Time_Peak"]
            )
            peak_interval = d["Peak_Interval"].to("day").value
            ss += "{:<40s} : {:.3} day\n".format("Peak interval", peak_interval)
        else:
            ss += "\nNo peak measured for this source.\n"

        # TODO: Add a lightcurve table with d['Flux_History'] and d['Unc_Flux_History']

        return ss

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        spec_type = self.data["SpectrumType"].strip()

        pars, errs = {}, {}
        pars["amplitude"] = self.data["Flux_Density"]
        errs["amplitude"] = self.data["Unc_Flux_Density"]
        pars["reference"] = self.data["Pivot_Energy"]

        if spec_type == "PowerLaw":
            pars["index"] = self.data["Spectral_Index"] * u.dimensionless_unscaled
            errs["index"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled
            model = PowerLaw(**pars)
        elif spec_type == "PLExpCutoff":
            pars["index"] = self.data["Spectral_Index"] * u.dimensionless_unscaled
            pars["ecut"] = self.data["Cutoff"]
            errs["index"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled
            errs["ecut"] = self.data["Unc_Cutoff"]
            model = ExponentialCutoffPowerLaw3FGL(**pars)
        elif spec_type == "LogParabola":
            pars["alpha"] = self.data["Spectral_Index"] * u.dimensionless_unscaled
            pars["beta"] = self.data["beta"] * u.dimensionless_unscaled
            errs["alpha"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled
            errs["beta"] = self.data["Unc_beta"] * u.dimensionless_unscaled
            model = LogParabola(**pars)
        elif spec_type == "PLSuperExpCutoff":
            # TODO: why convert to GeV here? Remove?
            pars["reference"] = pars["reference"].to("GeV")
            pars["index_1"] = self.data["Spectral_Index"] * u.dimensionless_unscaled
            pars["index_2"] = self.data["Exp_Index"] * u.dimensionless_unscaled
            pars["ecut"] = self.data["Cutoff"].to("GeV")
            errs["index_1"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled
            errs["index_2"] = self.data["Unc_Exp_Index"] * u.dimensionless_unscaled
            errs["ecut"] = self.data["Unc_Cutoff"].to("GeV")
            model = PLSuperExpCutoff3FGL(**pars)
        else:
            raise ValueError("Invalid spec_type: {!r}".format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def spatial_model(self):
        """
        Source spatial model (`~gammapy.image.models.SkySpatialModel`).
        """
        d = self.data

        pars = {}
        glon = d["GLON"]
        glat = d["GLAT"]

        if self.is_pointlike:
            pars["lon_0"] = glon
            pars["lat_0"] = glat
            return SkyPointSource(lon_0=glon, lat_0=glat)
        else:
            de = self.data_extended
            morph_type = de["Model_Form"].strip()

            if morph_type == "Disk":
                r_0 = de["Model_SemiMajor"].to("deg")
                return SkyDisk(lon_0=glon, lat_0=glat, r_0=r_0)
            elif morph_type in ["Map", "Ring", "2D Gaussian x2"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v15/Templates/"
                )
                return SkyDiffuseMap.read(path / filename)
            elif morph_type == "2D Gaussian":
                # TODO: fill elongation info as soon as model supports it
                sigma = de["Model_SemiMajor"].to("deg")
                return SkyGaussian(lon_0=glon, lat_0=glat, sigma=sigma)
            else:
                raise ValueError("Invalid spatial model: {!r}".format(morph_type))

    @property
    def sky_model(self):
        """Source sky model (`~gammapy.cube.models.SkyModel`)."""
        spatial_model = self.spatial_model
        spectral_model = self.spectral_model
        return SkyModel(spatial_model, spectral_model)

    @property
    def is_pointlike(self):
        return self.data["Extended_Source_Name"].strip() == ""

    @property
    def _flux_points_table_formatted(self):
        """Returns formatted version of self.flux_points.table"""
        table = self.flux_points.table.copy()
        flux_cols = [
            "flux",
            "flux_errn",
            "flux_errp",
            "e2dnde",
            "e2dnde_errn",
            "e2dnde_errp",
            "flux_ul",
            "e2dnde_ul",
            "dnde",
        ]
        table["sqrt_TS"].format = ".1f"
        table["e_ref"].format = ".1f"
        for _ in flux_cols:
            table[_].format = ".3"

        return table

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        e_ref = self._ebounds.log_centers
        table["e_ref"] = e_ref
        table["e_min"] = self._ebounds.lower_bounds
        table["e_max"] = self._ebounds.upper_bounds

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

        table["dnde"] = (nuFnu * e_ref ** -2).to("TeV-1 cm-2 s-1")
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.time.LightCurve`).

        Examples
        --------

        >>> from gammapy.catalog import source_catalogs
        >>> source = source_catalogs['3fgl']['3FGL J0349.9-2102']
        >>> lc = source.lightcurve
        >>> lc.plot()
        """
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


class SourceCatalogObject1FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 1FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog1FHL`.
    """

    _ebounds = EnergyBounds([10, 30, 100, 500], "GeV")
    _ebounds_suffix = ["10_30", "30_100", "100_500"]
    energy_range = u.Quantity([0.01, 0.5], "TeV")
    """Energy range of the Fermi 1FHL source catalog"""

    def __str__(self):
        return self.info()

    def info(self):
        """Print summary info."""
        # TODO: can we share code with 3FGL summary function?
        d = self.data

        ss = "Source: {}\n".format(d["Source_Name"])
        ss += "\n"

        ss += "RA (J2000)  : {}\n".format(d["RAJ2000"])
        ss += "Dec (J2000) : {}\n".format(d["DEJ2000"])
        ss += "GLON        : {}\n".format(d["GLON"])
        ss += "GLAT        : {}\n".format(d["GLAT"])
        ss += "\n"

        # val, err = d['Energy_Flux100'], d['Unc_Energy_Flux100']
        # ss += 'Energy flux (100 MeV - 100 GeV) : {} +- {} erg cm^-2 s^-1\n'.format(val, err)
        # ss += 'Detection significance : {}\n'.format(d['Signif_Avg'])

        return ss

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _ + "GeV"] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._ebounds.lower_bounds
        table["e_max"] = self._ebounds.upper_bounds
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

        flux_points = FluxPoints(table)

        # TODO: change this and leave it up to the caller to convert to dnde
        # See https://github.com/gammapy/gammapy/issues/1034
        return flux_points.to_sed_type("dnde", model=self.spectral_model)

    @property
    def spectral_model(self):
        """Best fit spectral model `~gammapy.spectrum.models.SpectralModel`."""
        pars, errs = {}, {}
        pars["amplitude"] = self.data["Flux"]
        pars["emin"], pars["emax"] = self.energy_range
        pars["index"] = self.data["Spectral_Index"] * u.dimensionless_unscaled
        errs["amplitude"] = self.data["Unc_Flux"]
        errs["index"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled
        model = PowerLaw2(**pars)
        model.parameters.set_parameter_errors(errs)
        return model


class SourceCatalogObject2FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 2FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2FHL`.
    """

    _ebounds = EnergyBounds([50, 171, 585, 2000], "GeV")
    _ebounds_suffix = ["50_171", "171_585", "585_2000"]
    energy_range = u.Quantity([0.05, 2], "TeV")
    """Energy range of the Fermi 2FHL source catalog"""

    def __str__(self):
        return self.info()

    def info(self):
        """Print summary info."""
        # TODO: can we share code with 3FGL summary funtion?
        d = self.data

        ss = "Source: {}\n".format(d["Source_Name"])
        ss += "\n"

        ss += "RA (J2000)  : {}\n".format(d["RAJ2000"])
        ss += "Dec (J2000) : {}\n".format(d["DEJ2000"])
        ss += "GLON        : {}\n".format(d["GLON"])
        ss += "GLAT        : {}\n".format(d["GLAT"])
        ss += "\n"

        # val, err = d['Energy_Flux100'], d['Unc_Energy_Flux100']
        # ss += 'Energy flux (100 MeV - 100 GeV) : {} +- {} erg cm^-2 s^-1\n'.format(val, err)
        # ss += 'Detection significance : {}\n'.format(d['Signif_Avg'])

        return ss

    def _get_flux_values(self, prefix, unit="cm-2 s-1"):
        values = [self.data[prefix + _ + "GeV"] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        table["e_min"] = self._ebounds.lower_bounds
        table["e_max"] = self._ebounds.upper_bounds
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

        flux_points = FluxPoints(table)

        # TODO: change this and leave it up to the caller to convert to dnde
        # See https://github.com/gammapy/gammapy/issues/1034
        return flux_points.to_sed_type("dnde", model=self.spectral_model)

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        pars, errs = {}, {}
        pars["amplitude"] = self.data["Flux50"]
        pars["emin"], pars["emax"] = self.energy_range
        pars["index"] = self.data["Spectral_Index"] * u.dimensionless_unscaled

        errs["amplitude"] = self.data["Unc_Flux50"]
        errs["index"] = self.data["Unc_Spectral_Index"] * u.dimensionless_unscaled

        model = PowerLaw2(**pars)
        model.parameters.set_parameter_errors(errs)
        return model


class SourceCatalogObject3FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FHL`.
    """

    energy_range = u.Quantity([0.01, 2], "TeV")
    """Energy range of the Fermi 1FHL source catalog"""

    _ebounds = EnergyBounds([10, 20, 50, 150, 500, 2000], "GeV")

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,spectral,other"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
            if not self.is_pointlike:
                ss += self._info_morphology()
        if "spectral" in ops:
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()
        if "other" in ops:
            ss += self._info_other()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(d["catalog_row_index"])
        ss += "{:<20s} : {}\n".format("Source name", d["Source_Name"])
        ss += "{:<20s} : {}\n".format("Extended name", d["Extended_Source_Name"])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ", ".join([_ for _ in vals if _ != ""])

        keys = ["ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM"]
        associations = get_nonentry_keys(keys)
        ss += "{:<16s} : {}\n".format("Associations", associations)
        ss += "{:<16s} : {:.3f}\n".format("ASSOC_PROB_BAY", d["ASSOC_PROB_BAY"])
        ss += "{:<16s} : {:.3f}\n".format("ASSOC_PROB_LR", d["ASSOC_PROB_LR"])

        ss += "{:<16s} : {}\n".format("Class", d["CLASS"])

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

        fmt = "\n{:<32s} : {:.3f}\n"
        ss += fmt.format("Significance (10 GeV - 2 TeV)", d["Signif_Avg"])
        ss += "{:<32s} : {:.1f}\n".format("Npred", d["Npred"])

        return ss

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

    def _info_morphology(self):
        e = self.data_extended
        ss = "*** Extended source information ***\n"
        ss += "{:<16s} : {}\n".format("Model form", e["Model_Form"])
        ss += "{:<16s} : {:.4f}\n".format("Model semimajor", e["Model_SemiMajor"])
        ss += "{:<16s} : {:.4f}\n".format("Model semiminor", e["Model_SemiMinor"])
        ss += "{:<16s} : {:.4f}\n".format("Position angle", e["Model_PosAng"])
        ss += "{:<16s} : {}\n".format("Spatial function", e["Spatial_Function"])
        ss += "{:<16s} : {}\n\n".format("Spatial filename", e["Spatial_Filename"])
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
                "LogParabola spectral index",
                d["Spectral_Index"],
                d["Unc_Spectral_Index"],
            )

            ss += "{:<32s} : {:.3f} +- {:.3f}\n".format(
                "LogParabola beta", d["beta"], d["Unc_beta"]
            )
        else:
            raise ValueError("Invalid spec_type")

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

    def _info_spectral_points(self):
        """Print spectral points."""
        ss = "\n*** Spectral points ***\n\n"
        lines = self._flux_points_table_formatted.pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)
        return ss + "\n"

    def _info_other(self):
        """Print other info."""
        d = self.data
        ss = "\n*** Other info ***\n\n"
        ss += "{:<16s} : {:.3f} {}\n".format(
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

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        d = self.data
        spec_type = self.data["SpectrumType"].strip()

        pars, errs = {}, {}
        pars["amplitude"] = d["Flux_Density"]
        errs["amplitude"] = d["Unc_Flux_Density"]
        pars["reference"] = d["Pivot_Energy"]

        if spec_type == "PowerLaw":
            pars["index"] = d["PowerLaw_Index"] * u.dimensionless_unscaled
            errs["index"] = d["Unc_PowerLaw_Index"] * u.dimensionless_unscaled
            model = PowerLaw(**pars)
        elif spec_type == "LogParabola":
            pars["alpha"] = d["Spectral_Index"] * u.dimensionless_unscaled
            pars["beta"] = d["beta"] * u.dimensionless_unscaled
            errs["alpha"] = d["Unc_Spectral_Index"] * u.dimensionless_unscaled
            errs["beta"] = d["Unc_beta"] * u.dimensionless_unscaled
            model = LogParabola(**pars)
        else:
            raise ValueError("Invalid spec_type: {!r}".format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def _flux_points_table_formatted(self):
        """Returns formatted version of self.flux_points.table"""
        table = self.flux_points.table.copy()
        flux_cols = [
            "flux",
            "flux_errn",
            "flux_errp",
            "e2dnde",
            "e2dnde_errn",
            "e2dnde_errp",
            "flux_ul",
            "e2dnde_ul",
            "dnde",
        ]
        table["sqrt_ts"].format = ".1f"
        table["e_ref"].format = ".1f"
        for _ in flux_cols:
            table[_].format = ".3"

        return table

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "flux"
        e_ref = self._ebounds.log_centers
        table["e_ref"] = e_ref
        table["e_min"] = self._ebounds.lower_bounds
        table["e_max"] = self._ebounds.upper_bounds

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

        # TODO: remove this computation here.
        # # Instead provide a method on the FluxPoints class like `to_dnde()` or something.
        table["dnde"] = (e2dnde * e_ref ** -2).to("cm-2 s-1 TeV-1")

        return FluxPoints(table)

    @property
    def spatial_model(self):
        """Source spatial model (`~gammapy.image.models.SkySpatialModel`)."""
        d = self.data

        pars = {}
        glon = d["GLON"]
        glat = d["GLAT"]

        if self.is_pointlike:
            pars["lon_0"] = glon
            pars["lat_0"] = glat
            return SkyPointSource(lon_0=glon, lat_0=glat)
        else:
            de = self.data_extended
            morph_type = de["Spatial_Function"].strip()

            if morph_type == "RadialDisk":
                r_0 = de["Model_SemiMajor"].to("deg")
                return SkyDisk(lon_0=glon, lat_0=glat, r_0=r_0)
            elif morph_type in ["SpatialMap"]:
                filename = de["Spatial_Filename"].strip()
                path = make_path(
                    "$GAMMAPY_DATA/catalogs/fermi/Extended_archive_v18/Templates/"
                )
                return SkyDiffuseMap.read(path / filename)
            elif morph_type == "RadialGauss":
                # TODO: fill elongation info as soon as model supports it
                sigma = de["Model_SemiMajor"].to("deg")
                return SkyGaussian(lon_0=glon, lat_0=glat, sigma=sigma)
            else:
                raise ValueError("Invalid morph_type: {!r}".format(morph_type))

    @property
    def sky_model(self):
        """Source sky model (`~gammapy.cube.models.SkyModel`)."""
        spatial_model = self.spatial_model
        spectral_model = self.spectral_model
        return SkyModel(spatial_model, spectral_model)

    @property
    def is_pointlike(self):
        return self.data["Extended_Source_Name"].strip() == ""


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
        filename = str(make_path(filename))

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
        super(SourceCatalog3FGL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="ExtendedSources")

    def is_source_class(self, source_class):
        """
        Check if source belongs to a given source class.

        The classes are described in Table 3 of the 3FGL paper:

        http://adsabs.harvard.edu/abs/2015arXiv150102003T

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
            category = set([source_class])
        else:
            raise ValueError("Invalid source_class: {!r}".format(source_class))

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


class SourceCatalog1FHL(SourceCatalog):
    """Fermi-LAT 1FHL source catalog.

    Reference: http://adsabs.harvard.edu/abs/2013ApJS..209...34A

    One source is represented by `~gammapy.catalog.SourceCatalogObject1FHL`.
    """

    name = "1fhl"
    description = "First Fermi-LAT Catalog of Sources above 10 GeV"
    source_object_class = SourceCatalogObject1FHL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psch_v07.fit.gz"):
        filename = str(make_path(filename))

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="LAT_Point_Source_Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = ("ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM")
        super(SourceCatalog1FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu="ExtendedSources")


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.

    Reference: http://adsabs.harvard.edu/abs/2016ApJS..222....5A

    One source is represented by `~gammapy.catalog.SourceCatalogObject2FHL`.
    """

    name = "2fhl"
    description = "LAT second high-energy source catalog"
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/fermi/gll_psch_v08.fit.gz"):
        filename = str(make_path(filename))

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="2FHL Source Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = ("ASSOC", "3FGL_Name", "1FHL_Name", "TeVCat_Name")
        super(SourceCatalog2FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.counts_image = Map.read(filename, hdu="Count Map")
        self.extended_sources_table = Table.read(filename, hdu="Extended Sources")
        self.rois = Table.read(filename, hdu="ROIs")


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.

    Reference: http://adsabs.harvard.edu/abs/2017ApJS..232...18A

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
        filename = str(make_path(filename))

        with warnings.catch_warnings():  # ignore FITS units warnings
            warnings.simplefilter("ignore", u.UnitsWarning)
            table = Table.read(filename, hdu="LAT_Point_Source_Catalog")

        table_standardise_units_inplace(table)

        source_name_key = "Source_Name"
        source_name_alias = ("ASSOC1", "ASSOC2", "ASSOC_TEV", "ASSOC_GAM")
        super(SourceCatalog3FHL, self).__init__(
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

        http://adsabs.harvard.edu/abs/2015arXiv150102003T

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
            category = set([source_class])
        else:
            raise ValueError("Invalid source_class: {!r}".format(source_class))

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
