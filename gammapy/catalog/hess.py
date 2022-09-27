# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HESS Galactic plane survey (HGPS) catalog."""
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import Model, Models, SkyModel
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_row_to_dict
from .core import SourceCatalog, SourceCatalogObject, format_flux_points_table

__all__ = [
    "SourceCatalogHGPS",
    "SourceCatalogLargeScaleHGPS",
    "SourceCatalogObjectHGPS",
    "SourceCatalogObjectHGPSComponent",
]

# Flux factor, used for printing
FF = 1e-12

# Multiplicative factor to go from cm^-2 s^-1 to % Crab for integral flux > 1 TeV
# Here we use the same Crab reference that's used in the HGPS paper
# CRAB = crab_integral_flux(energy_min=1, reference='hess_ecpl')
FLUX_TO_CRAB = 100 / 2.26e-11
FLUX_TO_CRAB_DIFF = 100 / 3.5060459323111307e-11


class SourceCatalogObjectHGPSComponent(SourceCatalogObject):
    """One Gaussian component from the HGPS catalog.

    See Also
    --------
    SourceCatalogHGPS, SourceCatalogObjectHGPS
    """

    _source_name_key = "Component_ID"

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        """Pretty-print source data"""
        d = self.data
        ss = "Component {}:\n".format(d["Component_ID"])
        fmt = "{:<20s} : {:8.3f} +/- {:.3f} deg\n"
        ss += fmt.format("GLON", d["GLON"].value, d["GLON_Err"].value)
        ss += fmt.format("GLAT", d["GLAT"].value, d["GLAT_Err"].value)
        fmt = "{:<20s} : {:.3f} +/- {:.3f} deg\n"
        ss += fmt.format("Size", d["Size"].value, d["Size_Err"].value)
        val, err = d["Flux_Map"].value, d["Flux_Map_Err"].value
        fmt = "{:<20s} : ({:.2f} +/- {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} +/- {:.1f}) % Crab"
        ss += fmt.format(
            "Flux (>1 TeV)", val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB
        )
        return ss

    @property
    def name(self):
        """Source name (str)"""
        return self.data[self._source_name_key]

    def spatial_model(self):
        """Component spatial model (`~gammapy.modeling.models.GaussianSpatialModel`)."""
        d = self.data
        tag = "GaussianSpatialModel"
        pars = {
            "lon_0": d["GLON"],
            "lat_0": d["GLAT"],
            "sigma": d["Size"],
            "frame": "galactic",
        }
        errs = {"lon_0": d["GLON_Err"], "lat_0": d["GLAT_Err"], "sigma": d["Size_Err"]}
        model = Model.create(tag, "spatial", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model


class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog.

    The catalog is represented by `SourceCatalogHGPS`.
    Examples are given there.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

    def __str__(self):
        return self.info()

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        reference_model = SkyModel(spectral_model=self.spectral_model(), name=self.name)
        return FluxPoints.from_table(
            self.flux_points_table,
            reference_model=reference_model,
        )

    def info(self, info="all"):
        """Info string.

        Parameters
        ----------
        info : {'all', 'basic', 'map', 'spec', 'flux_points', 'components', 'associations', 'id'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,associations,id,map,spec,flux_points,components"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "map" in ops:
            ss += self._info_map()
        if "spec" in ops:
            ss += self._info_spec()
        if "flux_points" in ops:
            ss += self._info_flux_points()
        if "components" in ops and hasattr(self, "components"):
            ss += self._info_components()
        if "associations" in ops:
            ss += self._info_associations()
        if "id" in ops and hasattr(self, "identification_info"):
            ss += self._info_id()
        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(self.row_index)
        ss += "{:<20s} : {}\n".format("Source name", self.name)

        ss += "{:<20s} : {}\n".format("Analysis reference", d["Analysis_Reference"])
        ss += "{:<20s} : {}\n".format("Source class", d["Source_Class"])
        ss += "{:<20s} : {}\n".format("Identified object", d["Identified_Object"])
        ss += "{:<20s} : {}\n".format("Gamma-Cat id", d["Gamma_Cat_Source_ID"])
        ss += "\n"
        return ss

    def _info_associations(self):
        ss = "\n*** Source associations info ***\n\n"
        lines = self.associations.pformat(max_width=-1, max_lines=-1)
        ss += "\n".join(lines)
        return ss + "\n"

    def _info_id(self):
        ss = "\n*** Source identification info ***\n\n"
        ss += "\n".join(f"{k}: {v}" for k, v in self.identification_info.items())
        return ss + "\n"

    def _info_map(self):
        """Print info from map analysis."""
        d = self.data
        ss = "\n*** Info from map analysis ***\n\n"

        ra_str = Angle(d["RAJ2000"], "deg").to_string(unit="hour", precision=0)
        dec_str = Angle(d["DEJ2000"], "deg").to_string(unit="deg", precision=0)
        ss += "{:<20s} : {:8.3f} = {}\n".format("RA", d["RAJ2000"], ra_str)
        ss += "{:<20s} : {:8.3f} = {}\n".format("DEC", d["DEJ2000"], dec_str)

        ss += "{:<20s} : {:8.3f} +/- {:.3f} deg\n".format(
            "GLON", d["GLON"].value, d["GLON_Err"].value
        )
        ss += "{:<20s} : {:8.3f} +/- {:.3f} deg\n".format(
            "GLAT", d["GLAT"].value, d["GLAT_Err"].value
        )

        ss += "{:<20s} : {:.3f}\n".format("Position Error (68%)", d["Pos_Err_68"])
        ss += "{:<20s} : {:.3f}\n".format("Position Error (95%)", d["Pos_Err_95"])

        ss += "{:<20s} : {:.0f}\n".format("ROI number", d["ROI_Number"])
        ss += "{:<20s} : {}\n".format("Spatial model", d["Spatial_Model"])
        ss += "{:<20s} : {}\n".format("Spatial components", d["Components"])

        ss += "{:<20s} : {:.1f}\n".format("TS", d["Sqrt_TS"] ** 2)
        ss += "{:<20s} : {:.1f}\n".format("sqrt(TS)", d["Sqrt_TS"])

        ss += "{:<20s} : {:.3f} +/- {:.3f} (UL: {:.3f}) deg\n".format(
            "Size", d["Size"].value, d["Size_Err"].value, d["Size_UL"].value
        )

        ss += "{:<20s} : {:.3f}\n".format("R70", d["R70"])
        ss += "{:<20s} : {:.3f}\n".format("RSpec", d["RSpec"])

        ss += "{:<20s} : {:.1f}\n".format("Total model excess", d["Excess_Model_Total"])
        ss += "{:<20s} : {:.1f}\n".format("Excess in RSpec", d["Excess_RSpec"])
        ss += "{:<20s} : {:.1f}\n".format(
            "Model Excess in RSpec", d["Excess_RSpec_Model"]
        )
        ss += "{:<20s} : {:.1f}\n".format("Background in RSpec", d["Background_RSpec"])

        ss += "{:<20s} : {:.1f} hours\n".format("Livetime", d["Livetime"].value)

        ss += "{:<20s} : {:.2f}\n".format("Energy threshold", d["Energy_Threshold"])

        val, err = d["Flux_Map"].value, d["Flux_Map_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: 501
            "Source flux (>1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB,
        )

        ss += "\nFluxes in RSpec (> 1 TeV):\n"

        ss += "{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n".format(
            "Map measurement",
            d["Flux_Map_RSpec_Data"].value / FF,
            d["Flux_Map_RSpec_Data"].value * FLUX_TO_CRAB,
        )

        ss += "{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n".format(
            "Source model",
            d["Flux_Map_RSpec_Source"].value / FF,
            d["Flux_Map_RSpec_Source"].value * FLUX_TO_CRAB,
        )

        ss += "{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n".format(
            "Other component model",
            d["Flux_Map_RSpec_Other"].value / FF,
            d["Flux_Map_RSpec_Other"].value * FLUX_TO_CRAB,
        )

        ss += "{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n".format(
            "Large scale component model",
            d["Flux_Map_RSpec_LS"].value / FF,
            d["Flux_Map_RSpec_LS"].value * FLUX_TO_CRAB,
        )

        ss += "{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n".format(
            "Total model",
            d["Flux_Map_RSpec_Total"].value / FF,
            d["Flux_Map_RSpec_Total"].value * FLUX_TO_CRAB,
        )

        ss += "{:<35s} : {:5.1f} %\n".format(
            "Containment in RSpec", 100 * d["Containment_RSpec"]
        )
        ss += "{:<35s} : {:5.1f} %\n".format(
            "Contamination in RSpec", 100 * d["Contamination_RSpec"]
        )
        label, val = (
            "Flux correction (RSpec -> Total)",
            100 * d["Flux_Correction_RSpec_To_Total"],
        )
        ss += f"{label:<35s} : {val:5.1f} %\n"
        label, val = (
            "Flux correction (Total -> RSpec)",
            100 * (1 / d["Flux_Correction_RSpec_To_Total"]),
        )
        ss += f"{label:<35s} : {val:5.1f} %\n"

        return ss

    def _info_spec(self):
        """Print info from spectral analysis."""
        d = self.data
        ss = "\n*** Info from spectral analysis ***\n\n"

        ss += "{:<20s} : {:.1f} hours\n".format("Livetime", d["Livetime_Spec"].value)

        lo = d["Energy_Range_Spec_Min"].value
        hi = d["Energy_Range_Spec_Max"].value
        ss += "{:<20s} : {:.2f} to {:.2f} TeV\n".format("Energy range:", lo, hi)

        ss += "{:<20s} : {:.1f}\n".format("Background", d["Background_Spec"])
        ss += "{:<20s} : {:.1f}\n".format("Excess", d["Excess_Spec"])
        ss += "{:<20s} : {}\n".format("Spectral model", d["Spectral_Model"])

        val = d["TS_ECPL_over_PL"]
        ss += "{:<20s} : {:.1f}\n".format("TS ECPL over PL", val)

        val = d["Flux_Spec_Int_1TeV"].value
        err = d["Flux_Spec_Int_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "Best-fit model flux(> 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB,
        )

        val = d["Flux_Spec_Energy_1_10_TeV"].value
        err = d["Flux_Spec_Energy_1_10_TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 erg cm^-2 s^-1\n".format(
            "Best-fit model energy flux(1 to 10 TeV)", val / FF, err / FF
        )

        ss += self._info_spec_pl()
        ss += self._info_spec_ecpl()

        return ss

    def _info_spec_pl(self):
        d = self.data
        ss = "{:<20s} : {:.2f}\n".format("Pivot energy", d["Energy_Spec_PL_Pivot"])

        val = d["Flux_Spec_PL_Diff_Pivot"].value
        err = d["Flux_Spec_PL_Diff_Pivot_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "Flux at pivot energy",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB_DIFF,
        )

        val = d["Flux_Spec_PL_Int_1TeV"].value
        err = d["Flux_Spec_PL_Int_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "PL   Flux(> 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB,
        )

        val = d["Flux_Spec_PL_Diff_1TeV"].value
        err = d["Flux_Spec_PL_Diff_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "PL   Flux(@ 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB_DIFF,
        )

        val = d["Index_Spec_PL"]
        err = d["Index_Spec_PL_Err"]
        ss += "{:<20s} : {:.2f} +/- {:.2f}\n".format("PL   Index", val, err)

        return ss

    def _info_spec_ecpl(self):
        d = self.data
        ss = ""
        # ss = '{:<20s} : {:.1f}\n'.format('Pivot energy', d['Energy_Spec_ECPL_Pivot'])

        val = d["Flux_Spec_ECPL_Diff_1TeV"].value
        err = d["Flux_Spec_ECPL_Diff_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "ECPL   Flux(@ 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB_DIFF,
        )

        val = d["Flux_Spec_ECPL_Int_1TeV"].value
        err = d["Flux_Spec_ECPL_Int_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(  # noqa: E501
            "ECPL   Flux(> 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB,
        )

        val = d["Index_Spec_ECPL"]
        err = d["Index_Spec_ECPL_Err"]
        ss += "{:<20s} : {:.2f} +/- {:.2f}\n".format("ECPL Index", val, err)

        val = d["Lambda_Spec_ECPL"].value
        err = d["Lambda_Spec_ECPL_Err"].value
        ss += "{:<20s} : {:.3f} +/- {:.3f} TeV^-1\n".format("ECPL Lambda", val, err)

        # Use Gaussian analytical error propagation,
        # tested against the uncertainties package
        err = err / val**2
        val = 1.0 / val

        ss += "{:<20s} : {:.2f} +/- {:.2f} TeV\n".format("ECPL E_cut", val, err)

        return ss

    def _info_flux_points(self):
        """Print flux point results"""
        d = self.data
        ss = "\n*** Flux points info ***\n\n"
        ss += "Number of flux points: {}\n".format(d["N_Flux_Points"])
        ss += "Flux points table: \n\n"
        lines = format_flux_points_table(self.flux_points_table).pformat(
            max_width=-1, max_lines=-1
        )
        ss += "\n".join(lines)
        return ss + "\n"

    def _info_components(self):
        """Print info about the components."""
        ss = "\n*** Gaussian component info ***\n\n"
        ss += "Number of components: {}\n".format(len(self.components))
        ss += "{:<20s} : {}\n\n".format("Spatial components", self.data["Components"])

        for component in self.components:
            ss += str(component)
            ss += "\n\n"

        return ss

    @property
    def energy_range(self):
        """Spectral model energy range (`~astropy.units.Quantity` with length 2)."""
        energy_min, energy_max = (
            self.data["Energy_Range_Spec_Min"],
            self.data["Energy_Range_Spec_Max"],
        )

        if np.isnan(energy_min):
            energy_min = u.Quantity(0.2, "TeV")

        if np.isnan(energy_max):
            energy_max = u.Quantity(50, "TeV")

        return u.Quantity([energy_min, energy_max], "TeV")

    def spectral_model(self, which="best"):
        """Spectral model (`~gammapy.modeling.models.SpectralModel`).

        One of the following models (given by ``Spectral_Model`` in the catalog):

        - ``PL`` : `~gammapy.modeling.models.PowerLawSpectralModel`
        - ``ECPL`` : `~gammapy.modeling.models.ExpCutoffPowerLawSpectralModel`

        Parameters
        ----------
        which : {'best', 'pl', 'ecpl'}
            Which spectral model
        """
        data = self.data

        if which == "best":
            spec_type = self.data["Spectral_Model"].strip().lower()
        elif which in {"pl", "ecpl"}:
            spec_type = which
        else:
            raise ValueError(f"Invalid selection: which = {which!r}")

        if spec_type == "pl":
            tag = "PowerLawSpectralModel"
            pars = {
                "index": data["Index_Spec_PL"],
                "amplitude": data["Flux_Spec_PL_Diff_Pivot"],
                "reference": data["Energy_Spec_PL_Pivot"],
            }
            errs = {
                "amplitude": data["Flux_Spec_PL_Diff_Pivot_Err"],
                "index": data["Index_Spec_PL_Err"],
            }
        elif spec_type == "ecpl":
            tag = "ExpCutoffPowerLawSpectralModel"
            pars = {
                "index": data["Index_Spec_ECPL"],
                "amplitude": data["Flux_Spec_ECPL_Diff_Pivot"],
                "reference": data["Energy_Spec_ECPL_Pivot"],
                "lambda_": data["Lambda_Spec_ECPL"],
            }
            errs = {
                "index": data["Index_Spec_ECPL_Err"],
                "amplitude": data["Flux_Spec_ECPL_Diff_Pivot_Err"],
                "lambda_": data["Lambda_Spec_ECPL_Err"],
            }
        else:
            raise ValueError(f"Invalid spec_type: {spec_type}")

        model = Model.create(tag, "spectral", **pars)
        errs["reference"] = 0 * u.TeV

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`).

        One of the following models (given by ``Spatial_Model`` in the catalog):

        - ``Point-Like`` or has a size upper limit : `~gammapy.modeling.models.PointSpatialModel`
        - ``Gaussian``: `~gammapy.modeling.models.GaussianSpatialModel`
        - ``2-Gaussian`` or ``3-Gaussian``: composite model (using ``+`` with Gaussians)
        - ``Shell``: `~gammapy.modeling.models.ShellSpatialModel`
        """
        d = self.data
        pars = {"lon_0": d["GLON"], "lat_0": d["GLAT"], "frame": "galactic"}
        errs = {"lon_0": d["GLON_Err"], "lat_0": d["GLAT_Err"]}

        spatial_type = self.data["Spatial_Model"]

        if spatial_type == "Point-Like":
            tag = "PointSpatialModel"
        elif spatial_type == "Gaussian":
            tag = "GaussianSpatialModel"
            pars["sigma"] = d["Size"]
            errs["sigma"] = d["Size_Err"]
        elif spatial_type in {"2-Gaussian", "3-Gaussian"}:
            raise ValueError("For Gaussian or Multi-Gaussian models, use sky_model()!")
        elif spatial_type == "Shell":
            # HGPS contains no information on shell width
            # Here we assume a 5% shell width for all shells.
            tag = "ShellSpatialModel"
            pars["radius"] = 0.95 * d["Size"]
            pars["width"] = d["Size"] - pars["radius"]
            errs["radius"] = d["Size_Err"]
        else:
            raise ValueError(f"Invalid spatial_type: {spatial_type}")

        model = Model.create(tag, "spatial", **pars)
        for name, value in errs.items():
            model.parameters[name].error = value
        return model

    def sky_model(self, which="best"):
        """Source sky model.

        Parameters
        ----------
        which : {'best', 'pl', 'ecpl'}
            Which spectral model

        Returns
        -------
        sky_model : `~gammapy.modeling.models.Models`
           Models of the catalog object.
        """

        models = self.components_models(which=which)
        if len(models) > 1:
            geom = self._get_components_geom(models)
            return models.to_template_sky_model(geom=geom, name=self.name)
        else:
            return models[0]

    def components_models(self, which="best", linked=False):
        """Models of the source components.

        Parameters
        ----------
        which : {'best', 'pl', 'ecpl'}
            Which spectral model

        linked : bool
             Each sub-component of a source is given as a different `SkyModel`
             If True the spectral parameters except the mormalisation are linked.
             Default is False

        Returns
        -------
        sky_model : `~gammapy.modeling.models.Models`
           Models of the catalog object.
        """

        spatial_type = self.data["Spatial_Model"]
        missing_size = (
            spatial_type == "Gaussian" and self.spatial_model().sigma.value == 0
        )
        if spatial_type in {"2-Gaussian", "3-Gaussian"} or missing_size:
            models = []
            spectral_model = self.spectral_model(which=which)
            for component in self.components:
                spec_component = spectral_model.copy()
                weight = component.data["Flux_Map"] / self.data["Flux_Map"]
                spec_component.parameters["amplitude"].value *= weight
                if linked:
                    for name in spec_component.parameters.names:
                        if name not in ["norm", "amplitude"]:
                            spec_component.__dict__[name] = spectral_model.parameters[
                                name
                            ]
                model = SkyModel(
                    spatial_model=component.spatial_model(),
                    spectral_model=spec_component,
                    name=component.name,
                )
                models.append(model)
        else:
            models = [
                SkyModel(
                    spatial_model=self.spatial_model(),
                    spectral_model=self.spectral_model(which=which),
                    name=self.name,
                )
            ]
        return Models(models)

    @staticmethod
    def _get_components_geom(models):
        energy_axis = MapAxis.from_energy_bounds(
            "100 GeV", "100 TeV", nbin=10, per_decade=True, name="energy_true"
        )
        regions = [m.spatial_model.evaluation_region for m in models]
        geom = RegionGeom.from_regions(
            regions, binsz_wcs="0.02 deg", axes=[energy_axis]
        )
        return geom.to_wcs_geom()

    @property
    def flux_points_table(self):
        """Flux points table (`~astropy.table.Table`)."""
        table = Table()
        table.meta["sed_type_init"] = "dnde"
        table.meta["n_sigma_ul"] = 2
        table.meta["n_sigma"] = 1
        table.meta["sqrt_ts_threshold_ul"] = 1
        mask = ~np.isnan(self.data["Flux_Points_Energy"])

        table["e_ref"] = self.data["Flux_Points_Energy"][mask]
        table["e_min"] = self.data["Flux_Points_Energy_Min"][mask]
        table["e_max"] = self.data["Flux_Points_Energy_Max"][mask]

        table["dnde"] = self.data["Flux_Points_Flux"][mask]
        table["dnde_errn"] = self.data["Flux_Points_Flux_Err_Lo"][mask]
        table["dnde_errp"] = self.data["Flux_Points_Flux_Err_Hi"][mask]
        table["dnde_ul"] = self.data["Flux_Points_Flux_UL"][mask]
        table["is_ul"] = self.data["Flux_Points_Flux_Is_UL"][mask].astype("bool")
        return table


class SourceCatalogHGPS(SourceCatalog):
    """HESS Galactic plane survey (HGPS) source catalog.

    Reference: https://www.mpi-hd.mpg.de/hfm/HESS/hgps/

    One source is represented by `SourceCatalogObjectHGPS`.

    Examples
    --------
    Let's assume you have downloaded the HGPS catalog FITS file, e.g. via:

    .. code-block:: bash

        curl -O https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/hgps_catalog_v1.fits.gz

    Then you can load it up like this:

    >>> import matplotlib.pyplot as plt
    >>> from gammapy.catalog import SourceCatalogHGPS
    >>> filename = '$GAMMAPY_DATA/catalogs/hgps_catalog_v1.fits.gz'
    >>> cat = SourceCatalogHGPS(filename)

    Access a source by name:

    >>> source = cat['HESS J1843-033']
    >>> print(source)
    <BLANKLINE>
    *** Basic info ***
    <BLANKLINE>
    Catalog row index (zero-based) : 64
    Source name          : HESS J1843-033
    Analysis reference   : HGPS
    Source class         : Unid
    Identified object    : --
    Gamma-Cat id         : 126
    <BLANKLINE>
    <BLANKLINE>
    *** Info from map analysis ***
    <BLANKLINE>
    RA                   :  280.952 deg = 18h43m48s
    DEC                  :   -3.554 deg = -3d33m15s
    GLON                 :   28.899 +/- 0.072 deg
    GLAT                 :    0.075 +/- 0.036 deg
    Position Error (68%) : 0.122 deg
    Position Error (95%) : 0.197 deg
    ROI number           : 3
    Spatial model        : 2-Gaussian
    Spatial components   : HGPSC 083, HGPSC 084
    TS                   : 256.6
    sqrt(TS)             : 16.0
    Size                 : 0.239 +/- 0.063 (UL: 0.000) deg
    R70                  : 0.376 deg
    RSpec                : 0.376 deg
    Total model excess   : 979.5
    Excess in RSpec      : 775.6
    Model Excess in RSpec : 690.2
    Background in RSpec  : 1742.4
    Livetime             : 41.8 hours
    Energy threshold     : 0.38 TeV
    Source flux (>1 TeV) : (2.882 +/- 0.305) x 10^-12 cm^-2 s^-1 = (12.75 +/- 1.35) % Crab
    <BLANKLINE>
    Fluxes in RSpec (> 1 TeV):
    Map measurement                : 2.267 x 10^-12 cm^-2 s^-1 = 10.03 % Crab
    Source model                   : 2.018 x 10^-12 cm^-2 s^-1 =  8.93 % Crab
    Other component model          : 0.004 x 10^-12 cm^-2 s^-1 =  0.02 % Crab
    Large scale component model    : 0.361 x 10^-12 cm^-2 s^-1 =  1.60 % Crab
    Total model                    : 2.383 x 10^-12 cm^-2 s^-1 = 10.54 % Crab
    Containment in RSpec                :  70.0 %
    Contamination in RSpec              :  15.3 %
    Flux correction (RSpec -> Total)    : 121.0 %
    Flux correction (Total -> RSpec)    :  82.7 %
    <BLANKLINE>
    *** Info from spectral analysis ***
    <BLANKLINE>
    Livetime             : 16.8 hours
    Energy range:        : 0.22 to 61.90 TeV
    Background           : 5126.9
    Excess               : 980.1
    Spectral model       : PL
    TS ECPL over PL      : --
    Best-fit model flux(> 1 TeV) : (3.043 +/- 0.196) x 10^-12 cm^-2 s^-1  = (13.47 +/- 0.87) % Crab
    Best-fit model energy flux(1 to 10 TeV) : (10.918 +/- 0.733) x 10^-12 erg cm^-2 s^-1
    Pivot energy         : 1.87 TeV
    Flux at pivot energy : (0.914 +/- 0.058) x 10^-12 cm^-2 s^-1 TeV^-1  = (4.04 +/- 0.17) % Crab
    PL   Flux(> 1 TeV)   : (3.043 +/- 0.196) x 10^-12 cm^-2 s^-1  = (13.47 +/- 0.87) % Crab
    PL   Flux(@ 1 TeV)   : (3.505 +/- 0.247) x 10^-12 cm^-2 s^-1 TeV^-1  = (15.51 +/- 0.70) % Crab
    PL   Index           : 2.15 +/- 0.05
    ECPL   Flux(@ 1 TeV) : (0.000 +/- 0.000) x 10^-12 cm^-2 s^-1 TeV^-1  = (0.00 +/- 0.00) % Crab
    ECPL   Flux(> 1 TeV) : (0.000 +/- 0.000) x 10^-12 cm^-2 s^-1  = (0.00 +/- 0.00) % Crab
    ECPL Index           : -- +/- --
    ECPL Lambda          : 0.000 +/- 0.000 TeV^-1
    ECPL E_cut           : inf +/- nan TeV
    <BLANKLINE>
    *** Flux points info ***
    <BLANKLINE>
    Number of flux points: 6
    Flux points table:
    <BLANKLINE>
    e_ref  e_min  e_max        dnde         dnde_errn       dnde_errp        dnde_ul     is_ul
     TeV    TeV    TeV   1 / (cm2 s TeV) 1 / (cm2 s TeV) 1 / (cm2 s TeV) 1 / (cm2 s TeV)
    ------ ------ ------ --------------- --------------- --------------- --------------- -----
     0.332  0.215  0.511       3.048e-11       6.890e-12       7.010e-12       4.455e-11 False
     0.787  0.511  1.212       5.383e-12       6.655e-13       6.843e-13       6.739e-12 False
     1.957  1.212  3.162       9.160e-13       9.732e-14       1.002e-13       1.120e-12 False
     4.870  3.162  7.499       1.630e-13       2.001e-14       2.097e-14       2.054e-13 False
    12.115  7.499 19.573       1.648e-14       3.124e-15       3.348e-15       2.344e-14 False
    30.142 19.573 46.416       7.777e-16       4.468e-16       5.116e-16       1.883e-15 False
    <BLANKLINE>
    *** Gaussian component info ***
    <BLANKLINE>
    Number of components: 2
    Spatial components   : HGPSC 083, HGPSC 084
    <BLANKLINE>
    Component HGPSC 083:
    GLON                 :   29.047 +/- 0.026 deg
    GLAT                 :    0.244 +/- 0.027 deg
    Size                 : 0.125 +/- 0.021 deg
    Flux (>1 TeV)        : (1.34 +/- 0.36) x 10^-12 cm^-2 s^-1 = (5.9 +/- 1.6) % Crab
    <BLANKLINE>
    Component HGPSC 084:
    GLON                 :   28.770 +/- 0.059 deg
    GLAT                 :   -0.073 +/- 0.069 deg
    Size                 : 0.229 +/- 0.046 deg
    Flux (>1 TeV)        : (1.54 +/- 0.47) x 10^-12 cm^-2 s^-1 = (6.8 +/- 2.1) % Crab
    <BLANKLINE>
    <BLANKLINE>
    *** Source associations info ***
    <BLANKLINE>
      Source_Name    Association_Catalog    Association_Name   Separation
                                                                  deg
    ---------------- ------------------- --------------------- ----------
      HESS J1843-033                3FGL     3FGL J1843.7-0322   0.178442
      HESS J1843-033                3FGL     3FGL J1844.3-0344   0.242835
      HESS J1843-033                 SNR             G28.6-0.1   0.330376
    <BLANKLINE>

    Access source spectral data and plot it:

    >>> ax = plt.subplot()
    >>> source.spectral_model().plot(source.energy_range, ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>
    >>> source.spectral_model().plot_error(source.energy_range, ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>
    >>> source.flux_points.plot(ax=ax) #doctest:+ELLIPSIS
    <AxesSubplot:...xlabel='Energy [TeV]', ylabel='dnde [1 / (cm2 s TeV)]'>

    Gaussian component information can be accessed as well,
    either via the source, or via the catalog:

    >>> source.components
    [SourceCatalogObjectHGPSComponent('HGPSC 083'), SourceCatalogObjectHGPSComponent('HGPSC 084')]

    >>> cat.gaussian_component(83)
    SourceCatalogObjectHGPSComponent('HGPSC 084')
    """

    tag = "hgps"
    """Source catalog name (str)."""

    description = "H.E.S.S. Galactic plane survey (HGPS) source catalog"
    """Source catalog description (str)."""

    source_object_class = SourceCatalogObjectHGPS

    def __init__(
        self,
        filename="$GAMMAPY_DATA/catalogs/hgps_catalog_v1.fits.gz",
        hdu="HGPS_SOURCES",
    ):
        filename = make_path(filename)
        table = Table.read(filename, hdu=hdu)

        source_name_alias = ("Identified_Object",)
        super().__init__(table=table, source_name_alias=source_name_alias)

        self._table_components = Table.read(filename, hdu="HGPS_GAUSS_COMPONENTS")
        self._table_associations = Table.read(filename, hdu="HGPS_ASSOCIATIONS")
        self._table_associations["Separation"].format = ".6f"
        self._table_identifications = Table.read(filename, hdu="HGPS_IDENTIFICATIONS")
        self._table_large_scale_component = Table.read(
            filename, hdu="HGPS_LARGE_SCALE_COMPONENT"
        )

    @property
    def table_components(self):
        """Gaussian component table (`~astropy.table.Table`)"""
        return self._table_components

    @property
    def table_associations(self):
        """Source association table (`~astropy.table.Table`)"""
        return self._table_associations

    @property
    def table_identifications(self):
        """Source identification table (`~astropy.table.Table`)"""
        return self._table_identifications

    @property
    def table_large_scale_component(self):
        """Large scale component table (`~astropy.table.Table`)"""
        return self._table_large_scale_component

    @property
    def large_scale_component(self):
        """Large scale component model (`~gammapy.catalog.SourceCatalogLargeScaleHGPS`)."""
        return SourceCatalogLargeScaleHGPS(self.table_large_scale_component)

    def _make_source_object(self, index):
        """Make `SourceCatalogObject` for given row index"""
        source = super()._make_source_object(index)

        if source.data["Components"] != "":
            source.components = list(self._get_gaussian_components(source))

        self._attach_association_info(source)

        if source.data["Source_Class"] != "Unid":
            self._attach_identification_info(source)

        return source

    def _get_gaussian_components(self, source):
        for name in source.data["Components"].split(", "):
            row_index = int(name.split()[-1]) - 1
            yield self.gaussian_component(row_index)

    def _attach_association_info(self, source):
        t = self.table_associations
        mask = source.data["Source_Name"] == t["Source_Name"]
        source.associations = t[mask]

    def _attach_identification_info(self, source):
        t = self._table_identifications
        idx = np.nonzero(source.name == t["Source_Name"])[0][0]
        source.identification_info = table_row_to_dict(t[idx])

    def gaussian_component(self, row_idx):
        """Gaussian component (`SourceCatalogObjectHGPSComponent`)."""
        data = table_row_to_dict(self.table_components[row_idx])
        data[SourceCatalogObject._row_index_key] = row_idx
        return SourceCatalogObjectHGPSComponent(data=data)

    def to_models(self, which="best", components_status="independent"):
        """Create Models object from catalogue

        Parameters
        ----------
        which : {'best', 'pl', 'ecpl'}
            Which spectral model

        components_status : {'independent', 'linked', 'merged'}
            Relation between the sources components:
                'independent' : each sub-component of a source is given as
                                a different `SkyModel` (Default)
                'linked' : each sub-component of a source is given as
                           a different `SkyModel` but the spectral parameters
                           except the mormalisation are linked.
                'merged' : the sub-components are merged into a single `SkyModel`
                           given as a `~gammapy.modeling.models.TemplateSpatialModel`
                           with a `~gammapy.modeling.models.PowerLawNormSpectralModel`.
                           In that case the relavtie weights between the components
                           cannot be adjusted.

        Returns
        -------
        models : `~gammapy.modeling.models.Models`
            Models of the catalog.
        """

        models = []
        for source in self:
            if components_status == "merged":
                m = [source.sky_model(which=which)]
            else:
                m = source.components_models(
                    which=which, linked=components_status == "linked"
                )
            models.extend(m)
        return Models(models)


class SourceCatalogLargeScaleHGPS:
    """Gaussian band model.

    This 2-dimensional model is Gaussian in ``y`` for a given ``x``,
    and the Gaussian parameters can vary in ``x``.

    One application of this model is the diffuse emission along the
    Galactic plane, i.e. ``x = GLON`` and ``y = GLAT``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table of Gaussian parameters.
        ``x``, ``amplitude``, ``mean``, ``stddev``.
    interp_kwargs : dict
        Keyword arguments passed to `ScaledRegularGridInterpolator`
    """

    def __init__(self, table, interp_kwargs=None):
        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("values_scale", "lin")

        self.table = table
        glon = Angle(self.table["GLON"]).wrap_at("180d")

        interps = {}

        for column in table.colnames:
            values = self.table[column].quantity
            interp = ScaledRegularGridInterpolator((glon,), values, **interp_kwargs)
            interps[column] = interp

        self._interp = interps

    def _interpolate_parameter(self, parname, glon):
        glon = glon.wrap_at("180d")
        return self._interp[parname]((np.asanyarray(glon),), clip=False)

    def peak_brightness(self, glon):
        """Peak brightness at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("Surface_Brightness", glon)

    def peak_brightness_error(self, glon):
        """Peak brightness error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("Surface_Brightness_Err", glon)

    def width(self, glon):
        """Width at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("Width", glon)

    def width_error(self, glon):
        """Width error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("Width_Err", glon)

    def peak_latitude(self, glon):
        """Peak position at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("GLAT", glon)

    def peak_latitude_error(self, glon):
        """Peak position error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter("GLAT_Err", glon)

    def evaluate(self, position):
        """Evaluate model at a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        """
        glon, glat = position.galactic.l, position.galactic.b
        width = self.width(glon)
        amplitude = self.peak_brightness(glon)
        mean = self.peak_latitude(glon)
        return Gaussian1D.evaluate(glat, amplitude=amplitude, mean=mean, stddev=width)
