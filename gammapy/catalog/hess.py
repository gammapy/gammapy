# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HESS Galactic plane survey (HGPS) catalog."""
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from astropy.table import Table
from gammapy.estimators import FluxPoints
from gammapy.modeling.models import Model, Models, SkyModel
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path
from gammapy.utils.table import table_row_to_dict
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalogHGPS",
    "SourceCatalogObjectHGPS",
    "SourceCatalogObjectHGPSComponent",
    "SourceCatalogLargeScaleHGPS",
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
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 = ({:.2f} +/- {:.2f}) % Crab\n".format(
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
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
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
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
            "Flux at pivot energy",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB_DIFF,
        )

        val = d["Flux_Spec_PL_Int_1TeV"].value
        err = d["Flux_Spec_PL_Int_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
            "PL   Flux(> 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB,
        )

        val = d["Flux_Spec_PL_Diff_1TeV"].value
        err = d["Flux_Spec_PL_Diff_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
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
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
            "ECPL   Flux(@ 1 TeV)",
            val / FF,
            err / FF,
            val * FLUX_TO_CRAB,
            err * FLUX_TO_CRAB_DIFF,
        )

        val = d["Flux_Spec_ECPL_Int_1TeV"].value
        err = d["Flux_Spec_ECPL_Int_1TeV_Err"].value
        ss += "{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n".format(
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
        err = err / val ** 2
        val = 1.0 / val

        ss += "{:<20s} : {:.2f} +/- {:.2f} TeV\n".format("ECPL E_cut", val, err)

        return ss

    def _info_flux_points(self):
        """Print flux point results"""
        d = self.data
        ss = "\n*** Flux points info ***\n\n"
        ss += "Number of flux points: {}\n".format(d["N_Flux_Points"])
        ss += "Flux points table: \n\n"
        lines = self.flux_points.table_formatted.pformat(max_width=-1, max_lines=-1)
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
        sky_model : `~gammapy.modeling.models.SkyModel`
            Sky model of the catalog object.
        """
        spatial_type = self.data["Spatial_Model"]
        if spatial_type in {"2-Gaussian", "3-Gaussian"}:
            models = []
            for component in self.components:
                spectral_model = self.spectral_model(which=which)
                weight = component.data["Flux_Map"] / self.data["Flux_Map"]
                spectral_model.parameters["amplitude"].value *= weight
                model = SkyModel(
                    spatial_model=component.spatial_model(),
                    spectral_model=spectral_model,
                    name=component.name,
                )
                models.append(model)

            return Models(models)
        else:
            return SkyModel(
                spatial_model=self.spatial_model(),
                spectral_model=self.spectral_model(which=which),
                name=self.name,
            )

    @property
    def flux_points(self):
        """Flux points (`~gammapy.estimators.FluxPoints`)."""
        table = Table()
        table.meta["SED_TYPE"] = "dnde"
        mask = ~np.isnan(self.data["Flux_Points_Energy"])

        table["e_ref"] = self.data["Flux_Points_Energy"][mask]
        table["e_min"] = self.data["Flux_Points_Energy_Min"][mask]
        table["e_max"] = self.data["Flux_Points_Energy_Max"][mask]

        table["dnde"] = self.data["Flux_Points_Flux"][mask]
        table["dnde_errn"] = self.data["Flux_Points_Flux_Err_Lo"][mask]
        table["dnde_errp"] = self.data["Flux_Points_Flux_Err_Hi"][mask]
        table["dnde_ul"] = self.data["Flux_Points_Flux_UL"][mask]
        table["is_ul"] = self.data["Flux_Points_Flux_Is_UL"][mask].astype("bool")

        return FluxPoints(table)


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

    >>> from gammapy.catalog import SourceCatalogHGPS
    >>> filename = 'hgps_catalog_v1.fits.gz'
    >>> cat = SourceCatalogHGPS(filename)

    Access a source by name:

    >>> source = cat['HESS J1843-033']
    >>> print(source)

    Access source spectral data and plot it:

    >>> source.spectral_model().plot(source.energy_range)
    >>> source.spectral_model().plot_error(source.energy_range)
    >>> source.flux_points.plot()

    Gaussian component information can be accessed as well,
    either via the source, or via the catalog:

    >>> source.components
    >>> cat.gaussian_component(83)
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
