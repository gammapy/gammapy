# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC catalogs (https://www.hawc-observatory.org)."""
import numpy as np
from astropy.table import Table
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    DiskSpatialModel,
    PointSpatialModel,
)
from gammapy.utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject

__all__ = ["SourceCatalog2HWC", "SourceCatalogObject2HWC"]


class SourceCatalogObject2HWC(SourceCatalogObject):
    """One source from the HAWC 2HWC catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2HWC`.
    """

    _source_name_key = "source_name"
    _source_index_key = "catalog_row_index"

    def __str__(self):
        return self.info()

    def info(self, info="all"):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectrum'}
            Comma separated list of options
        """
        if info == "all":
            info = "basic,position,spectrum"

        ss = ""
        ops = info.split(",")
        if "basic" in ops:
            ss += self._info_basic()
        if "position" in ops:
            ss += self._info_position()
        if "spectrum" in ops:
            ss += self._info_spectrum()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = "\n*** Basic info ***\n\n"
        ss += "Catalog row index (zero-based) : {}\n".format(d["catalog_row_index"])
        ss += "{:<15s} : {}\n".format("Source name:", d["source_name"])

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = "\n*** Position info ***\n\n"
        ss += "{:20s} : {:.3f}\n".format("RA", d["ra"])
        ss += "{:20s} : {:.3f}\n".format("DEC", d["dec"])
        ss += "{:20s} : {:.3f}\n".format("GLON", d["glon"])
        ss += "{:20s} : {:.3f}\n".format("GLAT", d["glat"])
        ss += "{:20s} : {:.3f}\n".format("Position error", d["pos_err"])

        return ss

    @staticmethod
    def _info_spectrum_one(d, idx):
        label = f"spec{idx}_"
        ss = f"Spectrum {idx}:\n"
        args = (
            "Flux at 7 TeV",
            d[label + "dnde"].value,
            d[label + "dnde_err"].value,
            "cm-2 s-1 TeV-1",
        )
        ss += "{:20s} : {:.3} +- {:.3} {}\n".format(*args)
        args = "Spectral index", d[label + "index"], d[label + "index_err"]
        ss += "{:20s} : {:.3f} +- {:.3f}\n".format(*args)
        ss += "{:20s} : {:1}\n\n".format("Test radius", d[label + "radius"])
        return ss

    def _info_spectrum(self):
        """Print spectral info."""
        d = self.data
        ss = "\n*** Spectral info ***\n\n"
        ss += self._info_spectrum_one(d, 0)

        if self.n_models == 2:
            ss += self._info_spectrum_one(d, 1)
        else:
            ss += "No second spectrum available for this source"

        return ss

    @property
    def n_models(self):
        """Number of measured spectra (1 or 2)."""
        return 1 if np.isnan(self.data["spec1_dnde"]) else 2

    def _get_spectral_model(self, idx):
        pars, errs = {}, {}
        data = self.data
        label = f"spec{idx}_"

        pars["amplitude"] = data[label + "dnde"]
        errs["amplitude"] = data[label + "dnde_err"]
        pars["index"] = data[label + "index"]
        errs["index"] = data[label + "index_err"]
        pars["reference"] = "7 TeV"

        model = PowerLawSpectralModel(**pars)
        model.parameters.set_parameter_errors(errs)

        return model

    @property
    def _spectral_models(self):
        """Spectral models (either one or two).

        The HAWC catalog has one or two spectral measurements for each source.

        Returns
        -------
        models : list
            List of `~gammapy.modeling.models.SpectralModel`
        """
        models = [self._get_spectral_model(0)]

        if self.n_models == 2:
            models.append(self._get_spectral_model(1))

        return models

    def _get_spatial_model(self, idx):
        d = self.data
        label = f"spec{idx}_"

        r_0 = d[label + "radius"]
        if r_0 != 0.0:
            model = DiskSpatialModel(d["glon"], d["glat"], r_0, frame="galactic")
        else:
            model = PointSpatialModel(d["glon"], d["glon"], frame="galactic")

        return model

    @property
    def _spatial_models(self):
        """Spatial models (either one or two).

        The HAWC catalog has one or two tested radius for each source.

        Returns
        -------
        models : list
            List of `~gammapy.modeling.models.SpatialModel`
        """
        models = [self._get_spatial_model(0)]

        if self.n_models == 2:
            models.append(self._get_spatial_model(1))

        return models

    @property
    def sky_models(self):
        """Sky models (either one or two).

        The HAWC catalog has one or two models for each source.
        The radius of secondary model is a rough estimate based on the residual excess
        above the point source model. This radius should not be regarded as a definite
        measurement of the source extent.

        Returns
        -------
        models : list
            List of `~gammapy.modeling.models.SkyModel`
        """
        sky_models = []
        for km in range(self.n_models):
            spatial_model = self._spatial_models[km]
            spectral_model = self._spectral_models[km]
            sky_models.append(SkyModel(spatial_model, spectral_model, name=self.name))
        return sky_models


class SourceCatalog2HWC(SourceCatalog):
    """HAWC 2HWC catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject2HWC`.

    The data is from tables 2 and 3 in the paper [1]_.

    The catalog table contains 40 rows / sources.
    The paper mentions 39 sources e.g. in the abstract.
    The difference is due to Geminga, which was detected as two "sources" by the algorithm
    used to make the catalog, but then in the discussion considered as one source.

    References
    ----------
    .. [1] Abeysekara et al, "The 2HWC HAWC Observatory Gamma Ray Catalog",
       On ADS: `2017ApJ...843...40A <https://ui.adsabs.harvard.edu/abs/2017ApJ...843...40A>`__
    """

    name = "2hwc"
    """Catalog name"""

    description = "2HWC catalog from the HAWC observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject2HWC

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/2HWC.ecsv"):
        filename = str(make_path(filename))
        table = Table.read(filename, format="ascii.ecsv")

        source_name_key = "source_name"

        super().__init__(table=table, source_name_key=source_name_key)
