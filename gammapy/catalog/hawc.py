# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC catalogs (https://www.hawc-observatory.org)."""
import abc
import numpy as np
from astropy.table import Table
from gammapy.modeling.models import Model, SkyModel
from gammapy.utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalog2HWC",
    "SourceCatalog3HWC",
    "SourceCatalogObject2HWC",
    "SourceCatalogObject3HWC",
]


class SourceCatalogObjectHWCBase(SourceCatalogObject, abc.ABC):
    """Base class for the HAWC catalogs objects"""

    _source_name_key = "source_name"

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
        return (
            f"\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based) : {self.row_index}\n"
            f"Source name : {self.name}\n"
        )

    def _info_position(self):
        """Print position info."""
        return (
            f"\n*** Position info ***\n\n"
            f"RA: {self.data.ra:.3f}\n"
            f"DEC: {self.data.dec:.3f}\n"
            f"GLON: {self.data.glon:.3f}\n"
            f"GLAT: {self.data.glat:.3f}\n"
            f"Position error: {self.data.pos_err:.3f}\n"
        )

    def _info_spectrum(self):
        """Print spectral info."""
        ss = "\n*** Spectral info ***\n\n"
        ss += self._info_spectrum_one(0)

        if self.n_models == 2:
            ss += self._info_spectrum_one(1)
        else:
            ss += "No second spectrum available"

        return ss


class SourceCatalogObject2HWC(SourceCatalogObjectHWCBase):
    """One source from the HAWC 2HWC catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2HWC`.
    """

    @property
    def n_models(self):
        """Number of models (1 or 2)."""
        if hasattr(self.data, "spec1_dnde"):
            return 1 if np.isnan(self.data.spec1_dnde) else 2
        else:
            return 1

    def _get_idx(self, which):
        if which == "point":
            return 0
        elif which == "extended":
            if self.n_models == 2:
                return 1
            else:
                raise ValueError(f"No extended source analysis available: {self.name}")
        else:
            raise ValueError(f"Invalid which: {which!r}")

    def _info_spectrum_one(self, idx):
        d = self.data
        ss = f"Spectrum {idx}:\n"
        val, err = d[f"spec{idx}_dnde"].value, d[f"spec{idx}_dnde_err"].value
        ss += f"Flux at 7 TeV: {val:.3} +- {err:.3} cm-2 s-1 TeV-1\n"
        val, err = d[f"spec{idx}_index"], d[f"spec{idx}_index_err"]
        ss += f"Spectral index: {val:.3f} +- {err:.3f}\n"
        radius = d[f"spec{idx}_radius"]
        ss += f"Test Radius: {radius:1}\n\n"
        return ss

    def spectral_model(self, which="point"):
        """Spectral model (`~gammapy.modeling.models.PowerLawSpectralModel`).

        * ``which="point"`` -- Spectral model under the point source assumption.
        * ``which="extended"`` -- Spectral model under the extended source assumption.
          Only available for some sources. Raise ValueError if not available.
        """
        idx = self._get_idx(which)

        pars = {
            "reference": "7 TeV",
            "amplitude": self.data[f"spec{idx}_dnde"],
            "index": -self.data[f"spec{idx}_index"],
        }

        errs = {
            "amplitude": self.data[f"spec{idx}_dnde_err"],
            "index": self.data[f"spec{idx}_index_err"],
        }

        model = Model.create("PowerLawSpectralModel", "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self, which="point"):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`).

        * ``which="point"`` - `~gammapy.modeling.models.PointSpatialModel`
        * ``which="extended"`` - `~gammapy.modeling.models.DiskSpatialModel`.
          Only available for some sources. Raise ValueError if not available.
        """
        idx = self._get_idx(which)
        pars = {"lon_0": self.data.glon, "lat_0": self.data.glat, "frame": "galactic"}

        if idx == 0:
            tag = "PointSpatialModel"
        else:
            tag = "DiskSpatialModel"
            pars["r_0"] = self.data[f"spec{idx}_radius"]

        errs = {
            "lat_0": self.data.pos_err,
            "lon_0": self.data.pos_err / np.cos(self.data.glat),
        }

        model = Model.create(tag, "spatial", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def sky_model(self, which="point"):
        """Sky model (`~gammapy.modeling.models.SkyModel`).

        * ``which="point"`` - Sky model for point source analysis
        * ``which="extended"`` - Sky model for extended source analysis.
          Only available for some sources. Raise ValueError if not available.

        According to the paper, the radius of the extended source model is only a rough estimate
        of the source size, based on the residual excess..
        """
        return SkyModel(
            spatial_model=self.spatial_model(which),
            spectral_model=self.spectral_model(which),
            name=self.name,
        )


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

    tag = "2hwc"
    """Catalog name"""

    description = "2HWC catalog from the HAWC observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject2HWC

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/2HWC.ecsv"):
        table = Table.read(make_path(filename), format="ascii.ecsv")

        source_name_key = "source_name"

        super().__init__(table=table, source_name_key=source_name_key)


class SourceCatalogObject3HWC(SourceCatalogObjectHWCBase):
    """One source from the HAWC 3HWC catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3HWC`.
    """

    @property
    def n_models(self):
        return 1

    def _info_spectrum_one(self, idx):
        d = self.data
        ss = f"Spectrum {idx}:\n"
        val, errn, errp = (
            d[f"spec{idx}_dnde"].value,
            d[f"spec{idx}_dnde_errn"].value,
            d[f"spec{idx}_dnde_errp"].value,
        )
        ss += f"Flux at 7 TeV: {val:.3} {errn:.3} + {errp:.3} cm-2 s-1 TeV-1\n"
        val, errn, errp = (
            d[f"spec{idx}_index"],
            d[f"spec{idx}_index_errn"],
            d[f"spec{idx}_index_errp"],
        )
        ss += f"Spectral index: {val:.3f} {errn:.3f} + {errp:.3f}\n"
        radius = d[f"spec{idx}_radius"]
        ss += f"Test Radius: {radius:1}\n\n"
        return ss

    @property
    def is_pointlike(self):
        return self.data["spec0_radius"] == 0.0

    def spectral_model(self):
        """Spectral model (`~gammapy.modeling.models.PowerLawSpectralModel`)."""

        pars = {
            "reference": "7 TeV",
            "amplitude": self.data["spec0_dnde"],
            "index": -self.data["spec0_index"],
        }

        errs = {
            "index": 0.5
            * (self.data["spec0_index_errp"] + np.abs(self.data["spec0_index_errn"])),
            "amplitude": 0.5
            * (self.data["spec0_dnde_errp"] + np.abs(self.data["spec0_dnde_errn"])),
        }

        model = Model.create("PowerLawSpectralModel", "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`)."""
        pars = {"lon_0": self.data.glon, "lat_0": self.data.glat, "frame": "galactic"}

        if self.is_pointlike:
            tag = "PointSpatialModel"
        else:
            tag = "DiskSpatialModel"
            pars["r_0"] = self.data["spec0_radius"]

        errs = {
            "lat_0": self.data.pos_err,
            "lon_0": self.data.pos_err / np.cos(self.data.glat),
        }

        model = Model.create(tag, "spatial", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def sky_model(self):
        """Sky model (`~gammapy.modeling.models.SkyModel`)."""
        return SkyModel(
            spatial_model=self.spatial_model(),
            spectral_model=self.spectral_model(),
            name=self.name,
        )


class SourceCatalog3HWC(SourceCatalog):
    """HAWC 3HWC catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3HWC`.

    The data is from tables 2 and 3 in the paper [1]_.

    The catalog table contains 65 rows / sources.

    References
    ----------
    .. [1] 3HWC: The Third HAWC Catalog of Very-High-Energy Gamma-ray Sources",
       <https://data.hawc-observatory.org/datasets/3hwc-survey/index.php>`__
    """

    tag = "3hwc"
    """Catalog name"""

    description = "3HWC catalog from the HAWC observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject3HWC

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/3HWC.ecsv"):
        table = Table.read(make_path(filename), format="ascii.ecsv")

        source_name_key = "source_name"

        super().__init__(table=table, source_name_key=source_name_key)
