# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC catalogs (https://www.hawc-observatory.org)."""
import numpy as np
from astropy.table import Table
from gammapy.modeling.models import (
    DiskSpatialModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
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
        return (
            f"\n*** Basic info ***\n\n"
            f"Catalog row index (zero-based) : {self.data['catalog_row_index']}\n"
            f"Source name : {self.data['source_name']}\n"
        )

    def _info_position(self):
        """Print position info."""
        return (
            f"\n*** Position info ***\n\n"
            f"RA: {self.data['ra']:.3f}\n"
            f"DEC: {self.data['dec']:.3f}\n"
            f"GLON: {self.data['glon']:.3f}\n"
            f"GLAT: {self.data['glat']:.3f}\n"
            f"Position error: {self.data['pos_err']:.3f}\n"
        )

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
        """Number of models (1 or 2)."""
        return 1 if np.isnan(self.data["spec1_dnde"]) else 2

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

    def spectral_model(self, which="point"):
        """Spectral model (`~gammapy.modeling.models.PowerLawSpectralModel`).

        * ``which="point"`` -- Spectral model under the point source assumption.
        * ``which="extended"`` -- Spectral model under the extended source assumption.
          Only available for some sources. Raise ValueError if not available.
        """
        idx = self._get_idx(which)

        pars, errs = {}, {}
        pars["amplitude"] = self.data[f"spec{idx}_dnde"]
        errs["amplitude"] = self.data[f"spec{idx}_dnde_err"]
        pars["index"] = -self.data[f"spec{idx}_index"]
        errs["index"] = self.data[f"spec{idx}_index_err"]
        pars["reference"] = "7 TeV"

        model = PowerLawSpectralModel(**pars)
        model.parameters.set_parameter_errors(errs)

        return model

    def spatial_model(self, which="point"):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`).

        * ``which="point"`` - `~gammapy.modeling.models.PointSpatialModel`
        * ``which="extended"`` - `~gammapy.modeling.models.DiskSpatialModel`.
          Only available for some sources. Raise ValueError if not available.
        """
        idx = self._get_idx(which)

        if idx == 0:
            model = PointSpatialModel(
                lon_0=self.data["glon"], lat_0=self.data["glat"], frame="galactic"
            )
        else:
            model = DiskSpatialModel(
                lon_0=self.data["glon"],
                lat_0=self.data["glat"],
                r_0=self.data[f"spec{idx}_radius"],
                frame="galactic",
            )

        lat_err = self.data["pos_err"].to("deg")
        lon_err = self.data["pos_err"].to("deg") / np.cos(self.data["glat"].to("rad"))
        model.parameters.set_parameter_errors(dict(lon_0=lon_err, lat_0=lat_err))

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
            self.spatial_model(which), self.spectral_model(which), name=self.name
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

    name = "2hwc"
    """Catalog name"""

    description = "2HWC catalog from the HAWC observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject2HWC

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/2HWC.ecsv"):
        table = Table.read(make_path(filename), format="ascii.ecsv")

        source_name_key = "source_name"

        super().__init__(table=table, source_name_key=source_name_key)
