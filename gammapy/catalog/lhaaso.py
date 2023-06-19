# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from gammapy.modeling.models import Model, SkyModel
from gammapy.utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    "SourceCatalogObject1LHAASO",
    "SourceCatalog1LHAASO",
]


class SourceCatalogObject1LHAASO(SourceCatalogObject):
    """One source from the 1LHAASO catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog1LHAASO`.
    """

    _source_name_key = "Source_Name"

    def _parse(self, name, which):
        if which == "all" or self.data["Model_a"] == which:
            tag = ""
        else:
            tag = "_b"
        is_ul = False
        value = self.data[f"{name}{tag}"]
        if (np.isnan(value) or value == 0) and f"{name}{tag}_ul" in self.data:
            value = self.data[f"{name}{tag}_ul"]
            is_ul = True
        if np.isnan(value) or value == 0:
            value = self.data[f"{name}"]
        return value, is_ul

    def _get(self, name, which):
        value, _ = self._parse(name, which)
        return value

    def spectral_model(self, which="all"):
        """Spectral model (`~gammapy.modeling.models.PowerLawSpectralModel`).

        * ``which="all"`` - First sky model listed for the source.
        * ``which="KM2A"`` - Sky model for KM2A analysis only.
        * ``which="WCDA"`` - Sky model for WCDA analysis only.

        If only a limit is given for a parameter it is used as value.
        Entries not repeated for the second analysis are taken from the first one.
        """

        pars = {
            "reference": self._get("E0", which),
            "amplitude": self._get("N0", which),
            "index": self._get("gamma", which),
        }

        errs = {
            "amplitude": self._get("N0_err", which),
            "index": self._get("gamma_err", which),
        }

        model = Model.create("PowerLawSpectralModel", "spectral", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def spatial_model(self, which="all"):
        """Spatial model (`~gammapy.modeling.models.SpatialModel`).

        * ``which="all"`` - First sky model listed for the source.
        * ``which="KM2A"`` - Sky model for KM2A analysis only.
        * ``which="WCDA"`` - Sky model for WCDA analysis only.

        If only a limit is given for a parameter it is used as value.
        Entries not repeated for the second analysis are taken from the first one.
        """

        lat_0 = self._get("DECJ2000", which)
        pars = {"lon_0": self._get("RAJ2000", which), "lat_0": lat_0, "frame": "fk5"}

        pos_err = self._get("pos_err", which)
        errs = {
            "lat_0": pos_err,
            "lon_0": pos_err / np.cos(lat_0),
        }

        r39, is_ul = self._parse("r39", which)
        if not is_ul:
            pars["sigma"] = r39
            model = Model.create("GaussianSpatialModel", "spatial", **pars)
        else:
            model = Model.create("PointSpatialModel", "spatial", **pars)

        for name, value in errs.items():
            model.parameters[name].error = value

        return model

    def sky_model(self, which="all"):
        """Sky model (`~gammapy.modeling.models.SkyModel`).

        * ``which="all"`` - First sky model listed for the source.
        * ``which="KM2A"`` - Sky model for KM2A analysis only.
        * ``which="WCDA"`` - Sky model for WCDA analysis only.

        If only a limit is given for a parameter it is used as value.
        Entries not repeated for the second analysis are taken from the first one.
        """
        return SkyModel(
            spatial_model=self.spatial_model(which),
            spectral_model=self.spectral_model(which),
            name=self.name,
        )


class SourceCatalog1LHAASO(SourceCatalog):
    """First LHAASO catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3HWC`.

    The data is from table 1 in the paper [1]_.

    The catalog table contains 90 rows / sources.

    References
    ----------
    .. [1] 1LHAASO: The First LHAASO Catalog of Gamma-Ray Sources,
       <https://ui.adsabs.harvard.edu/abs/2023arXiv230517030C/abstract>`__
    """

    tag = "1LHAASO"
    """Catalog name"""

    description = "1LHAASO catalog from the LHAASO observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject1LHAASO

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/1LHAASO.fits"):
        table = Table.read(make_path(filename))

        source_name_key = "Source_Name"

        super().__init__(table=table, source_name_key=source_name_key)
