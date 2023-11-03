# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.table import Table
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import Model, Models, SkyModel
from gammapy.utils.gauss import Gauss2DPDF
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
        if which in self.data["Model_a"]:
            tag = ""
        elif which in self.data["Model_b"]:
            tag = "_b"
        else:
            raise ValueError("Invalid model component name")
        is_ul = False
        value = u.Quantity(self.data[f"{name}{tag}"])
        if (
            np.isnan(value) or value == 0 * value.unit
        ) and f"{name}_ul{tag}" in self.data:
            value = self.data[f"{name}_ul{tag}"]
            is_ul = True
        return value, is_ul

    def _get(self, name, which):
        value, _ = self._parse(name, which)
        return value

    def spectral_model(self, which):
        """Spectral model as a `~gammapy.modeling.models.PowerLawSpectralModel` object.

        * ``which="KM2A"`` - Sky model for KM2A analysis only.
        * ``which="WCDA"`` - Sky model for WCDA analysis only.

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

    def spatial_model(self, which):
        """Spatial model as a `~gammapy.modeling.models.SpatialModel` object.

        * ``which="KM2A"`` - Sky model for KM2A analysis only.
        * ``which="WCDA"`` - Sky model for WCDA analysis only.

        """
        lat_0 = self._get("DECJ2000", which)
        pars = {"lon_0": self._get("RAJ2000", which), "lat_0": lat_0, "frame": "fk5"}

        pos_err = self._get("pos_err", which)
        scale_r95 = Gauss2DPDF().containment_radius(0.95)

        errs = {
            "lat_0": pos_err / scale_r95,
            "lon_0": pos_err / scale_r95 / np.cos(lat_0),
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

    @staticmethod
    def _get_components_geom(models):
        energy_axis = MapAxis.from_energy_bounds(
            "0.1 TeV", "2000 TeV", nbin=10, per_decade=True, name="energy"
        )
        regions = [m.spatial_model.evaluation_region for m in models]
        geom = RegionGeom.from_regions(
            regions, binsz_wcs="0.05 deg", axes=[energy_axis]
        )
        return geom.to_wcs_geom()

    def sky_model(self, which="both"):
        """Sky model as a `~gammapy.modeling.models.SkyModel` object.

        * ``which="both"`` -  Create a composite template if both models are available, or, use the available one
           if only one is present.
        * ``which="KM2A"`` - Sky model for KM2A analysis if available.
        * ``which="WCDA"`` - Sky model for WCDA analysis if available.

        """
        if which == "both":
            wcda = self.sky_model(which="WCDA")
            km2a = self.sky_model(which="KM2A")
            models = [m for m in [wcda, km2a] if m is not None]
            if len(models) == 2:
                geom = self._get_components_geom(models)
                mask = geom.energy_mask(energy_max=25 * u.TeV)
                geom = geom.as_energy_true
                wcda_map = Models(wcda).to_template_sky_model(geom).spatial_model.map
                model = Models(km2a).to_template_sky_model(geom, name=km2a.name)
                model.spatial_model.map.data[mask] = wcda_map.data[mask]
                model.spatial_model.filename = f"{model.name}.fits"
                return model
            else:
                return models[0]
        else:
            _, is_ul = self._parse("N0", which)
            if is_ul:
                return None
            else:
                return SkyModel(
                    spatial_model=self.spatial_model(which),
                    spectral_model=self.spectral_model(which),
                    name=self.name,
                )


class SourceCatalog1LHAASO(SourceCatalog):
    """First LHAASO catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject1LHAASO`.

    The data is from table 1 in the paper [1]_.

    The catalog table contains 90 rows / sources.

    References
    ----------
    .. [1] 1LHAASO: The First LHAASO Catalog of Gamma-Ray Sources,
       https://ui.adsabs.harvard.edu/abs/2023arXiv230517030C/abstract
    """

    tag = "1LHAASO"
    """Catalog name"""

    description = "1LHAASO catalog from the LHAASO observatory"
    """Catalog description"""

    source_object_class = SourceCatalogObject1LHAASO

    def __init__(self, filename="$GAMMAPY_DATA/catalogs/1LHAASO_catalog.fits"):
        table = Table.read(make_path(filename))

        source_name_key = "Source_Name"

        super().__init__(table=table, source_name_key=source_name_key)

    def to_models(self, which="both"):
        """Create Models object from catalog.

        * ``which="both"`` - Use first model or create a composite template if both models are available.
        * ``which="KM2A"`` - Sky model for KM2A analysis if available.
        * ``which="WCDA"`` - Sky model for WCDA analysis if available.
        """
        models = Models()
        for _ in self:
            model = _.sky_model(which)
            if model:
                models.append(model)
        return models
