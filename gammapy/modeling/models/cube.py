# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube models (axes: lon, lat, energy)."""
import copy
from pathlib import Path
import numpy as np
import astropy.units as u
from gammapy.maps import Map
from gammapy.modeling import Model, Parameter, Parameters
from gammapy.utils.scripts import make_path, read_yaml, write_yaml


class SkyModelBase(Model):
    """Sky model base class"""

    def __add__(self, other):
        if isinstance(other, (SkyModels, list)):
            return SkyModels([self, *other])
        elif isinstance(other, (SkyModel, SkyDiffuseCube)):
            return SkyModels([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy):
        return self.evaluate(lon, lat, energy)

    def evaluate_geom(self, geom):
        coords = geom.get_coord(coordsys=self.frame)
        return self(coords.lon, coords.lat, coords["energy"])


class SkyModels:
    """Collection of `~gammapy.modeling.models.SkyModel`

    Parameters
    ----------
    skymodels : list of `~gammapy.modeling.models.SkyModel`
        Sky models
    """

    def __init__(self, skymodels):
        self._skymodels = skymodels

    @property
    def parameters(self):
        return Parameters.from_stack([_.parameters for _ in self._skymodels])

    @classmethod
    def from_yaml(cls, filename):
        """Write to YAML file."""
        from gammapy.modeling.serialize import dict_to_models

        data = read_yaml(filename)
        skymodels = dict_to_models(data)
        return cls(skymodels)

    def to_yaml(self, filename):
        """Write to YAML file."""
        from gammapy.modeling.serialize import models_to_dict

        components_dict = models_to_dict(self._skymodels)
        write_yaml(components_dict, filename, sort_keys=False)

    def evaluate(self, lon, lat, energy):
        out = self._skymodels[0].evaluate(lon, lat, energy)
        for skymodel in self._skymodels[1:]:
            out += skymodel.evaluate(lon, lat, energy)
        return out

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        for idx, skymodel in enumerate(self):
            str_ += f"Component {idx}: {skymodel}\n\n\t\n\n"

        return str_

    def __add__(self, other):
        if isinstance(other, (SkyModels, list)):
            return SkyModels([*self, *other])
        elif isinstance(other, (SkyModel, SkyDiffuseCube)):
            return SkyModels([*self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __getitem__(self, val):
        if isinstance(val, int):
            return self._skymodels[val]
        elif isinstance(val, str):
            for idx, model in enumerate(self._skymodels):
                if val == model.name:
                    return self._skymodels[idx]
            raise IndexError(f"No model: {val!r}")
        else:
            raise TypeError(f"Invalid type: {type(val)!r}")

    def __len__(self):
        return len(self._skymodels)


class SkyModel(SkyModelBase):
    """Sky model component.

    This model represents a factorised sky model.
    It has `~gammapy.modeling.Parameters`
    combining the spatial and spectral parameters.

    TODO: add possibility to have a temporal model component also.

    Parameters
    ----------
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model
    name : str
        Model identifier
    """

    tag = "SkyModel"

    def __init__(self, spatial_model, spectral_model, name="source"):
        self.name = name
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        super().__init__()

        # TODO: this hack is needed for compound models to work
        self.__dict__.pop("_parameters")

    @property
    def parameters(self):
        return self.spatial_model.parameters + self.spectral_model.parameters

    @property
    def spatial_model(self):
        """`~gammapy.modeling.models.SpatialModel`"""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        from .spatial import SpatialModel

        if not isinstance(model, SpatialModel):
            raise TypeError(f"Invalid type: {model!r}")
        self._spatial_model = model

    @property
    def spectral_model(self):
        """`~gammapy.modeling.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        from .spectral import SpectralModel

        if not isinstance(model, SpectralModel):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.spatial_model.position

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return self.spatial_model.evaluation_radius

    @property
    def frame(self):
        return self.spatial_model.frame

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"spatial_model={self.spatial_model!r}, "
            f"spectral_model={self.spectral_model!r})"
        )

    def evaluate(self, lon, lat, energy):
        """Evaluate the model at given points.

        The model evaluation follows numpy broadcasting rules.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        val_spatial = self.spatial_model(lon, lat)  # pylint:disable=not-callable
        val_spectral = self.spectral_model(energy)  # pylint:disable=not-callable
        return val_spatial * val_spectral

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`."""
        val_spatial = self.spatial_model.evaluate_geom(geom.to_image())
        energy = geom.get_axis_by_name("energy").center[:, np.newaxis, np.newaxis]
        val_spectral = self.spectral_model(energy)
        return val_spatial * val_spectral

    def copy(self, **kwargs):
        """Copy SkyModel"""
        kwargs.setdefault("spatial_model", self.spatial_model.copy())
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("name", self.name + "-copy")
        return self.__class__(**kwargs)

    def to_dict(self):
        """Create dict for YAML serilisation"""
        data = {}
        data["name"] = self.name
        data["type"] = self.tag
        data["spatial"] = self.spatial_model.to_dict()
        data["spectral"] = self.spectral_model.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        """Create SkyModel from dict"""
        from gammapy.modeling.models import SPATIAL_MODELS, SPECTRAL_MODELS

        model_class = SPECTRAL_MODELS.get_cls(data["spectral"]["type"])
        spectral_model = model_class.from_dict(data["spectral"])

        model_class = SPATIAL_MODELS.get_cls(data["spatial"]["type"])
        spatial_model = model_class.from_dict(data["spatial"])

        return cls(
            name=data["name"],
            spatial_model=spatial_model,
            spectral_model=spectral_model,
        )


class SkyDiffuseCube(SkyModelBase):
    """Cube sky map template model (3D).

    This is for a 3D map with an energy axis.
    Use `~gammapy.modeling.models.TemplateSpatialModel` for 2D maps.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    tilt : float
        Additional tilt in the spectrum
    reference : `~astropy.units.Quantity`
        Reference energy of the tilt.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.
    """

    tag = "SkyDiffuseCube"
    norm = Parameter("norm", 1)
    tilt = Parameter("tilt", 0, unit="", frozen=True)
    reference = Parameter("reference", "1 TeV", frozen=True)

    def __init__(
        self,
        map,
        norm=norm.quantity,
        tilt=tilt.quantity,
        reference=reference.quantity,
        meta=None,
        interp_kwargs=None,
        name="diffuse",
        filename=None,
    ):
        self.name = name
        axis = map.geom.get_axis_by_name("energy")

        if axis.node_type != "center":
            raise ValueError('Need a map with energy axis node_type="center"')

        self.map = map
        self.meta = {} if meta is None else meta
        self.filename = filename

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

        # TODO: onve we have implement a more general and better model caching
        #  remove this again
        self._cached_value = None
        self._cached_coordinates = (None, None, None)

        super().__init__(norm=norm, tilt=tilt, reference=reference)

    @classmethod
    def read(cls, filename, **kwargs):
        """Read map from FITS file.

        The default unit used if none is found in the file is ``cm-2 s-1 MeV-1 sr-1``.

        Parameters
        ----------
        filename : str
            FITS image filename.
        """
        m = Map.read(filename, **kwargs)
        if m.unit == "":
            m.unit = "cm-2 s-1 MeV-1 sr-1"
        name = Path(filename).stem
        return cls(m, name=name, filename=filename)

    def _interpolate(self, lon, lat, energy):
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
            "energy": energy,
        }
        return self.map.interp_by_coord(coord, **self._interp_kwargs)

    def evaluate(self, lon, lat, energy):
        """Evaluate model."""
        is_cached_coord = [
            _ is coord for _, coord in zip((lon, lat, energy), self._cached_coordinates)
        ]

        # reset cache
        if not np.all(is_cached_coord):
            self._cached_value = None

        if self._cached_value is None:
            self._cached_coordinates = (lon, lat, energy)
            self._cached_value = self._interpolate(lon, lat, energy)

        norm = self.parameters["norm"].value

        tilt = self.parameters["tilt"].value
        reference = self.parameters["reference"].quantity
        tilt_factor = np.power((energy / reference).to(""), -tilt)

        val = norm * self._cached_value * tilt_factor.value
        return u.Quantity(val, self.map.unit, copy=False)

    def copy(self):
        """A shallow copy"""
        return copy.copy(self)

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.map.geom.center_skydir

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return np.max(self.map.geom.width) / 2.0

    @property
    def frame(self):
        return self.position.frame.name

    @classmethod
    def from_dict(cls, data):
        model = cls.read(data["filename"])
        model._update_from_dict(data)
        return model

    def to_dict(self):
        data = super().to_dict()
        data["name"] = self.name
        data["type"] = data.pop("type")
        data["filename"] = self.filename

        # Move parameters at the end
        data["parameters"] = data.pop("parameters")
        return data


class BackgroundModel(Model):
    """Background model.

    Create a new map by a tilt and normalization on the available map

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Background model map
    norm : float
        Background normalization
    tilt : float
        Additional tilt in the spectrum
    reference : `~astropy.units.Quantity`
        Reference energy of the tilt.
    """

    tag = "BackgroundModel"
    norm = Parameter("norm", 1, unit="", min=0)
    tilt = Parameter("tilt", 0, unit="", frozen=True)
    reference = Parameter("reference", "1 TeV", frozen=True)

    def __init__(
        self,
        map,
        norm=norm.quantity,
        tilt=tilt.quantity,
        reference=reference.quantity,
        name="background",
        filename=None,
    ):
        axis = map.geom.get_axis_by_name("energy")
        if axis.node_type != "edges":
            raise ValueError('Need an integrated map, energy axis node_type="edges"')

        self.map = map
        self.name = name
        self.filename = filename
        super().__init__(norm=norm, tilt=tilt, reference=reference)

    @property
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.map.geom.get_axis_by_name("energy")
        energy = energy_axis.center
        return energy[:, np.newaxis, np.newaxis]

    def evaluate(self):
        """Evaluate background model.

        Returns
        -------
        background_map : `~gammapy.maps.Map`
            Background evaluated on the Map
        """
        norm = self.parameters["norm"].value
        tilt = self.parameters["tilt"].value
        reference = self.parameters["reference"].quantity
        tilt_factor = np.power((self.energy_center / reference).to(""), -tilt)
        back_values = norm * self.map.data * tilt_factor.value
        return self.map.copy(data=back_values)

    def to_dict(self):
        data = {}
        data["name"] = self.name
        data.update(super().to_dict())
        if self.filename is not None:
            data["filename"] = self.filename
        data["parameters"] = data.pop("parameters")
        return data

    @classmethod
    def from_dict(cls, data):
        if "filename" in data:
            map = Map.read(data["filename"])
        elif "map" in data:
            map = data["map"]
        else:
            raise ValueError("Requires either filename or `Map` object")

        model = cls(map=map, name=data["name"])
        model._update_from_dict(data)
        return model


def create_fermi_isotropic_diffuse_model(filename, **kwargs):
    """Read Fermi isotropic diffuse model.

    See `LAT Background models <https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html>`_

    Parameters
    ----------
    filename : str
        filename
    kwargs : dict
        Keyword arguments forwarded to `TemplateSpectralModel`

    Returns
    -------
    diffuse_model : `SkyModel`
        Fermi isotropic diffuse sky model.
    """
    from .spectral import TemplateSpectralModel
    from .spatial import ConstantSpatialModel

    vals = np.loadtxt(make_path(filename))
    energy = u.Quantity(vals[:, 0], "MeV", copy=False)
    values = u.Quantity(vals[:, 1], "MeV-1 s-1 cm-2", copy=False)

    spatial_model = ConstantSpatialModel()
    spectral_model = TemplateSpectralModel(energy=energy, values=values, **kwargs)
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="fermi-diffuse-iso",
    )
