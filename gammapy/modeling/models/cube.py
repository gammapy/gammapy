# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube models (axes: lon, lat, energy)."""
import copy
from pathlib import Path
import numpy as np
import astropy.units as u
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.utils.scripts import make_name, make_path
from .core import Model, Models


class SkyModelBase(Model):
    """Sky model base class"""

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([self, *other])
        elif isinstance(other, (SkyModel, SkyDiffuseCube)):
            return Models([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy):
        return self.evaluate(lon, lat, energy)

    def evaluate_geom(self, geom):
        coords = geom.get_coord(frame=self.frame)
        return self(coords.lon, coords.lat, coords["energy_true"])


class SkyModel(SkyModelBase):
    """Sky model component.

    This model represents a factorised sky model.
    It has `~gammapy.modeling.Parameters`
    combining the spatial and spectral parameters.

    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    temporal_model : `~gammapy.modeling.models.temporalModel`
        Temporal model
    name : str
        Model identifier
    """

    tag = "SkyModel"
    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
        self,
        spectral_model,
        spatial_model=None,
        temporal_model=None,
        name=None,
        apply_irf=None,
        datasets_names=None,
    ):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        self.temporal_model = temporal_model
        super().__init__()
        # TODO: this hack is needed for compound models to work
        self.__dict__.pop("_parameters")

        self._name = make_name(name)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        self.datasets_names = datasets_names

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        parameters = []

        if self.spatial_model is not None:
            parameters.append(self.spatial_model.parameters)

        parameters.append(self.spectral_model.parameters)
        return Parameters.from_stack(parameters)

    @property
    def spatial_model(self):
        """`~gammapy.modeling.models.SpatialModel`"""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        from .spatial import SpatialModel

        if not (model is None or isinstance(model, SpatialModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._spatial_model = model

    @property
    def spectral_model(self):
        """`~gammapy.modeling.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        from .spectral import SpectralModel

        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def temporal_model(self):
        """`~gammapy.modeling.models.TemporalModel`"""
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, model):
        from .temporal import TemporalModel

        if not (model is None or isinstance(model, TemporalModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._temporal_model = model

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
            f"temporal_model={self.temporal_model!r})"
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
        value = self.spectral_model(energy)  # pylint:disable=not-callable
        # TODO: case if self.temporal_model is not None, introduce time in arguments ?

        if self.spatial_model is not None:
            value = value * self.spatial_model(lon, lat)  # pylint:disable=not-callable

        return value

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`."""
        energy = geom.get_axis_by_name("energy_true").center[:, np.newaxis, np.newaxis]
        value = self.spectral_model(energy)
        # TODO: case with temporal_model is not None
        if self.spatial_model is not None:
            value = value * self.spatial_model.evaluate_geom(geom.to_image())

        return value

    def copy(self, name=None, **kwargs):
        """Copy SkyModel"""
        if self.spatial_model is not None:
            spatial_model = self.spatial_model.copy()
        else:
            spatial_model = None

        if self.temporal_model is not None:
            temporal_model = self.temporal_model.copy()
        else:
            temporal_model = None

        kwargs.setdefault("name", make_name(name))
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("spatial_model", spatial_model)
        kwargs.setdefault("temporal_model", temporal_model)
        kwargs.setdefault("apply_irf", self.apply_irf.copy())
        kwargs.setdefault("datasets_names", self.datasets_names)

        return self.__class__(**kwargs)

    def to_dict(self):
        """Create dict for YAML serilisation"""
        data = {}
        data["name"] = self.name
        data["type"] = self.tag
        data["spectral"] = self.spectral_model.to_dict()

        if self.spatial_model is not None:
            data["spatial"] = self.spatial_model.to_dict()

        if self.temporal_model is not None:
            data["temporal"] = self.temporal_model.to_dict()

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        return data

    @classmethod
    def from_dict(cls, data):
        """Create SkyModel from dict"""
        from gammapy.modeling.models import (
            SPATIAL_MODELS,
            SPECTRAL_MODELS,
            TEMPORAL_MODELS,
        )

        model_class = SPECTRAL_MODELS.get_cls(data["spectral"]["type"])
        spectral_model = model_class.from_dict(data["spectral"])

        spatial_data = data.get("spatial")
        if spatial_data is not None:
            model_class = SPATIAL_MODELS.get_cls(spatial_data["type"])
            spatial_model = model_class.from_dict(spatial_data)
        else:
            spatial_model = None

        temporal_data = data.get("temporal")
        if temporal_data is not None:
            model_class = TEMPORAL_MODELS.get_cls(temporal_data["type"])
            temporal_model = model_class.from_dict(temporal_data)
        else:
            temporal_model = None

        return cls(
            name=data["name"],
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            temporal_model=temporal_model,
            apply_irf=data.get("apply_irf", cls._apply_irf_default),
            datasets_names=data.get("datasets_names"),
        )

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\t{:26}: {}\n".format("Name", self.name)

        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)

        str_ += "\t{:26}: {}\n".format("Spectral model type", self.spectral_model.tag)

        if self.spatial_model is not None:
            spatial_type = self.spatial_model.tag
        else:
            spatial_type = "None"
        str_ += "\t{:26}: {}\n".format("Spatial  model type", spatial_type)

        if self.temporal_model is not None:
            temporal_type = self.temporal_model.tag
        else:
            temporal_type = "None"
        str_ += "\t{:26}: {}\n".format("Temporal model type", temporal_type)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)


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

    _apply_irf_default = {"exposure": True, "psf": True, "edisp": True}

    def __init__(
        self,
        map,
        norm=norm.quantity,
        tilt=tilt.quantity,
        reference=reference.quantity,
        meta=None,
        interp_kwargs=None,
        name=None,
        filename=None,
        apply_irf=None,
        datasets_names=None,
    ):

        self._name = make_name(name)

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

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        self.datasets_names = datasets_names
        super().__init__(norm=norm, tilt=tilt, reference=reference)

    @property
    def name(self):
        return self._name

    @classmethod
    def read(cls, filename, name=None, **kwargs):
        """Read map from FITS file.

        The default unit used if none is found in the file is ``cm-2 s-1 MeV-1 sr-1``.

        Parameters
        ----------
        filename : str
            FITS image filename.
        name : str
            Name of the output model
            The default used if none is filename.
        """
        m = Map.read(filename, **kwargs)
        if m.unit == "":
            m.unit = "cm-2 s-1 MeV-1 sr-1"
        if name is None:
            name = Path(filename).stem

        if m.geom.axes[0].name == "energy":
            m.geom.axes[0].name = "energy_true"

        return cls(m, name=name, filename=filename)

    def _interpolate(self, lon, lat, energy):
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
            "energy_true": energy,
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

    def copy(self, name=None):
        """A shallow copy"""
        new = copy.copy(self)
        new._name = make_name(name)
        return new

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
        apply_irf = data.get("apply_irf", cls._apply_irf_default)
        model.apply_irf.update(apply_irf)
        model.datasets_names = data.get("datasets_names")

        return model

    def to_dict(self):
        data = super().to_dict()
        data["name"] = self.name
        data["type"] = data.pop("type")
        data["filename"] = self.filename

        # Move parameters at the end
        data["parameters"] = data.pop("parameters")

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        return data

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\t{:26}: {}\n".format("Name", self.name)
        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)


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
        name=None,
        filename=None,
        datasets_names=None,
    ):
        axis = map.geom.get_axis_by_name("energy")
        if axis.node_type != "edges":
            raise ValueError('Need an integrated map, energy axis node_type="edges"')

        self.map = map

        self._name = make_name(name)
        self.filename = filename

        if isinstance(datasets_names, list):
            if len(datasets_names) > 1:
                raise ValueError(
                    "Currently background models can only be assigned to one dataset."
                )

        self.datasets_names = datasets_names
        super().__init__(norm=norm, tilt=tilt, reference=reference)

    @property
    def name(self):
        return self._name

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

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        return data

    @classmethod
    def from_dict(cls, data):
        if "filename" in data:
            map = Map.read(data["filename"])
        elif "map" in data:
            map = data["map"]
        else:
            raise ValueError("Requires either filename or `Map` object")
        model = cls(
            map=map,
            name=data["name"],
            datasets_names=data.get("datasets_names"),
            filename=data.get("filename"),
        )
        model._update_from_dict(data)
        return model

    def copy(self, name=None):
        """A deep copy."""
        new = copy.deepcopy(self)
        new._name = make_name(name)
        return new

    def cutout(self, position, width, mode="trim", name=None):
        """Cutout background model.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        name : str
            Name of the returned background model.

        Returns
        -------
        cutout : `BackgroundModel`
            Cutout background model.
        """
        cutout_kwargs = {"position": position, "width": width, "mode": mode}

        bkg_map = self.map.cutout(**cutout_kwargs)
        model = self.__class__(bkg_map, name=name)
        model.parameters.update_from_dict(self.parameters.to_dict())
        return model

    def stack(self, other, weights=None):
        """Stack background model in place.

        Stacking the background model resets the current parameters values.

        Parameters
        ----------
        other : `BackgroundModel`
            Other background model.
        """
        bkg = self.evaluate()
        other_bkg = other.evaluate()
        bkg.stack(other_bkg, weights=weights)
        self.map = bkg

        # reset parameter values
        self.norm.value = 1
        self.tilt.value = 0

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"
        str_ += "\t{:26}: {}\n".format("Name", self.name)
        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.map.geom.center_skydir

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return np.max(self.map.geom.width) / 2.0


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
