# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube models (axes: lon, lat, energy)."""

import numpy as np
import astropy.units as u
from astropy.nddata import NoOverlapError
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Covariance, Parameters
from gammapy.modeling.covariance import copy_covariance
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.utils.fits import LazyFitsData
from gammapy.utils.scripts import make_name, make_path
from .core import Model, ModelBase, Models
from .spatial import ConstantSpatialModel, SpatialModel
from .spectral import PowerLawNormSpectralModel, SpectralModel, TemplateSpectralModel
from .temporal import TemporalModel

__all__ = [
    "create_fermi_isotropic_diffuse_model",
    "FoVBackgroundModel",
    "SkyModel",
    "TemplateNPredModel",
]


class SkyModel(ModelBase):
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
    apply_irf : dict
        Dictionary declaring which IRFs should be applied to this model. Options
        are {"exposure": True, "psf": True, "edisp": True}
    datasets_names : list of str
        Which datasets this model is applied to.
    """

    tag = ["SkyModel", "sky-model"]
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
        self._name = make_name(name)

        if apply_irf is None:
            apply_irf = self._apply_irf_default.copy()

        self.apply_irf = apply_irf
        self.datasets_names = datasets_names
        self._check_unit()

        is_norm = np.array([par.is_norm for par in spectral_model.parameters])

        if not np.any(is_norm):
            raise ValueError(
                "Spectral model used with SkyModel requires a norm type parameter."
            )

        super().__init__()

    @property
    def _models(self):
        models = self.spectral_model, self.spatial_model, self.temporal_model
        return [model for model in models if model is not None]

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models],
            )

    def _check_unit(self):
        from gammapy.data.gti import GTI

        # evaluate over a test geom to check output unit
        # TODO simpler way to test this ?
        axis = MapAxis.from_energy_bounds(
            "0.1 TeV", "10 TeV", nbin=1, name="energy_true"
        )

        geom = WcsGeom.create(skydir=self.position, npix=(2, 2), axes=[axis])

        gti = GTI.create(1 * u.day, 2 * u.day)
        value = self.evaluate_geom(geom, gti)

        if self.apply_irf["exposure"]:
            ref_unit = u.Unit("cm-2 s-1 MeV-1 sr-1")
        else:
            ref_unit = u.Unit("sr-1")

        if self.spatial_model is None:
            ref_unit = ref_unit / u.Unit("sr-1")

        if not value.unit.is_equivalent(ref_unit):
            raise ValueError(
                f"SkyModel unit {value.unit} is not equivalent to {ref_unit}"
            )

    @property
    def covariance(self):
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            subcovar = self._covariance.get_subcovariance(model.covariance.parameters)
            model.covariance = subcovar

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        parameters = []

        parameters.append(self.spectral_model.parameters)

        if self.spatial_model is not None:
            parameters.append(self.spatial_model.parameters)

        if self.temporal_model is not None:
            parameters.append(self.temporal_model.parameters)

        return Parameters.from_stack(parameters)

    @property
    def spatial_model(self):
        """`~gammapy.modeling.models.SpatialModel`"""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        if not (model is None or isinstance(model, SpatialModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._spatial_model = model

    @property
    def spectral_model(self):
        """`~gammapy.modeling.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def temporal_model(self):
        """`~gammapy.modeling.models.TemporalModel`"""
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, model):
        if not (model is None or isinstance(model, TemporalModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._temporal_model = model

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return getattr(self.spatial_model, "position", None)

    @property
    def position_lonlat(self):
        """Spatial model center position `(lon, lat)` in rad and frame of the model"""
        return getattr(self.spatial_model, "position_lonlat", None)

    @property
    def evaluation_bin_size_min(self):
        """Minimal spatial bin size for spatial model evaluation."""
        if (
            self.spatial_model is not None
            and self.spatial_model.evaluation_bin_size_min is not None
        ):
            return self.spatial_model.evaluation_bin_size_min
        else:
            return None

    @property
    def evaluation_radius(self):
        """`~astropy.coordinates.Angle`"""
        return self.spatial_model.evaluation_radius

    @property
    def evaluation_region(self):
        """`~astropy.coordinates.Angle`"""
        return self.spatial_model.evaluation_region

    @property
    def frame(self):
        return self.spatial_model.frame

    def __add__(self, other):
        if isinstance(other, (Models, list)):
            return Models([self, *other])
        elif isinstance(other, (SkyModel, TemplateNPredModel)):
            return Models([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy, time=None):
        return self.evaluate(lon, lat, energy, time)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"spatial_model={self.spatial_model!r}, "
            f"spectral_model={self.spectral_model!r})"
            f"temporal_model={self.temporal_model!r})"
        )

    def contributes(self, mask, margin="0 deg"):
        """Check if a skymodel contributes within a mask map.

        Parameters
        ----------
        mask : `~gammapy.maps.WcsNDMap` of boolean type
            Map containing a boolean mask
        margin : `~astropy.units.Quantity`
            Add a margin in degree to the source evaluation radius.
            Used to take into account PSF width.


        Returns
        -------
        models : `DatasetModels`
            Selected models contributing inside the region where mask==True
        """
        from gammapy.datasets.evaluator import CUTOUT_MARGIN

        margin = u.Quantity(margin)

        if not mask.geom.is_image:
            mask = mask.reduce_over_axes(func=np.logical_or)

        if mask.geom.is_region and mask.geom.region is not None:
            if mask.geom.is_all_point_sky_regions:
                return True

            geom = mask.geom.to_wcs_geom()
            mask = geom.region_mask([mask.geom.region])

        try:
            mask_cutout = mask.cutout(
                position=self.position,
                width=(2 * self.evaluation_radius + CUTOUT_MARGIN + margin),
            )
            contributes = np.any(mask_cutout.data)
        except (NoOverlapError, ValueError):
            contributes = False

        return contributes

    def evaluate(self, lon, lat, energy, time=None):
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
        time: `~astropy.time.Time`
            Time coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        value = self.spectral_model(energy)  # pylint:disable=not-callable
        # TODO: case if self.temporal_model is not None, introduce time in arguments ?

        if self.spatial_model is not None:
            if self.spatial_model.is_energy_dependent:
                spatial = self.spatial_model(lon, lat, energy)
            else:
                spatial = self.spatial_model(lon, lat)

            value = value * spatial  # pylint:disable=not-callable

        if (self.temporal_model is not None) and (time is not None):
            value = value * self.temporal_model(time)

        return value

    def evaluate_geom(self, geom, gti=None):
        """Evaluate model on `~gammapy.maps.Geom`."""
        coords = geom.get_coord(sparse=True)
        value = self.spectral_model(coords["energy_true"])

        if self.spatial_model:
            value = value * self.spatial_model.evaluate_geom(geom)

        if self.temporal_model:
            integral = self.temporal_model.integral(gti.time_start, gti.time_stop)
            value = value * np.sum(integral)

        return value

    def integrate_geom(self, geom, gti=None, oversampling_factor=None):
        """Integrate model on `~gammapy.maps.Geom`.

        See `~gammapy.modeling.models.SpatialModel.integrate_geom` and
        `~gammapy.modeling.models.SpectralModel.integral`.

        Parameters
        ----------
        geom : `Geom` or `~gammapy.maps.RegionGeom`
            Map geometry
        gti : `GTI`
            GIT table
        oversampling_factor : int or None
            The oversampling factor to use for spatial integration.
            Default is None: the factor is estimated from the model minimal bin size

        Returns
        -------
        flux : `Map`
            Predicted flux map
        """
        energy = geom.axes["energy_true"].edges
        value = self.spectral_model.integral(
            energy[:-1],
            energy[1:],
        ).reshape((-1, 1, 1))

        if self.spatial_model:
            value = (
                value
                * self.spatial_model.integrate_geom(
                    geom, oversampling_factor=oversampling_factor
                ).quantity
            )

        if self.temporal_model:
            integral = self.temporal_model.integral(gti.time_start, gti.time_stop)
            value = value * np.sum(integral)

        return Map.from_geom(geom=geom, data=value.value, unit=value.unit)

    @copy_covariance
    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy sky model

        Parameters
        ----------
        name : str
            Assign a new name to the copied model.
        copy_data : bool
            Copy the data arrays attached to models.
        **kwargs : dict
            Keyword arguments forwarded to `SkyModel`

        Returns
        -------
        model : `SkyModel`
            Copied sky model.
        """
        if self.spatial_model is not None:
            spatial_model = self.spatial_model.copy(copy_data=copy_data)
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

    def to_dict(self, full_output=False):
        """Create dict for YAML serilisation"""
        data = {}
        data["name"] = self.name
        data["type"] = self.tag[0]

        if self.apply_irf != self._apply_irf_default:
            data["apply_irf"] = self.apply_irf

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        data.update(self.spectral_model.to_dict(full_output))

        if self.spatial_model is not None:
            data.update(self.spatial_model.to_dict(full_output))

        if self.temporal_model is not None:
            data.update(self.temporal_model.to_dict(full_output))

        return data

    @classmethod
    def from_dict(cls, data):
        """Create SkyModel from dict"""
        from gammapy.modeling.models import (
            SPATIAL_MODEL_REGISTRY,
            SPECTRAL_MODEL_REGISTRY,
            TEMPORAL_MODEL_REGISTRY,
        )

        model_class = SPECTRAL_MODEL_REGISTRY.get_cls(data["spectral"]["type"])
        spectral_model = model_class.from_dict({"spectral": data["spectral"]})

        spatial_data = data.get("spatial")

        if spatial_data is not None:
            model_class = SPATIAL_MODEL_REGISTRY.get_cls(spatial_data["type"])
            spatial_model = model_class.from_dict({"spatial": spatial_data})
        else:
            spatial_model = None

        temporal_data = data.get("temporal")

        if temporal_data is not None:
            model_class = TEMPORAL_MODEL_REGISTRY.get_cls(temporal_data["type"])
            temporal_model = model_class.from_dict({"temporal": temporal_data})
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
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)

        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)

        str_ += "\t{:26}: {}\n".format(
            "Spectral model type", self.spectral_model.__class__.__name__
        )

        if self.spatial_model is not None:
            spatial_type = self.spatial_model.__class__.__name__
        else:
            spatial_type = ""
        str_ += "\t{:26}: {}\n".format("Spatial  model type", spatial_type)

        if self.temporal_model is not None:
            temporal_type = self.temporal_model.__class__.__name__
        else:
            temporal_type = ""
        str_ += "\t{:26}: {}\n".format("Temporal model type", temporal_type)

        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    @classmethod
    def create(cls, spectral_model, spatial_model=None, temporal_model=None, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        spectral_model : str
            Tag to create spectral model
        spatial_model : str
            Tag to create spatial model
        temporal_model : str
            Tag to create temporal model
        **kwargs : dict
            Keyword arguments passed to `SkyModel`

        Returns
        -------
        model : SkyModel
            Sky model
        """
        spectral_model = Model.create(spectral_model, model_type="spectral")

        if spatial_model:
            spatial_model = Model.create(spatial_model, model_type="spatial")

        if temporal_model:
            temporal_model = Model.create(temporal_model, model_type="temporal")

        return cls(
            spectral_model=spectral_model,
            spatial_model=spatial_model,
            temporal_model=temporal_model,
            **kwargs,
        )

    def freeze(self, model_type=None):
        """Freeze parameters depending on model type

        Parameters
        ----------
        model_type : {None, "spatial", "spectral", "temporal"}
           freeze all parameters or only or only spatial/spectral/temporal.
           Default is None so all parameters are frozen.
        """
        if model_type is None:
            self.parameters.freeze_all()
        else:
            model = getattr(self, f"{model_type}_model")
            model.freeze()

    def unfreeze(self, model_type=None):
        """Restore parameters frozen status to default depending on model type

        Parameters
        ----------
        model_type : {None, "spatial", "spectral", "temporal"}
           restore frozen status to default for all parameters or only spatial/spectral/temporal
           Default is None so all parameters are restore to default frozen status.

        """
        if model_type is None:
            for model_type in ["spectral", "spatial", "temporal"]:
                self.unfreeze(model_type)
        else:
            model = getattr(self, f"{model_type}_model")
            if model:
                model.unfreeze()


class FoVBackgroundModel(ModelBase):
    """Field of view background model

    The background model holds the correction parameters applied to
    the instrumental background attached to a `MapDataset` or
    `SpectrumDataset`.

    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Normalized spectral model.
    dataset_name : str
        Dataset name

    """

    tag = ["FoVBackgroundModel", "fov-bkg"]

    def __init__(self, spectral_model=None, dataset_name=None):
        if dataset_name is None:
            raise ValueError("Dataset name a is required argument")

        self.datasets_names = [dataset_name]

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()

        if not spectral_model.is_norm_spectral_model:
            raise ValueError("A norm spectral model is required.")

        self._spectral_model = spectral_model
        super().__init__()

    @staticmethod
    def contributes(*args, **kwargs):
        """FoV background models always contribute"""
        return True

    @property
    def spectral_model(self):
        """Spectral norm model"""
        return self._spectral_model

    @property
    def name(self):
        """Model name"""
        return self.datasets_names[0] + "-bkg"

    @property
    def parameters(self):
        """Model parameters"""
        parameters = []
        parameters.append(self.spectral_model.parameters)
        return Parameters.from_stack(parameters)

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)
        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)
        str_ += "\t{:26}: {}\n".format(
            "Spectral model type", self.spectral_model.__class__.__name__
        )
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    def evaluate_geom(self, geom):
        """Evaluate map"""
        coords = geom.get_coord(sparse=True)
        return self.evaluate(energy=coords["energy"])

    def evaluate(self, energy):
        """Evaluate model"""
        return self.spectral_model(energy)

    @copy_covariance
    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy FoVBackgroundModel

        Parameters
        ----------
        name : str
            Ignored, present for API compatibility.
        copy_data : bool
            Ignored, present for API compatibility.
        **kwargs : dict
            Keyword arguments forwarded to `FoVBackgroundModel`

        Returns
        -------
        model : `FoVBackgroundModel`
            Copied FoV background model.
        """
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("dataset_name", self.datasets_names[0])
        return self.__class__(**kwargs)

    def to_dict(self, full_output=False):
        data = {}
        data["type"] = self.tag[0]
        data["datasets_names"] = self.datasets_names
        data["spectral"] = self.spectral_model.to_dict(full_output=full_output)[
            "spectral"
        ]
        return data

    @classmethod
    def from_dict(cls, data):
        """Create model from dict

        Parameters
        ----------
        data : dict
            Data dictionary
        """
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        spectral_data = data.get("spectral")
        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict(spectral_data)
        else:
            spectral_model = None

        datasets_names = data.get("datasets_names")

        if datasets_names is None:
            raise ValueError("FoVBackgroundModel must define a dataset name")

        if len(datasets_names) > 1:
            raise ValueError("FoVBackgroundModel can only be assigned to one dataset")

        return cls(
            spectral_model=spectral_model,
            dataset_name=datasets_names[0],
        )

    def reset_to_default(self):
        """Reset parameter values to default"""
        values = self.spectral_model.default_parameters.value
        self.spectral_model.parameters.value = values

    def freeze(self, model_type="spectral"):
        """Freeze model parameters"""
        if model_type is None or model_type == "spectral":
            self._spectral_model.freeze()

    def unfreeze(self, model_type="spectral"):
        """Restore parameters frozen status to default"""
        if model_type is None or model_type == "spectral":
            self._spectral_model.unfreeze()


class TemplateNPredModel(ModelBase):
    """Background model.

    Create a new map by a tilt and normalization on the available map

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Background model map
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Normalized spectral model,
        default is `~gammapy.modeling.models.PowerLawNormSpectralModel`
    copy_data : bool
        Create a deepcopy of the map data or directly use the original. True by
        default, can be turned to False to save memory in case of large maps.
    """

    tag = "TemplateNPredModel"
    map = LazyFitsData(cache=True)

    def __init__(
        self,
        map,
        spectral_model=None,
        name=None,
        filename=None,
        datasets_names=None,
        copy_data=True,
    ):
        if isinstance(map, Map):
            axis = map.geom.axes["energy"]
            if axis.node_type != "edges":
                raise ValueError(
                    'Need an integrated map, energy axis node_type="edges"'
                )

        if copy_data:
            self.map = map.copy()
        else:
            self.map = map

        self._name = make_name(name)
        self.filename = filename

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()
            spectral_model.tilt.frozen = True

        self.spectral_model = spectral_model

        if isinstance(datasets_names, str):
            datasets_names = [datasets_names]

        if isinstance(datasets_names, list):
            if len(datasets_names) != 1:
                raise ValueError(
                    "Currently background models can only be assigned to one dataset."
                )
        self.datasets_names = datasets_names
        super().__init__()

    @copy_covariance
    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy template npred model.

        Parameters
        ----------
        name : str
            Assign a new name to the copied model.
        copy_data : bool
            Copy the data arrays attached to models.
        **kwargs : dict
            Keyword arguments forwarded to `TemplateNPredModel`

        Returns
        -------
        model : `TemplateNPredModel`
            Copied template npred model.
        """
        name = make_name(name)
        kwargs.setdefault("map", self.map)
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("filename", self.filename)
        kwargs.setdefault("datasets_names", self.datasets_names)
        return self.__class__(name=name, copy_data=copy_data, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.map.geom.axes["energy"]
        energy = energy_axis.center
        return energy[:, np.newaxis, np.newaxis]

    @property
    def spectral_model(self):
        """`~gammapy.modeling.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def parameters(self):
        parameters = []
        parameters.append(self.spectral_model.parameters)
        return Parameters.from_stack(parameters)

    def evaluate(self):
        """Evaluate background model.

        Returns
        -------
        background_map : `~gammapy.maps.Map`
            Background evaluated on the Map
        """
        value = self.spectral_model(self.energy_center).value
        back_values = self.map.data * value
        return self.map.copy(data=back_values)

    def to_dict(self, full_output=False):
        data = {}
        data["name"] = self.name
        data["type"] = self.tag
        data["spectral"] = self.spectral_model.to_dict(full_output)["spectral"]

        if self.filename is not None:
            data["filename"] = self.filename

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        return data

    @classmethod
    def from_dict(cls, data):
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        spectral_data = data.get("spectral")

        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict(spectral_data)
        else:
            spectral_model = None

        if "filename" in data:
            bkg_map = Map.read(data["filename"])
        elif "map" in data:
            bkg_map = data["map"]
        else:
            # TODO: for now create a fake map for serialization,
            # uptdated in MapDataset.from_dict()
            axis = MapAxis.from_edges(np.logspace(-1, 1, 2), unit=u.TeV, name="energy")
            geom = WcsGeom.create(
                skydir=(0, 0), npix=(1, 1), frame="galactic", axes=[axis]
            )
            bkg_map = Map.from_geom(geom)

        return cls(
            map=bkg_map,
            spectral_model=spectral_model,
            name=data["name"],
            datasets_names=data.get("datasets_names"),
            filename=data.get("filename"),
        )

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
        cutout : `TemplateNPredModel`
            Cutout background model.
        """
        cutout_kwargs = {"position": position, "width": width, "mode": mode}

        bkg_map = self.map.cutout(**cutout_kwargs)
        spectral_model = self.spectral_model.copy()
        return self.__class__(bkg_map, spectral_model=spectral_model, name=name)

    def stack(self, other, weights=None, nan_to_num=True):
        """Stack background model in place.

        Stacking the background model resets the current parameters values.

        Parameters
        ----------
        other : `TemplateNPredModel`
            Other background model.
        nan_to_num: bool
            Non-finite values are replaced by zero if True (default).
        """
        bkg = self.evaluate()
        if nan_to_num:
            bkg.data[~np.isfinite(bkg.data)] = 0
        other_bkg = other.evaluate()
        bkg.stack(other_bkg, weights=weights, nan_to_num=nan_to_num)
        self.map = bkg

        # reset parameter values
        self.spectral_model.norm.value = 1
        self.spectral_model.tilt.value = 0

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

    def freeze(self, model_type="spectral"):
        """Freeze model parameters"""
        if model_type is None or model_type == "spectral":
            self._spectral_model.freeze()

    def unfreeze(self, model_type="spectral"):
        """Restore parameters frozen status to default"""
        if model_type is None or model_type == "spectral":
            self._spectral_model.unfreeze()


def create_fermi_isotropic_diffuse_model(filename, **kwargs):
    """Read Fermi isotropic diffuse model.

    See `LAT Background models <https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html>`__  # noqa: E501

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
    vals = np.loadtxt(make_path(filename))
    energy = u.Quantity(vals[:, 0], "MeV", copy=False)
    values = u.Quantity(vals[:, 1], "MeV-1 s-1 cm-2", copy=False)

    kwargs.setdefault("interp_kwargs", {"fill_value": None})

    spatial_model = ConstantSpatialModel()
    spectral_model = (
        TemplateSpectralModel(energy=energy, values=values, **kwargs)
        * PowerLawNormSpectralModel()
    )
    return SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="fermi-diffuse-iso",
        apply_irf={"psf": False, "exposure": True, "edisp": True},
    )
