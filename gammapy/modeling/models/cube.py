# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Cube models (axes: lon, lat, energy)."""

import logging
import warnings
import os
import numpy as np
import astropy.units as u
from astropy.nddata import NoOverlapError
from astropy.time import Time
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Parameters
from gammapy.modeling.covariance import CovarianceMixin
from gammapy.modeling.parameter import _get_parameters_str
from gammapy.utils.compat import COPY_IF_NEEDED
from gammapy.utils.fits import LazyFitsData
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.deprecation import GammapyDeprecationWarning
from .core import Model, ModelBase, Models
from .spatial import ConstantSpatialModel, SpatialModel
from .spectral import PowerLawNormSpectralModel, SpectralModel, TemplateSpectralModel
from .temporal import TemporalModel

log = logging.getLogger(__name__)


__all__ = [
    "create_fermi_isotropic_diffuse_model",
    "FoVBackgroundModel",
    "SkyModel",
    "TemplateNPredModel",
]


class SkyModel(CovarianceMixin, ModelBase):
    """Sky model component.

    This model represents a factorised sky model.
    It has `~gammapy.modeling.Parameters` combining the spatial and spectral parameters.

    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model.
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1).
    temporal_model : `~gammapy.modeling.models.TemporalModel`
        Temporal model.
    name : str
        Model identifier.
    apply_irf : dict
        Dictionary declaring which IRFs should be applied to this model. Options
        are {"exposure": True, "psf": True, "edisp": True}.
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
        covariance_data=None,
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

        super().__init__(covariance_data=covariance_data)

    @property
    def _models(self):
        models = self.spectral_model, self.spatial_model, self.temporal_model
        return [model for model in models if model is not None]

    def _check_unit(self):
        axis = MapAxis.from_energy_bounds(
            "0.1 TeV", "10 TeV", nbin=1, name="energy_true"
        )

        geom = WcsGeom.create(skydir=self.position, npix=(2, 2), axes=[axis])
        time = Time(55555, format="mjd")
        if self.apply_irf["exposure"]:
            ref_unit = u.Unit("cm-2 s-1 MeV-1")
        else:
            ref_unit = u.Unit("")
        obt_unit = self.spectral_model(axis.center).unit

        if self.spatial_model:
            obt_unit = obt_unit * self.spatial_model.evaluate_geom(geom).unit
            ref_unit = ref_unit / u.sr

        if self.temporal_model:
            if u.Quantity(self.temporal_model(time)).unit.is_equivalent(
                self.spectral_model(axis.center).unit
            ):
                obt_unit = (
                    (
                        self.temporal_model(time)
                        * self.spatial_model.evaluate_geom(geom).unit
                    )
                    .to(obt_unit.to_string())
                    .unit
                )
            else:
                obt_unit = obt_unit * u.Quantity(self.temporal_model(time)).unit

        if not obt_unit.is_equivalent(ref_unit):
            raise ValueError(
                f"SkyModel unit {obt_unit} is not equivalent to {ref_unit}"
            )

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
    def parameters_unique_names(self):
        """List of unique parameter names. Return formatted as par_type.par_name."""
        names = []
        for model in self._models:
            for par_name in model.parameters_unique_names:
                components = [model.type, par_name]
                name = ".".join(components)
                names.append(name)
        return names

    @property
    def spatial_model(self):
        """Spatial model as a `~gammapy.modeling.models.SpatialModel` object."""
        return self._spatial_model

    @spatial_model.setter
    def spatial_model(self, model):
        if not (model is None or isinstance(model, SpatialModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._spatial_model = model

    @property
    def spectral_model(self):
        """Spectral model as a `~gammapy.modeling.models.SpectralModel` object."""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def temporal_model(self):
        """Temporal model as a `~gammapy.modeling.models.TemporalModel` object."""
        return self._temporal_model

    @temporal_model.setter
    def temporal_model(self, model):
        if not (model is None or isinstance(model, TemporalModel)):
            raise TypeError(f"Invalid type: {model!r}")

        self._temporal_model = model

    @property
    def position(self):
        """Position as a `~astropy.coordinates.SkyCoord`."""
        return getattr(self.spatial_model, "position", None)

    @property
    def position_lonlat(self):
        """Spatial model center position `(lon, lat)` in radians and frame of the model."""
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
        """Evaluation radius as an `~astropy.coordinates.Angle`."""
        return self.spatial_model.evaluation_radius

    @property
    def evaluation_region(self):
        """Evaluation region as an `~astropy.coordinates.Angle`."""
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
        """Check if a sky model contributes within a mask map.

        Parameters
        ----------
        mask : `~gammapy.maps.WcsNDMap` of boolean type
            Map containing a boolean mask.
        margin : `~astropy.units.Quantity`
            Add a margin in degree to the source evaluation radius.
            Used to take into account PSF width.


        Returns
        -------
        models : `DatasetModels`
            Selected models contributing inside the region where mask is True.
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
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``.

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates.
        energy : `~astropy.units.Quantity`
            Energy coordinate.
        time: `~astropy.time.Time`, optional
            Time coordinate. Default is None.

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

        additional_axes = set(coords.axis_names) - {
            "lon",
            "lat",
            "energy_true",
            "time",
        }
        for axis in additional_axes:
            value = value * np.ones_like(coords[axis])

        if self.spatial_model:
            value = value * self.spatial_model.evaluate_geom(geom)

        if self.temporal_model:
            value = self._compute_time_integral(value, geom, gti)

        return value

    def integrate_geom(self, geom, gti=None, oversampling_factor=None):
        """Integrate model on `~gammapy.maps.Geom`.

        See `~gammapy.modeling.models.SpatialModel.integrate_geom` and
        `~gammapy.modeling.models.SpectralModel.integral`.

        Parameters
        ----------
        geom : `Geom` or `~gammapy.maps.RegionGeom`
            Map geometry.
        gti : `GTI`, optional
            GIT table. Default is None.
        oversampling_factor : int, optional
            The oversampling factor to use for spatial integration.
            Default is None: the factor is estimated from the model minimal bin size.

        Returns
        -------
        flux : `Map`
            Predicted flux map.
        """
        energy = geom.axes["energy_true"].edges
        shape = len(geom.data_shape) * [
            1,
        ]
        shape[geom.axes.index_data("energy_true")] = -1
        value = self.spectral_model.integral(
            energy[:-1],
            energy[1:],
        ).reshape(shape)

        if self.spatial_model:
            value = (
                value
                * self.spatial_model.integrate_geom(
                    geom, oversampling_factor=oversampling_factor
                ).quantity
            )

        if self.temporal_model:
            value = self._compute_time_integral(value, geom, gti)

        value = value * np.ones(geom.data_shape)

        return Map.from_geom(geom=geom, data=value.value, unit=value.unit)

    def _compute_time_integral(self, value, geom, gti):
        """Multiply input value with time integral for the given geometry and GTI."""
        if "time" in geom.axes.names:
            if geom.axes.names[-1] != "time":
                raise ValueError(
                    "Incorrect axis order. The time axis must be the last axis"
                )
            time_axis = geom.axes["time"]

            temp_eval = np.ones(time_axis.nbin)
            for idx in range(time_axis.nbin):
                if gti is None:
                    t1, t2 = time_axis.time_min[idx], time_axis.time_max[idx]
                else:
                    gti_in_bin = gti.select_time(
                        time_interval=[
                            time_axis.time_min[idx],
                            time_axis.time_max[idx],
                        ]
                    )
                    t1, t2 = gti_in_bin.time_start, gti_in_bin.time_stop
                integral = self.temporal_model.integral(t1, t2)
                temp_eval[idx] = np.sum(integral)
            value = (value.T * temp_eval).T

        else:
            if gti is not None:
                integral = self.temporal_model.integral(gti.time_start, gti.time_stop)
                value = value * np.sum(integral)
        return value

    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy sky model.

        Parameters
        ----------
        name : str, optional
            Assign a new name to the copied model. Default is None.
        copy_data : bool, optional
            Copy the data arrays attached to models. Default is False.
        **kwargs : dict
            Keyword arguments forwarded to `SkyModel`.

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
        kwargs.setdefault("covariance_data", self.covariance.data.copy())

        return self.__class__(**kwargs)

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serilisation."""
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
    def from_dict(cls, data, **kwargs):
        """Create SkyModel from dictionary."""
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
            Tag to create spectral model.
        spatial_model : str, optional
            Tag to create spatial model. Default is None.
        temporal_model : str, optional
            Tag to create temporal model. Default is None.
        **kwargs : dict
            Keyword arguments passed to `SkyModel`.

        Returns
        -------
        model : SkyModel
            Sky model.
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
        """Freeze parameters depending on model type.

        Parameters
        ----------
        model_type : {None, "spatial", "spectral", "temporal"}
           Freeze all parameters or only spatial/spectral/temporal.
           Default is None, such that all parameters are frozen.
        """
        if model_type is None:
            self.parameters.freeze_all()
        else:
            model = getattr(self, f"{model_type}_model")
            model.freeze()

    def unfreeze(self, model_type=None):
        """Restore parameters frozen status to default depending on model type.

        Parameters
        ----------
        model_type : {None, "spatial", "spectral", "temporal"}
           Restore frozen status to default for all parameters or only spatial/spectral/temporal.
           Default is None, such that all parameters are restored to default frozen status.

        """
        if model_type is None:
            for model_type in ["spectral", "spatial", "temporal"]:
                self.unfreeze(model_type)
        else:
            model = getattr(self, f"{model_type}_model")
            if model:
                model.unfreeze()


class FoVBackgroundModel(ModelBase):
    """Field of view background model.

    The background model holds the correction parameters applied to
    the instrumental background attached to a `MapDataset` or
    `SpectrumDataset`.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    spectral_model : `~gammapy.modeling.models.SpectralModel`, Optional
        Normalized spectral model.
        Default is `~gammapy.modeling.models.PowerLawNormSpectralModel`
    spatial_model : `~gammapy.modeling.models.SpatialModel`, Optional
        Unitless Spatial model (unit is dropped on evaluation if defined).
        Default is None.
    """

    tag = ["FoVBackgroundModel", "fov-bkg"]

    def __init__(
        self,
        dataset_name,
        spectral_model=None,
        spatial_model=None,
        covariance_data=None,
    ):
        # TODO: remove this in v2.0
        if isinstance(dataset_name, SpectralModel):
            warnings.warn(
                "dataset_name has been made first argument since v1.3.",
                GammapyDeprecationWarning,
                stacklevel=2,
            )
            buf = dataset_name
            dataset_name = spectral_model
            spectral_model = buf

        self.datasets_names = [dataset_name]

        if spectral_model is None:
            spectral_model = PowerLawNormSpectralModel()

        if not spectral_model.is_norm_spectral_model:
            raise ValueError("A norm spectral model is required.")

        self._spatial_model = spatial_model
        self._spectral_model = spectral_model
        super().__init__(covariance_data=covariance_data)

    @staticmethod
    def contributes(*args, **kwargs):
        """FoV background models always contribute."""
        return True

    @property
    def spectral_model(self):
        """Spectral norm model."""
        return self._spectral_model

    @property
    def spatial_model(self):
        """Spatial norm model."""
        return self._spatial_model

    @property
    def _models(self):
        models = self.spectral_model, self.spatial_model
        return [model for model in models if model is not None]

    @property
    def name(self):
        """Model name."""
        return self.datasets_names[0] + "-bkg"

    @property
    def parameters(self):
        """Model parameters."""
        parameters = []
        parameters.append(self.spectral_model.parameters)
        if self.spatial_model is not None:
            parameters.append(self.spatial_model.parameters)
        return Parameters.from_stack(parameters)

    @property
    def parameters_unique_names(self):
        """List of unique parameter names. Return formatted as par_type.par_name."""
        names = []
        for model in self._models:
            for par_name in model.parameters_unique_names:
                components = [model.type, par_name]
                name = ".".join(components)
                names.append(name)
        return names

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n\n"

        str_ += "\t{:26}: {}\n".format("Name", self.name)
        str_ += "\t{:26}: {}\n".format("Datasets names", self.datasets_names)
        str_ += "\t{:26}: {}\n".format(
            "Spectral model type", self.spectral_model.__class__.__name__
        )
        if self.spatial_model is not None:
            str_ += "\t{:26}: {}\n".format(
                "Spatial model type", self.spatial_model.__class__.__name__
            )
        str_ += "\tParameters:\n"
        info = _get_parameters_str(self.parameters)
        lines = info.split("\n")
        str_ += "\t" + "\n\t".join(lines[:-1])

        str_ += "\n\n"
        return str_.expandtabs(tabsize=2)

    def evaluate_geom(self, geom):
        """Evaluate map."""
        coords = geom.get_coord(sparse=True)
        return self.evaluate(**coords._data)

    def evaluate(self, energy, lon=None, lat=None):
        """Evaluate model."""
        value = self.spectral_model(energy)
        if self.spatial_model is not None:
            if lon is not None and lat is not None:
                if self.spatial_model.is_energy_dependent:
                    return self.spatial_model(lon, lat, energy).value * value
                else:
                    return self.spatial_model(lon, lat).value * value
            else:
                raise ValueError(
                    "lon and lat are required if a spatial model is defined"
                )
        else:
            return value

    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy the `FoVBackgroundModel` instance.

        Parameters
        ----------
        name : str, optional
            Ignored, present for API compatibility.
            Default is None.
        copy_data : bool, optional
            Ignored, present for API compatibility.
            Default is False.
        **kwargs : dict
            Keyword arguments forwarded to `FoVBackgroundModel`.

        Returns
        -------
        model : `FoVBackgroundModel`
            Copied FoV background model.
        """
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("dataset_name", self.datasets_names[0])
        kwargs.setdefault("covariance_data", self.covariance.data.copy())
        if self.spatial_model is not None:
            kwargs.setdefault("spatial_model", self.spatial_model.copy())
        return self.__class__(**kwargs)

    def to_dict(self, full_output=False):
        data = {}
        data["type"] = self.tag[0]
        data["datasets_names"] = self.datasets_names
        data.update(self.spectral_model.to_dict(full_output=full_output))
        if self.spatial_model is not None:
            data.update(self.spatial_model.to_dict(full_output))
        return data

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create model from dictionary.

        Parameters
        ----------
        data : dict
            Data dictionary.
        """
        from gammapy.modeling.models import (
            SPATIAL_MODEL_REGISTRY,
            SPECTRAL_MODEL_REGISTRY,
        )

        spectral_data = data.get("spectral")
        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict({"spectral": spectral_data})
        else:
            spectral_model = None

        spatial_data = data.get("spatial")
        if spatial_data is not None:
            model_class = SPATIAL_MODEL_REGISTRY.get_cls(spatial_data["type"])
            spatial_model = model_class.from_dict({"spatial": spatial_data})
        else:
            spatial_model = None

        datasets_names = data.get("datasets_names")

        if datasets_names is None:
            raise ValueError("FoVBackgroundModel must define a dataset name")

        if len(datasets_names) > 1:
            raise ValueError("FoVBackgroundModel can only be assigned to one dataset")

        return cls(
            spatial_model=spatial_model,
            spectral_model=spectral_model,
            dataset_name=datasets_names[0],
        )

    def reset_to_default(self):
        """Reset parameter values to default."""
        values = self.spectral_model.default_parameters.value
        self.spectral_model.parameters.value = values

    def freeze(self, model_type="spectral"):
        """Freeze model parameters."""
        if model_type is None or model_type == "spectral":
            self._spectral_model.freeze()

    def unfreeze(self, model_type="spectral"):
        """Restore parameters frozen status to default."""
        if model_type is None or model_type == "spectral":
            self._spectral_model.unfreeze()


class TemplateNPredModel(ModelBase):
    """Background model.

    Create a new map by a tilt and normalization on the available map.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Background model map.
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Normalized spectral model.
        Default is `~gammapy.modeling.models.PowerLawNormSpectralModel`.
    copy_data : bool
        Create a deepcopy of the map data or directly use the original. Default is True.
        Use False to save memory in case of large maps.
    spatial_model : `~gammapy.modeling.models.SpatialModel`
        Unitless Spatial model (unit is dropped on evaluation if defined).
        Default is None.
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
        spatial_model=None,
        covariance_data=None,
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

        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

        if isinstance(datasets_names, str):
            datasets_names = [datasets_names]

        if isinstance(datasets_names, list):
            if len(datasets_names) != 1:
                raise ValueError(
                    "Currently background models can only be assigned to one dataset."
                )
        self.datasets_names = datasets_names
        super().__init__(covariance_data=covariance_data)

    def copy(self, name=None, copy_data=False, **kwargs):
        """Copy template npred model.

        Parameters
        ----------
        name : str, optional
            Assign a new name to the copied model.
            Default is None.
        copy_data : bool, optional
            Copy the data arrays attached to models.
            Default is False.
        **kwargs : dict
            Keyword arguments forwarded to `TemplateNPredModel`.

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
        kwargs.setdefault("covariance_data", self.covariance.data.copy())
        return self.__class__(name=name, copy_data=copy_data, **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def energy_center(self):
        """True energy axis bin centers as a `~astropy.units.Quantity`."""
        energy_axis = self.map.geom.axes["energy"]
        energy = energy_axis.center
        return energy[:, np.newaxis, np.newaxis]

    @property
    def spectral_model(self):
        """Spectral model as a `~gammapy.modeling.models.SpectralModel` object."""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if not (model is None or isinstance(model, SpectralModel)):
            raise TypeError(f"Invalid type: {model!r}")
        self._spectral_model = model

    @property
    def _models(self):
        models = self.spectral_model, self.spatial_model
        return [model for model in models if model is not None]

    @property
    def parameters(self):
        parameters = []
        parameters.append(self.spectral_model.parameters)
        return Parameters.from_stack(parameters)

    @property
    def parameters_unique_names(self):
        """List of unique parameter names. Return formatted as par_type.par_name."""
        names = []
        for model in self._models:
            for par_name in model.parameters_unique_names:
                components = [model.type, par_name]
                name = ".".join(components)
                names.append(name)
        return names

    def evaluate(self):
        """Evaluate background model.

        Returns
        -------
        background_map : `~gammapy.maps.Map`
            Background evaluated on the Map.
        """
        value = self.spectral_model(self.energy_center).value
        back_values = self.map.data * value
        if self.spatial_model is not None:
            value = self.spatial_model.evaluate_geom(self.map.geom).value
            back_values *= value
        return self.map.copy(data=back_values)

    def to_dict(self, full_output=False):
        data = {}
        data["name"] = self.name
        data["type"] = self.tag
        if self.spatial_model is not None:
            data["spatial"] = self.spatial_model.to_dict(full_output)["spatial"]
        data["spectral"] = self.spectral_model.to_dict(full_output)["spectral"]

        if self.filename is not None:
            data["filename"] = self.filename

        if self.datasets_names is not None:
            data["datasets_names"] = self.datasets_names

        return data

    def write(self, overwrite=False):
        """
        Write the map.

        Parameters
        ----------
        overwrite: bool, optional
            Overwrite existing file.
            Default is False, which will raise a warning if the template file exists already.
        """
        if self.filename is None:
            raise IOError("Missing filename")
        elif os.path.isfile(make_path(self.filename)) and not overwrite:
            log.warning("Template file already exits, and overwrite is False")
        else:
            self.map.write(self.filename, overwrite=overwrite)

    @classmethod
    def from_dict(cls, data, **kwargs):
        from gammapy.modeling.models import (
            SPATIAL_MODEL_REGISTRY,
            SPECTRAL_MODEL_REGISTRY,
        )

        spectral_data = data.get("spectral")
        if spectral_data is not None:
            model_class = SPECTRAL_MODEL_REGISTRY.get_cls(spectral_data["type"])
            spectral_model = model_class.from_dict({"spectral": spectral_data})
        else:
            spectral_model = None

        spatial_data = data.get("spatial")
        if spatial_data is not None:
            model_class = SPATIAL_MODEL_REGISTRY.get_cls(spatial_data["type"])
            spatial_model = model_class.from_dict({"spatial": spatial_data})
        else:
            spatial_model = None

        if "filename" in data:
            bkg_map = Map.read(data["filename"])
        else:
            raise IOError("Missing filename")

        return cls(
            map=bkg_map,
            spatial_model=spatial_model,
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
            Default is "trim".
        name : str, optional
            Name of the returned background model. Default is None.

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
        weights : float, optional
            Weights. Default is None.
        nan_to_num: bool, optional
            Non-finite values are replaced by zero if True. Default is True.
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

    def slice_by_energy(self, energy_min=None, energy_max=None, name=None):
        """Select and slice model template in energy range

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds of the slice. Default is None.
        name : str
            Name of the sliced model. Default is None.

        Returns
        -------
        model : `TemplateNpredModel`
            Sliced Model.

        """
        name = make_name(name)

        energy_axis = self.map._geom.axes["energy"]

        if energy_min is None:
            energy_min = energy_axis.bounds[0]

        if energy_max is None:
            energy_max = energy_axis.bounds[1]

        if name is None:
            name = self.name

        energy_min, energy_max = u.Quantity(energy_min), u.Quantity(energy_max)

        group = energy_axis.group_table(edges=[energy_min, energy_max])

        is_normal = group["bin_type"] == "normal   "
        group = group[is_normal]

        slices = {
            "energy": slice(int(group["idx_min"][0]), int(group["idx_max"][0]) + 1)
        }

        model = self.copy(name=name)
        model.map = model.map.slice_by_idx(slices=slices)
        return model

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
        """Position as a `~astropy.coordinates.SkyCoord`."""
        return self.map.geom.center_skydir

    @property
    def evaluation_radius(self):
        """Evaluation radius as a `~astropy.coordinates.Angle`."""
        return np.max(self.map.geom.width) / 2.0

    def freeze(self, model_type="spectral"):
        """Freeze model parameters."""
        if model_type is None or model_type == "spectral":
            self._spectral_model.freeze()

    def unfreeze(self, model_type="spectral"):
        """Restore parameters frozen status to default."""
        if model_type is None or model_type == "spectral":
            self._spectral_model.unfreeze()


def create_fermi_isotropic_diffuse_model(filename, **kwargs):
    """Read Fermi isotropic diffuse model.

    See `LAT Background models <https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html>`__.

    Parameters
    ----------
    filename : str
        Filename.
    kwargs : dict
        Keyword arguments forwarded to `TemplateSpectralModel`.

    Returns
    -------
    diffuse_model : `SkyModel`
        Fermi isotropic diffuse sky model.
    """
    vals = np.loadtxt(make_path(filename))
    energy = u.Quantity(vals[:, 0], "MeV", copy=COPY_IF_NEEDED)
    values = u.Quantity(vals[:, 1], "MeV-1 s-1 cm-2", copy=COPY_IF_NEEDED)

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
