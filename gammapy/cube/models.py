# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from pathlib import Path
import numpy as np
import astropy.units as u
from ..utils.fitting import Parameter, Model, Parameters
from ..spectrum.models import SpectralModel
from ..image.models import SkySpatialModel
from ..utils.scripts import make_path, write_yaml
from ..maps import Map

__all__ = [
    "SkyModelBase",
    "SkyModels",
    "SkyModel",
    "SkyDiffuseCube",
    "BackgroundModel",
    "BackgroundModels",
]


class SkyModelBase(Model):
    """Sky model base class"""

    def __add__(self, skymodel):
        skymodels = [self]
        if isinstance(skymodel, SkyModels):
            skymodels += skymodel.skymodels
        elif isinstance(skymodel, (SkyModel, SkyDiffuseCube)):
            skymodels += [skymodel]
        else:
            raise NotImplementedError
        return SkyModels(skymodels)

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy):
        return self.evaluate(lon, lat, energy)


class SkyModels:
    """Collection of `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    skymodels : list of `~gammapy.cube.models.SkyModel`
        Sky models

    Examples
    --------
    Read from an XML file::

        from gammapy.cube import SkyModels
        filename = '$GAMMAPY_DATA/tests/models/fermi_model.xml'
        sourcelib = SkyModels.read(filename)
    """

    frame = None

    __slots__ = ["skymodels"]

    def __init__(self, skymodels):

        existing_names = []

        for model in skymodels:
            if model.name in existing_names:
                raise ValueError(
                    "SkyModel '{}' already exists, please choose"
                    " another name.".format(model.name)
                )
            existing_names.append(model.name)

        self.skymodels = skymodels

    @property
    def parameters(self):
        parameters = []
        for skymodel in self.skymodels:
            for p in skymodel.parameters:
                parameters.append(p)
        return Parameters(parameters)

    @property
    def names(self):
        """Sky model names"""
        return [_.name for _ in self.skymodels]

    @classmethod
    def from_xml(cls, xml):
        """Read from XML string."""
        from ..utils.serialization import xml_to_sky_models

        return xml_to_sky_models(xml)

    @classmethod
    def read(cls, filename):
        """Read from XML file.

        The XML definition of some models is uncompatible with the models
        currently implemented in gammapy. Therefore the following modifications
        happen to the XML model definition

        * PowerLaw: The spectral index is negative in XML but positive in
          gammapy. Parameter limits are ignored

        * ExponentialCutoffPowerLaw: The cutoff energy is transferred to
          lambda = 1 / cutof energy on read
        """
        path = make_path(filename)
        xml = path.read_text()
        return cls.from_xml(xml)

    def to_xml(self, filename):
        """Write to XML file."""
        from ..utils.serialization import sky_models_to_xml

        xml = sky_models_to_xml(self)
        filename = make_path(filename)
        with filename.open("w") as output:
            output.write(xml)

    @classmethod
    def from_yaml(cls, filename):
        """Write to YAML file."""
        from ..utils.serialization import dict_to_models
        from ..utils.scripts import read_yaml

        data = read_yaml(filename)
        skymodels = dict_to_models(data)
        return cls(skymodels)

    def to_yaml(self, filename, selection="all"):
        """Write to YAML file."""
        from ..utils.serialization import models_to_dict

        components_dict = models_to_dict(self.skymodels, selection)
        write_yaml(components_dict, filename)

    def evaluate(self, lon, lat, energy):
        out = self.skymodels[0].evaluate(lon, lat, energy)
        for skymodel in self.skymodels[1:]:
            out += skymodel.evaluate(lon, lat, energy)
        return out

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"

        for idx, skymodel in enumerate(self.skymodels):
            str_ += "Component {idx}: {skymodel}\n\n\t".format(
                idx=idx, skymodel=skymodel
            )
            str_ += "\n\n"

        if self.parameters.covariance is not None:
            str_ += "\n\nCovariance: \n\n\t"
            covariance = self.parameters.covariance_to_table()
            str_ += "\n\t".join(covariance.pformat())
        return str_

    def __iadd__(self, skymodel):
        if isinstance(skymodel, SkyModels):
            self.skymodels += skymodel.skymodels
        elif isinstance(skymodel, (SkyModel, SkyDiffuseCube)):
            self.skymodels += [skymodel]
        else:
            raise NotImplementedError
        return self

    def __add__(self, skymodel):
        skymodels = self.skymodels.copy()
        if isinstance(skymodel, SkyModels):
            skymodels += skymodel.skymodels
        elif isinstance(skymodel, (SkyModel, SkyDiffuseCube)):
            skymodels += [skymodel]
        else:
            raise NotImplementedError
        return SkyModels(skymodels)

    def __getitem__(self, item):
        idx = self.names.index(item)
        return self.skymodels[idx]


class SkyModel(SkyModelBase):
    """Sky model component.

    This model represents a factorised sky model.
    It has a `~gammapy.utils.modeling.Parameters`
    combining the spatial and spectral parameters.

    TODO: add possibility to have a temporal model component also.

    Parameters
    ----------
    spatial_model : `~gammapy.image.models.SkySpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    name : str
        Model identifier
    """

    __slots__ = ["name", "_spatial_model", "_spectral_model"]

    def __init__(self, spatial_model, spectral_model, name="source"):
        self.name = name

        if not isinstance(spatial_model, SkySpatialModel):
            raise ValueError(
                "Spatial model must be instance / subclass "
                " of `SkySpatialModel` and not {}.".format(
                    spatial_model.__class__.__name__
                )
            )

        self._spatial_model = spatial_model

        if not isinstance(spectral_model, SpectralModel):
            raise ValueError(
                "Spectral model model must be instance / subclass "
                "of `SpectralModel` and not {}.".format(
                    spatial_model.__class__.__name__
                )
            )

        self._spectral_model = spectral_model

        parameters = (
            spatial_model.parameters.parameters + spectral_model.parameters.parameters
        )
        super().__init__(parameters)

    @property
    def spatial_model(self):
        """`~gammapy.image.models.SkySpatialModel`"""
        return self._spatial_model

    @property
    def spectral_model(self):
        """`~gammapy.spectrum.models.SpectralModel`"""
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        """`~gammapy.spectrum.models.SpectralModel`"""
        self._spectral_model = model
        self._parameters = Parameters(
            self.spatial_model.parameters.parameters
            + self.spectral_model.parameters.parameters
        )

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
        fmt = "{}(spatial_model={!r}, spectral_model={!r})"
        return fmt.format(
            self.__class__.__name__, self.spatial_model, self.spectral_model
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

    def copy(self, **kwargs):
        """Copy SkyModel"""
        kwargs.setdefault("spatial_model", self.spatial_model.copy())
        kwargs.setdefault("spectral_model", self.spectral_model.copy())
        kwargs.setdefault("name", self.name + "-copy")
        return self.__class__(**kwargs)


class SkyDiffuseCube(SkyModelBase):
    """Cube sky map template model (3D).

    This is for a 3D map with an energy axis. Use `~gammapy.image.models.SkyDiffuseMap`
    for 2D maps.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.

    """

    __slots__ = ["map", "norm", "meta", "_interp_kwargs"]

    def __init__(
        self,
        map,
        norm=1,
        meta=None,
        interp_kwargs=None,
        name="diffuse",
        filename=None,
        obs_id="Global",
    ):
        self.name = name
        axis = map.geom.get_axis_by_name("energy")

        if axis.node_type != "center":
            raise ValueError('Need a map with energy axis node_type="center"')

        self.map = map
        self.norm = Parameter("norm", norm)
        self.meta = {} if meta is None else meta
        self.filename = filename
        self.obs_id = obs_id

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

        super().__init__([self.norm])

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

    def evaluate(self, lon, lat, energy):
        """Evaluate model."""
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
            "energy": energy,
        }
        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        norm = self.parameters["norm"].value
        return u.Quantity(norm * val, self.map.unit, copy=False)

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
        radius = np.max(self.map.geom.width) / 2.0
        return radius

    @property
    def frame(self):
        return self.position.frame.name


class BackgroundModel(Model):
    """Background model.

    Create a new map by a tilt and normalisation on the available map

    Parameters
    ----------
    background : `~gammapy.maps.Map`
        Background model map
    norm : float
        Background normalisation
    tilt : float
        Additional tilt in the spectrum
    reference : `~astropy.units.Quantity`
        Reference energy of the tilt.
    """

    __slots__ = ["map", "norm", "tilt", "reference", "name", "filename", "obs_id"]

    def __init__(
        self,
        background,
        norm=1,
        tilt=0,
        reference="1 TeV",
        name="background",
        filename=None,
        obs_id=None,
    ):
        axis = background.geom.get_axis_by_name("energy")
        if axis.node_type != "edges":
            raise ValueError('Need an integrated map, energy axis node_type="edges"')

        self.map = background
        self.norm = Parameter("norm", norm, unit="", min=0)
        self.tilt = Parameter("tilt", tilt, unit="", frozen=True)
        self.reference = Parameter("reference", reference, frozen=True)
        self.name = name
        self.filename = filename
        self.obs_id = obs_id
        super().__init__([self.norm, self.tilt, self.reference])

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

    @classmethod
    def from_skymodel(cls, skymodel, exposure, edisp=None, psf=None, **kwargs):
        """Create background model from sky model by applying IRFs.

        Typically used for diffuse Galactic or constant emission models.

        Parameters
        ----------
        skymodel : `~gammapy.cube.models.SkyModel` or `~gammapy.cube.models.SkyDiffuseCube`
            Sky model
        exposure : `~gammapy.maps.Map`
            Exposure map
        edisp : `~gammapy.irf.EnergyDispersion`
            Energy dispersion
        psf : `~gammapy.cube.PSFKernel`
            PSF kernel
        """
        from .fit import MapEvaluator

        evaluator = MapEvaluator(
            model=skymodel, exposure=exposure, edisp=edisp, psf=psf
        )
        background = evaluator.compute_npred()
        background_model = cls(background=background, **kwargs)
        background_model.name = skymodel.name
        if skymodel.__class__.__name__ == "SkyDiffuseCube":
            background_model.filename = skymodel.filename
        return background_model

    def __add__(self, model):
        models = [self]
        if isinstance(model, BackgroundModels):
            models += model.models
        elif isinstance(model, BackgroundModel):
            models += [model]
        else:
            raise NotImplementedError
        return BackgroundModels(models)


class BackgroundModels(Model):
    """Background models.

    Parameters
    ----------
    models : list of `BackgroundModel`
        List of background models.
    """

    __slots__ = ["models", "_parameters"]

    def __init__(self, models):
        self.models = models
        parameters = []
        for model in models:
            for p in model.parameters:
                parameters.append(p)
        super().__init__(parameters)

    def evaluate(self):
        """Evaluate background models."""
        for idx, model in enumerate(self.models):
            if idx == 0:
                vals = model.evaluate()
            else:
                vals += model.evaluate()
        return vals

    def __iadd__(self, model):
        if isinstance(model, BackgroundModels):
            self.models += model.models
        elif isinstance(model, BackgroundModel):
            self.models += [model]
        else:
            raise NotImplementedError
        return self

    def __add__(self, model):
        model_ = self.copy()
        model_ += model
        return model_

    def to_yaml(self, filename, selection="all"):
        """Write to yaml file."""
        from ..utils.serialization import models_to_dict

        components_dict = models_to_dict(self.models, selection)
        write_yaml(components_dict, filename)
