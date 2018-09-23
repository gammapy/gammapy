# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import astropy.units as u
import operator
from ..utils.fitting import Parameters, Parameter
from ..utils.scripts import make_path
from ..maps import Map

__all__ = [
    "SkyModelBase",
    "SkyModels",
    "SkyModel",
    "CompoundSkyModel",
    "SkyDiffuseCube",
]


class SkyModelBase(object):
    """Sky model base class"""

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    def __add__(self, skymodel):
        return CompoundSkyModel(self, skymodel, operator.add)

    def __radd__(self, model):
        return self.__add__(model)

    def __call__(self, lon, lat, energy):
        return self.evaluate(lon, lat, energy)


class SkyModels(object):
    """Collection of `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    skymodels : list of `~gammapy.cube.models.SkyModel`
        Sky models

    Examples
    --------

    Read from an XML file::

        from gammapy.cube import SkyModels
        filename = '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml'
        sourcelib = SkyModels.read(filename)
    """

    def __init__(self, skymodels):
        self.skymodels = skymodels

        pars = []
        for skymodel in skymodels:
            for p in skymodel.parameters:
                pars.append(p)
        self._parameters = Parameters(pars)

    @property
    def parameters(self):
        """Concatenated parameters.

        Currently no way to distinguish spectral and spatial.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        idx = 0
        for skymodel in self.skymodels:
            n_par = len(skymodel.parameters.parameters)
            skymodel.parameters.parameters = parameters.parameters[idx : idx + n_par]
            idx += n_par

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

    def to_compound_model(self):
        """Return `~gammapy.cube.models.CompoundSkyModel`"""
        return np.sum([m for m in self.skymodels])

    def evaluate(self, lon, lat, energy):
        out = self.skymodels[0].evaluate(lon, lat, energy)
        for skymodel in self.skymodels[1:]:
            out += skymodel.evaluate(lon, lat, energy)
        return out


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

    def __init__(self, spatial_model, spectral_model, name="SkyModel"):
        self.name = name
        self._spatial_model = spatial_model
        self._spectral_model = spectral_model
        self._parameters = Parameters(
            spatial_model.parameters.parameters + spectral_model.parameters.parameters
        )

    @property
    def spatial_model(self):
        """`~gammapy.image.models.SkySpatialModel`"""
        # propagate sub-covariance
        if self.parameters.covariance is not None:
            idx = len(self._spatial_model.parameters.parameters)
            self._spatial_model.parameters.covariance = self.parameters.covariance[
                :idx, :idx
            ]
        return self._spatial_model

    @property
    def spectral_model(self):
        """`~gammapy.spectrum.models.SpectralModel`"""
        # propagate sub-covariance
        if self.parameters.covariance is not None:
            idx = len(self._spatial_model.parameters.parameters)
            self._spectral_model.parameters.covariance = self.parameters.covariance[
                idx:, idx:
            ]
        return self._spectral_model

    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.Parameters`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        idx = len(self.spatial_model.parameters.parameters)
        self._spatial_model.parameters.parameters = parameters.parameters[:idx]
        self._spectral_model.parameters.parameters = parameters.parameters[idx:]

    def __repr__(self):
        fmt = "{}(spatial_model={!r}, spectral_model={!r})"
        return fmt.format(
            self.__class__.__name__, self.spatial_model, self.spectral_model
        )

    def __str__(self):
        ss = "{}\n\n".format(self.__class__.__name__)
        ss += "spatial_model = {}\n\n".format(self.spatial_model)
        ss += "spectral_model = {}\n".format(self.spectral_model)
        return ss

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
        val = val_spatial * val_spectral
        # TODO: shall remove hard coded return units? If really needed users can
        # always do this themselves. For fitting this also adds a performance penalty...
        return val.to("cm-2 s-1 TeV-1 deg-2")


class CompoundSkyModel(SkyModelBase):
    """Represents the algebraic combination of two
    `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    model1, model2 : `SkyModel`
        Two sky models
    operator : callable
        Binary operator to combine the models
    """

    def __init__(self, model1, model2, operator):
        self.model1 = model1
        self.model2 = model2
        self.operator = operator
        self._parameters = Parameters(
            self.model1.parameters.parameters + self.model2.parameters.parameters
        )

    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.Parameters`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters
        idx = len(self.model1.parameters.parameters)
        self.model1.parameters.parameters = parameters.parameters[:idx]
        self.model2.parameters.parameters = parameters.parameters[idx:]

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n    Component 1 : {}".format(self.model1)
        ss += "\n    Component 2 : {}".format(self.model2)
        ss += "\n    Operator : {}".format(self.operator)
        return ss

    def evaluate(self, lon, lat, energy):
        """Evaluate the compound model at given points.

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
        val1 = self.model1.evaluate(lon, lat, energy)
        val2 = self.model2.evaluate(lon, lat, energy)
        return self.operator(val1, val2)


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
        Interpolation keyword arguments passed to `Map.interp_by_coord()`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.

    """

    def __init__(self, map, norm=1, meta=None, interp_kwargs=None):
        axis = map.geom.get_axis_by_name("energy")

        if axis.node_type != "center":
            raise ValueError('Need a map with energy axis node_type="center"')

        self.map = map
        self.parameters = Parameters([Parameter("norm", norm)])
        self.meta = {} if meta is None else meta

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

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
        return cls(m)

    def evaluate(self, lon, lat, energy):
        """Evaluate model."""
        coord = {
            "lon": lon.to("deg").value,
            "lat": lat.to("deg").value,
            "energy": energy,
        }
        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        norm = self.parameters["norm"].value
        return u.Quantity(norm * val, self.map.unit, copy=False)

    def copy(self):
        """A shallow copy"""
        return copy.copy(self)
