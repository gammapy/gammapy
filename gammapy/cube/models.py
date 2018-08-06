# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import astropy.units as u
import operator
from astropy.utils import lazyproperty
from ..utils.modeling import ParameterList, Parameter
from ..utils.scripts import make_path
from ..maps import Map

__all__ = [
    'SourceLibrary',
    'SkyModel',
    'CompoundSkyModel',
    'SumSkyModel',
    'MapEvaluator',
    'SkyDiffuseCube',
]


class SourceLibrary(object):
    """Collection of `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    skymodels : list of `~gammapy.cube.models.SkyModel`
        Sky models

    Examples
    --------

    Read a SourceLibrary from an XML file::

        from gammapy.cube import SourceLibrary
        filename = '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml'
        sourcelib = SourceLibrary.from_xml(filename)
    """

    def __init__(self, skymodels):
        self.skymodels = skymodels

    @classmethod
    def from_xml(cls, xml):
        """Read SourceLibrary from XML string"""
        from ..utils.serialization import xml_to_source_library
        return xml_to_source_library(xml)

    @classmethod
    def read(cls, filename):
        """Read SourceLibrary from XML file

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
        """Write SourceLibrary to XML file"""
        from ..utils.serialization import source_library_to_xml
        xml = source_library_to_xml(self)
        filename = make_path(filename)
        with filename.open('w') as output:
            output.write(xml)

    def to_compound_model(self):
        """Return `~gammapy.cube.models.CompoundSkyModel`"""
        return np.sum([m for m in self.skymodels])

    def to_sum_model(self):
        """Return `~gammapy.cube.models.SumSkyModel`"""
        return SumSkyModel(self.skymodels)


class SkyModel(object):
    """Sky model component.

    This model represents a factorised sky model.
    It has a `~gammapy.utils.modeling.ParameterList`
    combining the spatial and spectral parameters.

    TODO: add possibility to have a temporal model component also.

    Parameters
    ----------
    spatial_model : `~gammapy.image.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    name : str
        Model identifier
    """

    def __init__(self, spatial_model, spectral_model, name='SkyModel'):
        self.name = name
        self._spatial_model = spatial_model
        self._spectral_model = spectral_model
        self._parameters = ParameterList(
            spatial_model.parameters.parameters +
            spectral_model.parameters.parameters
        )

    @property
    def spatial_model(self):
        """`~gammapy.image.models.SkySpatialModel`"""
        return self._spatial_model

    @property
    def spectral_model(self):
        """`~gammapy.spectrum.models.SpectralModel`"""
        return self._spectral_model

    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.ParameterList`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        idx = len(self.spatial_model.parameters.parameters)
        self._spatial_model.parameters.parameters = parameters.parameters[:idx]
        self._spectral_model.parameters.parameters = parameters.parameters[idx:]

    def __repr__(self):
        fmt = '{}(spatial_model={!r}, spectral_model={!r})'
        return fmt.format(self.__class__.__name__,
                          self.spatial_model, self.spectral_model)

    def __str__(self):
        ss = '{}\n\n'.format(self.__class__.__name__)
        ss += 'spatial_model = {}\n\n'.format(self.spatial_model)
        ss += 'spectral_model = {}\n'.format(self.spectral_model)
        return ss

    def evaluate(self, lon, lat, energy):
        """Evaluate the model at given points.

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
        val_spatial = self.spatial_model(lon, lat)
        val_spectral = self.spectral_model(energy)
        val_spectral = np.atleast_1d(val_spectral)[:, np.newaxis, np.newaxis]

        val = val_spatial * val_spectral

        return val.to('cm-2 s-1 TeV-1 deg-2')

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    def __add__(self, skymodel):
        return CompoundSkyModel(self, skymodel, operator.add)

    def __radd__(self, model):
        return self.__add__(model)


class CompoundSkyModel(object):
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

    # TODO: Think about how to deal with covariance matrix
    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.ParameterList`)"""
        return ParameterList(
            self.model1.parameters.parameters +
            self.model2.parameters.parameters
        )

    @parameters.setter
    def parameters(self, parameters):
        idx = len(self.model1.parameters.parameters)
        self.model1.parameters.parameters = parameters.parameters[:idx]
        self.model2.parameters.parameters = parameters.parameters[idx:]

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n    Component 1 : {}'.format(self.model1)
        ss += '\n    Component 2 : {}'.format(self.model2)
        ss += '\n    Operator : {}'.format(self.operator)
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


class SumSkyModel(object):
    """Sum of independent `SkyModel` components.

    Not sure if we want this class, or only a + operator on SkyModel.
    If we keep it, then probably SkyModel should become an ABC
    and the current SkyModel renamed to SkyModelFactorised or something like that?

    Parameters
    ----------
    components : list
        List of SkyModel objects
    """

    def __init__(self, components):
        self.components = components
        pars = []
        for model in self.components:
            for p in model.parameters.parameters:
                pars.append(p)
        self._parameters = ParameterList(pars)

    @property
    def parameters(self):
        """Concatenated parameters.

        Currently no way to distinguish spectral and spatial.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        idx = 0
        for component in self.components:
            n_par = len(component.parameters.parameters)
            component.parameters.parameters = parameters.parameters[idx:idx + n_par]
            idx += n_par

    def evaluate(self, lon, lat, energy):
        out = self.components[0].evaluate(lon, lat, energy)
        for component in self.components[1:]:
            out += component.evaluate(lon, lat, energy)
        return out


class MapEvaluator(object):
    """Sky model evaluation on maps.

    This is a first attempt to compute flux as well as predicted counts maps.

    The basic idea is that this evaluator is created once at the start
    of the analysis, and pre-computes some things.
    It it then evaluated many times during likelihood fit when model parameters
    change, re-using pre-computed quantities each time.
    At the moment it does some things, e.g. cache and re-use energy and coordinate grids,
    but overall it is not an efficient implementation yet.

    For now, we only make it work for 3D WCS maps with an energy axis.
    No HPX, no other axes, those can be added later here or via new
    separate model evaluator classes.

    We should discuss how to organise the model and IRF evaluation code,
    and things like integrations and convolutions in a good way.

    Parameters
    ----------
    sky_model : `~gammapy.cube.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    background : `~gammapy.maps.Map`
        background map
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(self, sky_model=None, exposure=None, background=None, psf=None, edisp=None):
        self.sky_model = sky_model
        self.exposure = exposure
        self.background = background
        self.psf = psf
        self.edisp = edisp

    @lazyproperty
    def geom(self):
        return self.exposure.geom

    @lazyproperty
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def energy_center(self):
        """Energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.axes[0]
        energy = energy_axis.center * energy_axis.unit
        return energy

    @lazyproperty
    def energy_edges(self):
        """Energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.axes[0]
        energy = energy_axis.edges * energy_axis.unit
        return energy

    @lazyproperty
    def energy_bin_width(self):
        """Energy axis bin widths (`astropy.units.Quantity`)"""
        return np.diff(self.energy_edges)

    @lazyproperty
    def lon_lat(self):
        """Spatial coordinate pixel centers.

        Returns ``lon, lat`` tuple of `~astropy.units.Quantity`.
        """
        lon, lat = self.geom_image.get_coord()
        return lon * u.deg, lat * u.deg

    @lazyproperty
    def lon(self):
        return self.lon_lat[0]

    @lazyproperty
    def lat(self):
        return self.lon_lat[1]

    @lazyproperty
    def solid_angle(self):
        """Solid angle per pixel"""
        return self.geom.solid_angle()

    @lazyproperty
    def bin_volume(self):
        """Map pixel bin volume (solid angle times energy bin width)."""
        omega = self.solid_angle
        de = self.energy_bin_width
        de = de[:, np.newaxis, np.newaxis]
        return omega * de

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.map.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``
        """
        coord = (self.lon, self.lat, self.energy_center)
        dnde = self.sky_model.evaluate(*coord)
        return dnde

    def compute_flux(self):
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        dnde = self.compute_dnde()
        volume = self.bin_volume
        flux = dnde * volume
        return flux.to('cm-2 s-1')

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred_ = (flux * self.exposure.quantity).to('')
        npred = Map.from_geom(self.geom, unit='')
        npred.data = npred_.value
        return npred

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        return self.psf.apply(npred)

    def apply_edisp(self, npred):
        """Convolve npred cube with edisp"""
        a = np.rollaxis(npred, 0, 3)
        npred1 = np.dot(a, self.edisp.pdf_matrix)
        return np.rollaxis(npred1, 2, 0)

    def compute_npred(self):
        """Evaluate model predicted counts.
        """
        flux = self.compute_flux()
        npred = self.apply_exposure(flux)
        if self.psf is not None:
            npred = self.apply_psf(npred)
        # TODO: discuss and decide whether we need to make map objects in `apply_aeff` and `apply_psf`.
        if self.edisp is not None:
            npred.data = self.apply_edisp(npred.data)
        if self.background:
            npred.data += self.background.data
        return npred.data


class SkyDiffuseCube(object):
    """Cube sky map template model (3D).

    This is for a 3D map with an energy axis.
    The map unit is assumed to be ``cm-2 s-1 MeV-1 sr-1``.
    Use `~gammapy.image.models.SkyDiffuseMap` for 2D maps.

    Parameters
    ----------
    map : `~gammapy.map.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    def __init__(self, map, norm=1, meta=None):
        if len(map.geom.axes) != 1:
            raise ValueError('Need a map with an energy axis')

        axis = map.geom.axes[0]
        if axis.name != 'energy':
            raise ValueError('Need a map with axis of name "energy"')

        if axis.node_type != 'center':
            raise ValueError('Need a map with energy axis node_type="center"')

        self.map = map
        self._interp_opts = {'fill_value': 0, 'interp': 'linear'}
        self.parameters = ParameterList([
            Parameter('norm', norm),
        ])
        self.meta = {} if meta is None else meta

    @classmethod
    def read(cls, filename, **kwargs):
        """Read map from FITS file.

        Parameters
        ----------
        filename : str
            FITS image filename.
        """
        m = Map.read(filename, **kwargs)
        if m.unit == '':
            m.unit = 'cm-2 s-1 MeV-1 sr-1'
        return cls(m)

    def evaluate(self, lon, lat, energy):
        """Evaluate model."""
        coord = dict(
            lon=lon.to('deg').value,
            lat=lat.to('deg').value,
            energy=energy.to(self.map.geom.axes[0].unit).value,
        )
        val = self.map.interp_by_coord(coord, **self._interp_opts)
        norm = self.parameters['norm'].value
        return norm * val * u.Unit('cm-2 s-1 MeV-1 sr-1')
