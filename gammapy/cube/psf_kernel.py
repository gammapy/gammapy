# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u
from astropy.units import Quantity
from astropy.convolution import convolve_fft
from ..maps import Map, WcsGeom
from ..image.models.gauss import Gauss2DPDF
from ..irf import TablePSF

__all__ = [
   'table_psf_to_kernel_map',
    'energy_dependent_table_psf_to_kernel_map',
    'PSFKernel',
]


def table_psf_to_kernel_map(table_psf, geom, normalize=True, factor=4):
    """Compute a PSF kernel on a given MapGeom.
    If the MapGeom is not an image, the same kernel will be present on all axes.

    The PSF is estimated by oversampling defined by a given factor.

    Parameters
    ----------
    table_psf : `~gammapy.irf.TablePSF`
        the input table PSF
    geom : `~gammapy.maps.MapGeom`
        the target geometry. The PSF kernel will be centered on the spatial center.
    normalize : bool
        normalize the PSF kernel (per energy)
    factor : int
        the oversample factor to compute the PSF
    """
    # First upsample spatial geom
    upsampled_image_geom = geom.to_image().upsample(factor)

    # get center coordinate
    center_coord = upsampled_image_geom.center_coord * u.deg

    # get coordinates
    map_c = upsampled_image_geom.get_coord()

    # get solid angles
    solid_angles = upsampled_image_geom.solid_angle()

    # compute distances to map center
    rads = angular_separation(center_coord[0], center_coord[1], map_c.lon * u.deg, map_c.lat * u.deg)

    vals = table_psf.evaluate(rad=rads).reshape(solid_angles.shape) * solid_angles

    # Create map
    kernel_map = Map.from_geom(geom=upsampled_image_geom.to_cube(axes=geom.axes),
                               unit='')

    # loop over images and fill map
    for img, idx in kernel_map.iter_by_image():
        img += vals.value

    # downsample the psf kernel map. Take the average
    kernel_map = kernel_map.downsample(factor, preserve_counts=False)

    if normalize:
        # normalize each image in map
        for img, idx in kernel_map.iter_by_image():
            norm = np.sum(img)
            img /= norm

    return kernel_map


def energy_dependent_table_psf_to_kernel_map(table_psf, geom, normalize=True, factor=4):
    """Compute an energy dependent PSF kernel on a given MapGeom.

    The PSF is estimated by oversampling defined by a given factor.

    Parameters
    ----------
    table_psf : `~gammapy.irf.EnergyDependentTablePSF`
        the input table PSF
    geom : `~gammapy.maps.MapGeom`
        the target geometry. The PSF kernel will be centered on the spatial centre.
        the geometry axes should contain an energy MapAxis.
    normalize : bool
        normalize the PSF kernel (per energy)
    factor : int
        the oversample factor to compute the PSF
    """
    # Find energy axis in geom
    energy_idx = -1
    for i, axis in enumerate(geom.axes):
        if axis.type is 'energy':
            energy_idx = i

    if energy_idx == -1:
        raise ValueError("No energy axis in target geometry for PSFKernel")

    energy_unit = u.Unit(geom.axes[energy_idx].unit)

    # TODO: change the logic to support non-regular geometry. This would allow energy dependent sizes for the kernel.
    if geom.is_regular is False:
        raise ValueError("Non regular geometries non supported yet.")

    # First upsample spatial geom
    upsampled_image_geom = geom.to_image().upsample(factor)

    # get center coordinate
    center_coord = upsampled_image_geom.center_coord * u.deg

    # get coordinates
    map_c = upsampled_image_geom.get_coord()

    # get solid angles
    solid_angles = upsampled_image_geom.solid_angle()

    # compute distances to map center
    rads = angular_separation(center_coord[0], center_coord[1], map_c.lon * u.deg, map_c.lat * u.deg)

    # Create map
    kernel_map = Map.from_geom(geom=upsampled_image_geom.to_cube(axes=geom.axes),
                               unit='')

    # loop over images
    for img, idx in kernel_map.iter_by_image():
        energy = geom.axes[energy_idx].center[idx[energy_idx]] * energy_unit
        vals = table_psf.evaluate(energy=energy, rad=rads).reshape(img.shape) * solid_angles
        img += vals.value

    # downsample the psf kernel map. Take the average
    kernel_map = kernel_map.downsample(factor, preserve_counts=False)

    if normalize:
        # normalize each image in map
        for img, idx in kernel_map.iter_by_image():
            norm = np.sum(img)
            img /= norm

    return kernel_map


class PSFKernel(object):
    """PSF kernel for `~gammapy.maps.Map`.

    This is a container class to store a PSF kernel
    that can be used to convolve `~gammapy.maps.WcsNDMap` objects.
    It is usually computed from an EnergyDependentTablePSF

    Parameters
    ----------
    psf_kernel_map : `~gammapy.maps.Map`
        PSF kernel stored in a Map
    """

    def __init__(self, psf_kernel_map):
        self._psf_kernel_map = psf_kernel_map

    @classmethod
    def from_map(cls, psf_kernel_map):
        return cls(psf_kernel_map)

    @classmethod
    def read(cls, *args, **kwargs):
        """Read kernel Map from file."""
        psf_kernel_map = WcsNDMap.read(*args, **kwargs)
        return cls.from_map(psf_kernel_map)

    @classmethod
    def from_energy_dependent_table_psf(cls, table_psf, geom, max_radius=None, normalize=True, factor=4):
        """Create a PSF kernel from an EnergyDependentTablePSF on a given MapGeom.

        The PSF is estimated by oversampling defined by a given factor.

        Parameters
        ----------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            the input table PSF
        geom : `~gammapy.maps.WcsGeom`
            the target geometry. The PSF kernel will be centered on the central pixel.
            the geometry axes should contain an energy MapAxis.
        max_radius : `~astropy.coordinates.Angle` or float
            the maximum radius of the PSF kernel. If float assume unit is degree.
        normalize : bool
            normalize the PSF kernel (per energy)
        factor : int
            the oversample factor to compute the PSF
        """
        if max_radius is not None:
            # Create a new geom accordingly
            center = geom.center_coord[:2]
            binsz = Quantity(np.abs(geom.wcs.wcs.cdelt[0]), 'deg')
            max_radius = Quantity(max_radius, 'deg')
            npix = 2 * int(max_radius / binsz) + 1
            geom = WcsGeom.create(skydir=center, binsz=binsz, npix=npix, proj=geom.projection,
                                  coordsys=geom.coordsys, axes=geom.axes)
        return cls(energy_dependent_table_psf_to_kernel_map(table_psf, geom, normalize, factor))

    @classmethod
    def from_table_psf(cls, table_psf, geom, max_radius=None, normalize=True, factor=4):
        """Create a PSF kernel from an TablePSF on a given MapGeom.
        If the MapGeom is not an image, the same kernel will be present on all axes.

        The PSF is estimated by oversampling defined by a given factor.

        Parameters
        ----------
        table_psf : `~gammapy.irf.TablePSF`
            the input table PSF
        geom : `~gammapy.maps.WcsGeom`
            the target geometry. The PSF kernel will be centered on the central pixel.
            the geometry axes should contain an energy MapAxis.
        max_radius : `~astropy.coordinates.Angle` or float
            the maximum radius of the PSF kernel. If float, we assume unit is degree.
        normalize : bool
            normalize the PSF kernel (per energy)
        factor : int
            the oversample factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the kernel Map with reduced geometry according to the max_radius
        """
        if max_radius is not None:
            # Create a new geom accordingly
            center = geom.center_coord[:2]
            binsz = Quantity(np.abs(geom.wcs.wcs.cdelt[0]), 'deg')
            max_radius = Quantity(max_radius, 'deg')
            npix = 2 * int(max_radius / binsz) + 1
            geom = WcsGeom.create(skydir=center, binsz=binsz, npix=npix, proj=geom.projection,
                                  coordsys=geom.coordsys, axes=geom.axes)
        return cls(table_psf_to_kernel_map(table_psf, geom, normalize, factor))

    @classmethod
    def from_gauss(cls, geom, sigma, max_radius=None, containment_fraction=0.99,
                   normalize=True, factor=4):
        """Create Gaussian PSF.

        This is used for testing and examples.
        The map geometry parameters (pixel size, energy bins) are taken from `geom`.
        The Gaussian width ``sigma`` is a scalar,

        TODO : support array input if it should vary along the energy axis.

        Parameters
        ----------
        geom : `~gammapy.map.WcsGeom`
            Map geometry
        sigma : `~astropy.coordinates.Angle` or float
            Gaussian width. Assume degrees if float input
        max_radius : `~astropy.coordinates.Angle` or float
            Desired kernel map size. Implicit unit is degree.
        normalize : bool
            normalize the PSF kernel (per energy)
        factor : int
            the oversample factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the kernel Map with reduced geometry according to the max_radius
        """
        sigma = Angle(sigma, 'deg')

        if max_radius is None:
            max_radius = Gauss2DPDF(sigma.to('deg').value).containment_radius(
                containment_fraction=containment_fraction)

        max_radius = Angle(max_radius, 'deg')

        # Create a new geom according to given input
        center = geom.center_coord[:2]
        binsz = Quantity(np.abs(geom.wcs.wcs.cdelt[0]), 'deg')
        npix = 2 * int(max_radius / binsz) + 1
        geom = WcsGeom.create(skydir=center, binsz=binsz, npix=npix, proj=geom.projection,
                              coordsys=geom.coordsys, axes=geom.axes)

        rad = Angle(np.linspace(0., max_radius, 200), 'deg')

        table_psf = TablePSF.from_shape(shape='gauss', width=sigma, rad=rad)

        return cls(table_psf_to_kernel_map(table_psf, geom))

    def to_map(self):
        return self._psf_kernel_map

    def write(self, *args, **kwargs):
        psf_kernel_map = self.to_map()
        psf_kernel_map.write(*args, **kwargs)

    def apply(self, map):
        """Apply the kernel to an input Map.

        Parameters
        ----------
        map : `~gammapy.maps.WcsNDMap`
            the model map to be convolved with the PSFKernel.
            It should have the same MapGeom than the current PSFKernel.

        Returns
        -------
        convolved_map : `~gammapy.maps.Map`
            the convolved map
        """
        # TODO : check that the MapGeom are consistent
        convolved_map = Map.from_geom(geom=map.geom, unit=map.unit, meta=map.meta)
        for img, idx in map.iter_by_image():
            convolved_map.data[idx] = convolve_fft(img, self.to_map().data[idx])
        return convolved_map