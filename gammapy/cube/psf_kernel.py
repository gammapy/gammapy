# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u
from ..utils.gauss import Gauss2DPDF
from ..maps import Map, WcsGeom
from ..irf import TablePSF

__all__ = ["PSFKernel"]


def _make_kernel_geom(geom, max_radius):
    # Create a new geom object with an odd number of pixel and a maximum size
    # This is useful for PSF kernel creation.
    center = geom.center_coord[:2]
    binsz = Angle(np.abs(geom.wcs.wcs.cdelt[0]), "deg")
    max_radius = Angle(max_radius)
    npix = 2 * int(max_radius.deg / binsz.deg) + 1
    return WcsGeom.create(
        skydir=center,
        binsz=binsz,
        npix=npix,
        proj=geom.projection,
        coordsys=geom.coordsys,
        axes=geom.axes,
    )


def _compute_kernel_separations(geom, factor):
    # utility function used for preparing distance to the center of the upsampled geom
    # TODO : take into account non regular geometry for energy dependent PSF kernel size
    if geom.is_regular is False:
        raise ValueError("Non regular geometries are not supported yet.")

    upsampled_image_geom = geom.to_image().upsample(factor)
    # get center coordinate
    center_coord = upsampled_image_geom.center_coord * u.deg
    # get coordinates
    map_c = upsampled_image_geom.get_coord()
    # compute distances to map center
    separations = angular_separation(
        center_coord[0], center_coord[1], map_c.lon * u.deg, map_c.lat * u.deg
    )

    # Create map
    kernel_map = Map.from_geom(geom=upsampled_image_geom.to_cube(axes=geom.axes))
    return kernel_map, separations


def table_psf_to_kernel_map(table_psf, geom, factor=4):
    """Compute a PSF kernel on a given MapGeom.

    If the MapGeom is not an image, the same kernel will be present on all axes.

    The PSF is estimated by oversampling defined by a given factor.
    The PSF kernel is normalized

    Parameters
    ----------
    table_psf : `~gammapy.irf.TablePSF`
        the input table PSF
    geom : `~gammapy.maps.MapGeom`
        the target geometry. The PSF kernel will be centered on the spatial center.
    factor : int
        the oversample factor to compute the PSF
    """
    # prepare map and compute distances to map center
    kernel_map, rads = _compute_kernel_separations(geom, factor)

    vals = table_psf.evaluate(rad=rads).value
    norm = vals.sum()

    for img, idx in kernel_map.iter_by_image():
        img += vals.reshape(img.shape) / norm

    return kernel_map.downsample(factor, preserve_counts=True)


def energy_dependent_table_psf_to_kernel_map(table_psf, geom, factor=4):
    """Compute an energy dependent PSF kernel on a given MapGeom.

    The PSF is estimated by oversampling defined by a given factor.

    Parameters
    ----------
    table_psf : `~gammapy.irf.EnergyDependentTablePSF`
        the input table PSF
    geom : `~gammapy.maps.MapGeom`
        the target geometry.
        The PSF kernel will be centered on the spatial centre.
        the geometry axes should contain an "energy" axis.
        The kernel will be duplicated along other axes.
    factor : int
        the oversample factor to compute the PSF
    """
    energy_axis = geom.get_axis_by_name("energy")
    energy_idx = geom.axes.index(energy_axis)
    energy_unit = u.Unit(energy_axis.unit)

    # prepare map and compute distances to map center
    kernel_map, rads = _compute_kernel_separations(geom, factor)

    # loop over images
    for img, idx in kernel_map.iter_by_image():
        # TODO: this is super complex. Find or invent a better way!
        energy = energy_axis.center[idx[energy_idx]] * energy_unit
        vals = table_psf.evaluate(energy=energy, rad=rads).reshape(img.shape)
        img += vals.value / vals.sum().value

    return kernel_map.downsample(factor, preserve_counts=True)


class PSFKernel(object):
    """PSF kernel for `~gammapy.maps.Map`.

    This is a container class to store a PSF kernel
    that can be used to convolve `~gammapy.maps.WcsNDMap` objects.
    It is usually computed from an `~gammapy.irf.EnergyDependentTablePSF`.

    Parameters
    ----------
    psf_kernel_map : `~gammapy.maps.Map`
        PSF kernel stored in a Map

    Examples
    --------

    .. code:: python

        import numpy as np
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import EnergyDependentMultiGaussPSF
        from gammapy.cube import PSFKernel
        from astropy import units as u

        # Define energy axis
        energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit='TeV', name='energy')

        # Create WcsGeom and map
        geom = WcsGeom.create(binsz=0.02*u.deg, width=2.0*u.deg, axes=[energy_axis])
        some_map = Map.from_geom(geom)
        # Fill map at two positions
        some_map.fill_by_coord([[0.2,0.4],[-0.1,0.6],[0.5,3.6]])

        # Extract EnergyDependentTablePSF from CTA 1DC IRF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        table_psf = psf.to_energy_dependent_table_psf(theta=0.5*u.deg)

        psf_kernel = PSFKernel.from_table_psf(table_psf,geom, max_radius=1*u.deg)

        # Do the convolution
        some_map_convolved = some_map.convolve(psf_kernel)

        some_map_convolved.get_image_by_coord(dict(energy=0.6*u.TeV)).plot()
    """

    def __init__(self, psf_kernel_map):
        self._psf_kernel_map = psf_kernel_map

    @property
    def data(self):
        """Access the PSFKernel numpy array"""
        return self._psf_kernel_map.data

    @property
    def psf_kernel_map(self):
        """The map object holding the kernel (`~gammapy.maps.Map`)"""
        return self._psf_kernel_map

    @classmethod
    def read(cls, *args, **kwargs):
        """Read kernel Map from file."""
        psf_kernel_map = Map.read(*args, **kwargs)
        return cls(psf_kernel_map)

    @classmethod
    def from_table_psf(cls, table_psf, geom, max_radius=None, factor=4):
        """Create a PSF kernel from a TablePSF or an EnergyDependentTablePSF on a given MapGeom.

        If the MapGeom is not an image, the same kernel will be present on all axes.

        The PSF is estimated by oversampling defined by a given factor.

        Parameters
        ----------
        table_psf : `~gammapy.irf.TablePSF` or `~gammapy.irf.EnergyDependentTablePSF`
            the input table PSF
        geom : `~gammapy.maps.WcsGeom`
            the target geometry. The PSF kernel will be centered on the central pixel.
            The geometry axes should contain an axis with name "energy"
        max_radius : `~astropy.coordinates.Angle`
            the maximum radius of the PSF kernel.
        factor : int
            the oversample factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the kernel Map with reduced geometry according to the max_radius
        """
        # TODO : use PSF containment radius if max_radius is None
        if max_radius is not None:
            geom = _make_kernel_geom(geom, max_radius)

        if isinstance(table_psf, TablePSF):
            return cls(table_psf_to_kernel_map(table_psf, geom, factor))
        else:
            return cls(
                energy_dependent_table_psf_to_kernel_map(table_psf, geom, factor)
            )

    @classmethod
    def from_gauss(
        cls, geom, sigma, max_radius=None, containment_fraction=0.99, factor=4
    ):
        """Create Gaussian PSF.

        This is used for testing and examples.
        The map geometry parameters (pixel size, energy bins) are taken from ``geom``.
        The Gaussian width ``sigma`` is a scalar,

        TODO : support array input if it should vary along the energy axis.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Map geometry
        sigma : `~astropy.coordinates.Angle`
            Gaussian width.
        max_radius : `~astropy.coordinates.Angle`
            Desired kernel map size.
        factor : int
            Oversample factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the kernel Map with reduced geometry according to the max_radius
        """
        sigma = Angle(sigma)

        if max_radius is None:
            max_radius = (
                Gauss2DPDF(sigma.deg).containment_radius(
                    containment_fraction=containment_fraction
                )
                * u.deg
            )

        max_radius = Angle(max_radius)

        # Create a new geom according to given input
        geom = _make_kernel_geom(geom, max_radius)

        rad = Angle(np.linspace(0., max_radius.deg, 200), "deg")

        table_psf = TablePSF.from_shape(shape="gauss", width=sigma, rad=rad)

        return cls(table_psf_to_kernel_map(table_psf, geom, factor))

    def write(self, *args, **kwargs):
        """Write the Map object which contains the PSF kernel to file."""
        self.psf_kernel_map.write(*args, **kwargs)
