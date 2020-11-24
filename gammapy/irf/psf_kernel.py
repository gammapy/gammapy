# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.maps import Map, WcsGeom
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.gauss import Gauss2DPDF
from .psf_table import EnergyDependentTablePSF, TablePSF

__all__ = ["PSFKernel"]


def _make_kernel_geom(geom, max_radius):
    # Create a new geom object with an odd number of pixel and a maximum size
    # This is useful for PSF kernel creation.
    center = geom.center_skydir
    binsz = Angle(np.abs(geom.wcs.wcs.cdelt[0]), "deg")
    max_radius = Angle(max_radius)
    npix = 2 * int(max_radius.deg / binsz.deg) + 1
    return WcsGeom.create(
        skydir=center,
        binsz=binsz,
        npix=npix,
        proj=geom.projection,
        frame=geom.frame,
        axes=geom.axes,
    )


class PSFKernel:
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
    ::

        import numpy as np
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import EnergyDependentMultiGaussPSF, PSFKernel
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

    def __init__(self, psf_kernel_map, normalize=True):
        self._psf_kernel_map = psf_kernel_map

        if normalize:
            self.normalize()

    def normalize(self):
        """Force normalisation of the kernel"""
        data = self.psf_kernel_map.data
        if self.psf_kernel_map.geom.is_image:
            axis = (0, 1)
        else:
            axis = (1, 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.nan_to_num(data / data.sum(axis=axis, keepdims=True))
            self.psf_kernel_map.data = data

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
        """Create a PSF kernel from a TablePSF or an EnergyDependentTablePSF on a given Geom.

        If the Geom is not an image, the same kernel will be present on all axes.

        The PSF is estimated by oversampling defined by a given factor.

        Parameters
        ----------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Input table PSF
        geom : `~gammapy.maps.WcsGeom`
            Target geometry. The PSF kernel will be centered on the central pixel.
            The geometry axes should contain an axis with name "energy"
        max_radius : `~astropy.coordinates.Angle`
            Maximum radius of the PSF kernel.
        factor : int
            Oversample factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.irf.PSFKernel`
            the kernel Map with reduced geometry according to the max_radius
        """
        # TODO : use PSF containment radius if max_radius is None
        if max_radius is not None:
            geom = _make_kernel_geom(geom, max_radius)

        geom_upsampled = geom.upsample(factor=factor)
        rad = geom_upsampled.separation(geom.center_skydir)

        if isinstance(table_psf, EnergyDependentTablePSF):
            energy_axis = geom.axes["energy_true"]
            energy = energy_axis.center[:, np.newaxis, np.newaxis]
            data = table_psf.evaluate(energy=energy, rad=rad).value
        else:
            try:
                nbin = geom.axes[0].nbin
            except IndexError:
                nbin = 1
            data = table_psf.evaluate(rad=rad).value
            data = data * np.ones(nbin).reshape((-1, 1, 1))

        kernel_map = Map.from_geom(geom=geom_upsampled, data=data)
        kernel_map = kernel_map.downsample(factor, preserve_counts=True)
        return cls(kernel_map, normalize=True)

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
        kernel : `~gammapy.irf.PSFKernel`
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

        rad = Angle(np.linspace(0.0, max_radius.deg, 200), "deg")

        table_psf = TablePSF.from_shape(shape="gauss", width=sigma, rad=rad)

        return cls.from_table_psf(table_psf, geom=geom, factor=factor)

    def write(self, *args, **kwargs):
        """Write the Map object which contains the PSF kernel to file."""
        self.psf_kernel_map.write(*args, **kwargs)

    def to_image(self, spectrum=None, exposure=None, keepdims=True):
        """Transform 3D PSFKernel into a 2D PSFKernel.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        exposure : `~astropy.units.Quantity` or `~numpy.ndarray`
            1D array containing exposure in each true energy bin.
            It must have the same size as the PSFKernel energy axis.
            Default is uniform exposure over energy.
        keepdims : bool
            If true, the resulting PSFKernel wil keep an energy axis with one bin.
            Default is True.

        Returns
        -------
        weighted_kernel : `~gammapy.irf.PSFKernel`
            the weighted kernel summed over energy
        """
        map = self.psf_kernel_map

        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0)

        if exposure is None:
            exposure = np.ones(map.geom.axes[0].center.shape)
        exposure = u.Quantity(exposure)
        if exposure.shape != map.geom.axes[0].center.shape:
            raise ValueError("Incorrect exposure_array shape")

        # Compute weights vector
        energy_edges = map.geom.axes["energy_true"].edges
        weights = spectrum.integral(
            energy_min=energy_edges[:-1], energy_max=energy_edges[1:], intervals=True
        )
        weights *= exposure
        weights /= weights.sum()

        spectrum_weighted_kernel = map.copy()
        spectrum_weighted_kernel.quantity *= weights[:, np.newaxis, np.newaxis]

        return self.__class__(spectrum_weighted_kernel.sum_over_axes(keepdims=keepdims))

    def slice_by_idx(self, slices):
        """Slice by idx"""
        kernel = self.psf_kernel_map.slice_by_idx(slices=slices)
        return self.__class__(psf_kernel_map=kernel)
