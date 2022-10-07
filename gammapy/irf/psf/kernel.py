# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.maps import Map
from gammapy.modeling.models import PowerLawSpectralModel

__all__ = ["PSFKernel"]


class PSFKernel:
    """PSF kernel for `~gammapy.maps.Map`.

    This is a container class to store a PSF kernel
    that can be used to convolve `~gammapy.maps.WcsNDMap` objects.
    It is usually computed from an `~gammapy.irf.PSFMap`.

    Parameters
    ----------
    psf_kernel_map : `~gammapy.maps.Map`
        PSF kernel stored in a Map

    Examples
    --------
    ::

        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import PSFMap
        from astropy import units as u

        # Define energy axis
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.1 TeV", "10 TeV", nbin=4, name="energy_true
        )

        # Create WcsGeom and map
        geom = WcsGeom.create(binsz=0.02 * u.deg, width=2.0 * u.deg, axes=[energy_axis_true])
        some_map = Map.from_geom(geom)

        # Fill map at three positions
        some_map.fill_by_coord([[0, 0, 0], [0, 0, 0], [0.3, 1, 3]])

        psf = PSFMap.from_gauss(
            energy_axis_true=energy_axis_true, sigma=[0.3, 0.2, 0.1] * u.deg
        )

        kernel = psf.get_psf_kernel(geom=geom, max_radius=1*u.deg)

        # Do the convolution
        some_map_convolved = some_map.convolve(kernel)

        some_map_convolved.plot_grid();
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
    def from_spatial_model(cls, model, geom, max_radius=None, factor=4):
        """Create PSF kernel from spatial model

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Map geometry
        model : `~gammapy.modeling.models.SpatiaModel`
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
        if max_radius is None:
            max_radius = model.evaluation_radius

        geom = geom.to_odd_npix(max_radius=max_radius)
        model.position = geom.center_skydir

        geom = geom.upsample(factor=factor)
        map = model.integrate_geom(geom, oversampling_factor=1)
        return cls(psf_kernel_map=map.downsample(factor=factor))

    @classmethod
    def from_gauss(cls, geom, sigma, max_radius=None, factor=4):
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
        from gammapy.modeling.models import GaussianSpatialModel

        gauss = GaussianSpatialModel(sigma=sigma)

        return cls.from_spatial_model(
            model=gauss, geom=geom, max_radius=max_radius, factor=factor
        )

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
            If true, the resulting PSFKernel will keep an energy axis with one bin.
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
        for name in map.geom.axes.names:
            if "energy" in name:
                energy_name = name
        energy_edges = map.geom.axes[energy_name].edges
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

    def plot_kernel(self, ax=None, energy=None, add_cbar=False, **kwargs):
        """Plot PSF kernel.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`, optional
            If None, the PSF kernel is summed over the energy axis. Otherwise, the kernel
            corresponding to the energy bin including the given energy is shown.
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        kwargs: dict
            Keyword arguments passed to `~matplotlib.axes.Axes.imshow`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if energy:
            kernel_map = self.psf_kernel_map
            energy_center = kernel_map.geom.axes["energy_true"].center.to(energy.unit)
            idx = np.argmin(np.abs(energy_center.value - energy.value))
            kernel_map.get_image_by_idx([idx]).plot(ax=ax, add_cbar=add_cbar, **kwargs)
        else:
            self.to_image().psf_kernel_map.plot(ax=ax, add_cbar=add_cbar, **kwargs)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        axes[0].set_title("Energy-integrated PSF kernel")
        self.plot_kernel(ax=axes[0], add_cbar=True)

        axes[1].set_title("PSF kernel at 1 TeV")
        self.plot_kernel(ax=axes[1], add_cbar=True, energy=1 * u.TeV)

        plt.tight_layout()
