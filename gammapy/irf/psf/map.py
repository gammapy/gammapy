# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gammapy.maps import Map, MapAxes, MapAxis, MapCoord, WcsGeom
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.random import InverseCDFSampler, get_random_state
from ..core import IRFMap
from .core import PSF
from .kernel import PSFKernel

__all__ = ["PSFMap", "RecoPSFMap"]


class IRFLikePSF(PSF):
    required_axes = ["energy_true", "rad", "lat_idx", "lon_idx"]
    tag = "irf_like_psf"


class PSFMap(IRFMap):
    """Class containing the Map of PSFs and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        the input PSF Map. Should be a Map with 2 non spatial axes.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    ::

        from astropy.coordinates import SkyCoord
        from gammapy.maps import WcsGeom, MapAxis
        from gammapy.data import Observation
        from gammapy.irf import load_cta_irfs
        from gammapy.makers import MapDatasetMaker

        # Define observation
        pointing = SkyCoord("0d", "0d")
        filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        irfs = load_cta_irfs(filename)
        obs = Observation.create(pointing=pointing, irfs=irfs, livetime="1h")

        # Create WcsGeom
        # Define energy axis. Note that the name is fixed.
        energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3, name="energy_true")

        # Define rad axis. Again note the axis name
        rad_axis = MapAxis.from_bounds(0, 0.5, nbin=100, name="rad", unit="deg")

        geom = WcsGeom.create(
            binsz=0.25, width="5 deg", skydir=pointing, axes=[rad_axis, energy_axis]
        )

        maker = MapDatasetMaker()

        psf = maker.make_psf(geom=geom, observation=obs)

        # Get a PSF kernel at the center of the image
        geom=exposure_geom.upsample(factor=10).drop("rad")
        psf_kernel = psf_map.get_psf_kernel(geom=geom)
    """

    tag = "psf_map"
    required_axes = ["rad", "energy_true"]

    def __init__(self, psf_map, exposure_map=None):
        super().__init__(irf_map=psf_map, exposure_map=exposure_map)

    @property
    def energy_name(self):
        return self.required_axes[-1]

    @property
    def psf_map(self):
        return self._irf_map

    @psf_map.setter
    def psf_map(self, value):
        self._irf_map = value

    def normalize(self):
        """Normalize PSF map"""
        self.psf_map.normalize(axis_name="rad")

    @classmethod
    def from_geom(cls, geom):
        """Create psf map from geom.

        Parameters
        ----------
        geom : `Geom`
            PSF map geometry.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        geom_exposure = geom.squash(axis_name="rad")
        exposure_psf = Map.from_geom(geom_exposure, unit="m2 s")
        psf_map = Map.from_geom(geom, unit="sr-1")
        return cls(psf_map, exposure_psf)

    # TODO: this is a workaround for now, probably add Map.integral() or similar
    @property
    def _psf_irf(self):
        geom = self.psf_map.geom
        npix_x, npix_y = geom.npix
        axis_lon = MapAxis.from_edges(np.arange(npix_x + 1) - 0.5, name="lon_idx")
        axis_lat = MapAxis.from_edges(np.arange(npix_y + 1) - 0.5, name="lat_idx")
        axes = MapAxes(
            [geom.axes[self.energy_name], geom.axes["rad"], axis_lat, axis_lon]
        )
        psf = IRFLikePSF
        psf.required_axes = axes.names
        return psf(
            axes=axes,
            data=self.psf_map.data,
            unit=self.psf_map.unit,
        )

    def _get_irf_coords(self, **kwargs):
        coords = MapCoord.create(kwargs)

        geom = self.psf_map.geom.to_image()
        lon_pix, lat_pix = geom.coord_to_pix(coords.skycoord)

        coords_irf = {
            "lon_idx": lon_pix,
            "lat_idx": lat_pix,
            self.energy_name: coords[self.energy_name],
        }

        try:
            coords_irf["rad"] = coords["rad"]
        except KeyError:
            pass

        return coords_irf

    def containment(self, rad, energy_true, position=None):
        """Containment at given coords

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        energy_true : `~astropy.units.Quantity`
            Energy true value
        position : `~astropy.coordinates.SkyCoord`
            Sky position. By default the center of the map is chosen

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values
        """
        if position is None:
            position = self.psf_map.geom.center_skydir

        coords = {"skycoord": position, "rad": rad, self.energy_name: energy_true}

        return self.psf_map.integral(axis_name="rad", coords=coords).to("")

    def containment_radius(self, fraction, energy_true, position=None):
        """Containment at given coords

        Parameters
        ----------
        fraction : float
            Containment fraction
        energy_true : `~astropy.units.Quantity`
            Energy true value
        position : `~astropy.coordinates.SkyCoord`
            Sky position. By default the center of the map is chosen

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values
        """
        if position is None:
            position = self.psf_map.geom.center_skydir

        kwargs = {self.energy_name: energy_true, "skycoord": position}
        coords = self._get_irf_coords(**kwargs)

        return self._psf_irf.containment_radius(fraction, **coords)

    def containment_radius_map(self, energy_true, fraction=0.68):
        """Containment radius map.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            Energy true at which to compute the containment radius
        fraction : float
            Containment fraction (range: 0 to 1)

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map
        """
        geom = self.psf_map.geom.to_image()

        data = self.containment_radius(
            fraction,
            energy_true,
            geom.get_coord().skycoord,
        )
        return Map.from_geom(geom=geom, data=data.value, unit=data.unit)

    def get_psf_kernel(
        self, geom, position=None, max_radius=None, containment=0.999, factor=4
    ):
        """Returns a PSF kernel at the given position.

        The PSF is returned in the form a WcsNDMap defined by the input Geom.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Target geometry to use
        position : `~astropy.coordinates.SkyCoord`
            Target position. Should be a single coordinate. By default the
            center position is used.
        max_radius : `~astropy.coordinates.Angle`
            maximum angular size of the kernel map
        containment : float
            Containment fraction to use as size of the kernel. The max. radius
            across all energies is used. The radius can be overwritten using
            the `max_radius` argument.
        factor : int
            oversampling factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.irf.PSFKernel`
            the resulting kernel
        """
        # TODO: try to simplify...is the oversampling needed?
        if position is None:
            position = self.psf_map.geom.center_skydir

        position = self._get_nearest_valid_position(position)

        if max_radius is None:
            energy_axis = self.psf_map.geom.axes[self.energy_name]
            kwargs = {
                "fraction": containment,
                "position": position,
                self.energy_name: energy_axis.center,
            }
            radii = self.containment_radius(**kwargs)
            max_radius = np.max(radii)

        geom = geom.to_odd_npix(max_radius=max_radius)
        geom_upsampled = geom.upsample(factor=factor)
        coords = geom_upsampled.get_coord(sparse=True)
        rad = coords.skycoord.separation(geom.center_skydir)

        coords = {
            self.energy_name: coords[self.energy_name],
            "rad": rad,
            "skycoord": position,
        }

        data = self.psf_map.interp_by_coord(
            coords=coords,
            method="linear",
        )

        kernel_map = Map.from_geom(geom=geom_upsampled, data=np.clip(data, 0, np.inf))
        kernel_map = kernel_map.downsample(factor, preserve_counts=True)
        return PSFKernel(kernel_map, normalize=True)

    def sample_coord(self, map_coord, random_state=0):
        """Apply PSF corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        corr_coord : `~gammapy.maps.MapCoord` object.
            Sequence of PSF-corrected coordinates of the input map_coord map.
        """

        random_state = get_random_state(random_state)
        rad_axis = self.psf_map.geom.axes["rad"]

        coord = {
            "skycoord": map_coord.skycoord.reshape(-1, 1),
            self.energy_name: map_coord[self.energy_name].reshape(-1, 1),
            "rad": rad_axis.center,
        }

        pdf = (
            self.psf_map.interp_by_coord(coord)
            * rad_axis.center.value
            * rad_axis.bin_width.value
        )

        sample_pdf = InverseCDFSampler(pdf, axis=1, random_state=random_state)
        pix_coord = sample_pdf.sample_axis()
        separation = rad_axis.pix_to_coord(pix_coord)

        position_angle = random_state.uniform(360, size=len(map_coord.lon)) * u.deg

        event_positions = map_coord.skycoord.directional_offset_by(
            position_angle=position_angle, separation=separation
        )
        return MapCoord.create(
            {"skycoord": event_positions, self.energy_name: map_coord[self.energy_name]}
        )

    @classmethod
    def from_gauss(cls, energy_axis_true, rad_axis=None, sigma=0.1 * u.deg, geom=None):
        """Create all -sky PSF map from Gaussian width.

        This is used for testing and examples.

        The width can be the same for all energies
        or be an array with one value per energy node.
        It does not depend on position.

        Parameters
        ----------
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis.
        rad_axis : `~gammapy.maps.MapAxis`
            Offset angle wrt source position axis.
        sigma : `~astropy.coordinates.Angle`
            Gaussian width.
        geom : `Geom`
            Image geometry. By default an allsky geometry is created.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        if rad_axis is None:
            rad_axis = RAD_AXIS_DEFAULT.copy()

        if geom is None:
            geom = WcsGeom.create(
                npix=(2, 1),
                proj="CAR",
                binsz=180,
            )

        geom = geom.to_cube([rad_axis, energy_axis_true])

        coords = geom.get_coord(sparse=True)

        sigma = u.Quantity(sigma).reshape((-1, 1, 1, 1))
        gauss = Gauss2DPDF(sigma=sigma)

        data = gauss(coords["rad"]) * np.ones(geom.data_shape)

        psf_map = Map.from_geom(geom=geom, data=data.to_value("sr-1"), unit="sr-1")

        exposure_map = Map.from_geom(
            geom=geom.squash(axis_name="rad"), unit="m2 s", data=1.0
        )
        return cls(psf_map=psf_map, exposure_map=exposure_map)

    def to_image(self, spectrum=None, keepdims=True):
        """Reduce to a 2-D map after weighing
        with the associated exposure and a spectrum

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If True, the energy axis is kept with one bin.
            If False, the axis is removed


        Returns
        -------
        psf_out : `PSFMap`
            `PSFMap` with the energy axis summed over
        """
        from gammapy.makers.utils import _map_spectrum_weight

        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0)

        exp_weighed = _map_spectrum_weight(self.exposure_map, spectrum)
        exposure = exp_weighed.sum_over_axes(
            axes_names=[self.energy_name], keepdims=keepdims
        )

        psf_data = exp_weighed.data * self.psf_map.data / exposure.data
        psf_map = Map.from_geom(geom=self.psf_map.geom, data=psf_data, unit="sr-1")

        psf = psf_map.sum_over_axes(axes_names=[self.energy_name], keepdims=keepdims)
        return self.__class__(psf_map=psf, exposure_map=exposure)

    def plot_containment_radius_vs_energy(
        self, ax=None, fraction=(0.68, 0.95), **kwargs
    ):
        """Plot containment fraction as a function of energy.

        The method plots the containment radius at the center of the map.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        fraction : list of float or `~numpy.ndarray`
            Containment fraction between 0 and 1.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.

        """
        ax = plt.gca() if ax is None else ax

        position = self.psf_map.geom.center_skydir
        energy_axis = self.psf_map.geom.axes[self.energy_name]
        energy_true = energy_axis.center

        for frac in fraction:
            radius = self.containment_radius(frac, energy_true, position)
            label = f"Containment: {100 * frac:.1f}%"
            with quantity_support():
                ax.plot(energy_true, radius, label=label, **kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_ylabel(f"Containment radius ({ax.yaxis.units})")
        return ax

    def plot_psf_vs_rad(self, ax=None, energy_true=[0.1, 1, 10] * u.TeV, **kwargs):
        """Plot PSF vs radius.

        The method plots the profile at the center of the map.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.

        """
        ax = plt.gca() if ax is None else ax

        rad = self.psf_map.geom.axes["rad"].center

        for value in energy_true:
            psf_value = self.psf_map.interp_by_coord(
                {
                    "skycoord": self.psf_map.geom.center_skydir,
                    self.energy_name: value,
                    "rad": rad,
                }
            )
            label = f"{value:.0f}"
            with quantity_support():
                ax.plot(rad, psf_value, label=label, **kwargs)

        ax.set_yscale("log")
        ax.set_xlabel(f"Rad ({ax.xaxis.units})")
        ax.set_ylabel(f"PSF ({ax.yaxis.units})")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.legend()
        return ax

    def __str__(self):
        return str(self.psf_map)

    def peek(self, figsize=(12, 10)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of figure.
        """
        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            subplot_kw={"projection": self.psf_map.geom.wcs},
            figsize=figsize,
            gridspec_kw={"hspace": 0.3, "wspace": 0.3},
        )

        axes = axes.flat
        axes[0].remove()
        ax0 = fig.add_subplot(2, 2, 1)
        ax0.set_title("Containment radius at center of map")
        self.plot_containment_radius_vs_energy(ax=ax0)

        axes[1].remove()
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.set_ylim(1e-4, 1e4)
        ax1.set_title("PSF at center of map")
        self.plot_psf_vs_rad(ax=ax1)

        axes[2].set_title("Exposure")
        self.exposure_map.reduce_over_axes().plot(ax=axes[2], add_cbar=True)

        axes[3].set_title("Containment radius at 1 TeV")
        kwargs = {self.energy_name: 1 * u.TeV}
        self.containment_radius_map(**kwargs).plot(ax=axes[3], add_cbar=True)


class RecoPSFMap(PSFMap):
    """Class containing the Map of PSFs in reconstructed energy and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        the input PSF Map. Should be a Map with 2 non spatial axes.
        rad and energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`
        Associated exposure map. Needs to have a consistent map geometry.
    """

    tag = "psf_map_reco"
    required_axes = ["rad", "energy"]

    @property
    def energy_name(self):
        return self.required_axes[-1]

    @classmethod
    def from_gauss(cls, energy_axis, rad_axis=None, sigma=0.1 * u.deg, geom=None):
        """Create all -sky PSF map from Gaussian width.

        This is used for testing and examples.

        The width can be the same for all energies
        or be an array with one value per energy node.
        It does not depend on position.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            Energy axis.
        rad_axis : `~gammapy.maps.MapAxis`
            Offset angle wrt source position axis.
        sigma : `~astropy.coordinates.Angle`
            Gaussian width.
        geom : `Geom`
            Image geometry. By default an allsky geometry is created.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        return super().from_gauss(energy_axis, rad_axis, sigma, geom)

    def containment(self, rad, energy, position=None):
        """Containment at given coords

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        energy : `~astropy.units.Quantity`
            Energy value
        position : `~astropy.coordinates.SkyCoord`
            Sky position. By default the center of the map is chosen

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values
        """
        return super().containment(rad, energy, position)

    def containment_radius(self, fraction, energy, position=None):
        """Containment at given coords

        Parameters
        ----------
        fraction : float
            Containment fraction
        energy : `~astropy.units.Quantity`
            Energy value
        position : `~astropy.coordinates.SkyCoord`
            Sky position. By default the center of the map is chosen

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values
        """
        return super().containment_radius(fraction, energy, position)

    def containment_radius_map(self, energy, fraction=0.68):
        """Containment radius map.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to compute the containment radius
        fraction : float
            Containment fraction (range: 0 to 1)

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map
        """
        return super().containment_radius_map(energy, fraction=0.68)

    def plot_psf_vs_rad(self, ax=None, energy=[0.1, 1, 10] * u.TeV, **kwargs):
        """Plot PSF vs radius.

        The method plots the profile at the center of the map.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.

        """
        return super().plot_psf_vs_rad(ax, energy_true=energy, **kwargs)

    def stack(self, other, weights=None, nan_to_num=True):
        """Stack IRF map with another one in place."""
        raise NotImplementedError(
            "Stacking is not supported for PSF in reconstructed energy."
        )
