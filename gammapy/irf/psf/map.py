# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gammapy.maps import HpxGeom, Map, MapAxes, MapAxis, MapCoord, WcsGeom
from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.deprecation import deprecated_renamed_argument
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.random import InverseCDFSampler, get_random_state
from ..core import IRFMap
from .core import PSF
from .kernel import PSFKernel

__all__ = ["PSFMap", "RecoPSFMap"]


PSF_MAX_OVERSAMPLING = 4  # for backward compatibility


def _psf_upsampling_factor(psf, geom, position, energy=None, precision_factor=12):
    """Minimal factor between the bin half-width of the geom and the median R68% containment radius."""
    if energy is None:
        energy = geom.axes[psf.energy_name].center
    psf_r68s = psf.containment_radius(
        0.68, geom.axes[psf.energy_name].center, position=position
    )
    factors = []
    for psf_r68 in psf_r68s:
        base_factor = (2 * psf_r68 / geom.pixel_scales.max()).to_value("")
        factor = np.minimum(
            int(np.ceil(precision_factor / base_factor)), PSF_MAX_OVERSAMPLING
        )
        if isinstance(geom, HpxGeom):
            factor = int(2 ** np.ceil(np.log(factor) / np.log(2)))
        factors.append(factor)
    return factors


class IRFLikePSF(PSF):
    required_axes = ["energy_true", "rad", "lat_idx", "lon_idx"]
    tag = "irf_like_psf"


class PSFMap(IRFMap):
    """Class containing the Map of PSFs and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        The input PSF Map. Should be a Map with 2 non spatial axes.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    .. testcode::

        from astropy.coordinates import SkyCoord
        from gammapy.maps import WcsGeom, MapAxis
        from gammapy.data import Observation, FixedPointingInfo
        from gammapy.irf import load_irf_dict_from_file
        from gammapy.makers import MapDatasetMaker

        # Define observation
        pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
        pointing = FixedPointingInfo(
            fixed_icrs=pointing_position.icrs,
        )
        filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        irfs = load_irf_dict_from_file(filename)
        obs = Observation.create(pointing=pointing, irfs=irfs, livetime="1h")

        # Define energy axis. Note that the name is fixed.
        energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3, name="energy_true")

        # Define rad axis. Again note the axis name
        rad_axis = MapAxis.from_bounds(0, 0.5, nbin=100, name="rad", unit="deg")

        # Create WcsGeom
        geom = WcsGeom.create(
            binsz=0.25, width="5 deg", skydir=pointing_position, axes=[rad_axis, energy_axis]
        )

        maker = MapDatasetMaker()
        psf = maker.make_psf(geom=geom, observation=obs)

        # Get a PSF kernel at the center of the image
        upsample_geom = geom.upsample(factor=10).drop("rad")
        psf_kernel = psf.get_psf_kernel(geom=upsample_geom)
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
        del self.has_single_spatial_bin
        self._irf_map = value

    def normalize(self):
        """Normalize PSF map."""
        self.psf_map.normalize(axis_name="rad")

    @classmethod
    def from_geom(cls, geom):
        """Create PSF map from geometry.

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
        axis_lon = MapAxis.from_edges(np.arange(npix_x[0] + 1) - 0.5, name="lon_idx")
        axis_lat = MapAxis.from_edges(np.arange(npix_y[0] + 1) - 0.5, name="lat_idx")
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
        """Containment at given coordinates.

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value.
        energy_true : `~astropy.units.Quantity`
            Energy true value.
        position : `~astropy.coordinates.SkyCoord`, optional
            Sky position. If None, the center of the map is chosen.
            Default is None.

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values.
        """
        if position is None:
            position = self.psf_map.geom.center_skydir

        coords = {"skycoord": position, "rad": rad, self.energy_name: energy_true}

        return self.psf_map.integral(axis_name="rad", coords=coords).to("")

    def containment_radius(self, fraction, energy_true, position=None):
        """Containment at given coordinates.

        Parameters
        ----------
        fraction : float
            Containment fraction.
        energy_true : `~astropy.units.Quantity`
            Energy true value.
        position : `~astropy.coordinates.SkyCoord`, optional
            Sky position. If None, the center of the map is chosen.
            Default is None.

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values.
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
            Energy true at which to compute the containment radius.
        fraction : float, optional
            Containment fraction (range: 0 to 1).
            Default is 0.68.

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map.
        """
        geom = self.psf_map.geom.to_image()

        data = self.containment_radius(
            fraction,
            energy_true,
            geom.get_coord().skycoord,
        )
        return Map.from_geom(geom=geom, data=data.value, unit=data.unit)

    @deprecated_renamed_argument(
        "factor", "precision_factor", "v1.2", arg_in_kwargs=True
    )
    def get_psf_kernel(
        self,
        geom,
        position=None,
        max_radius=None,
        containment=0.999,
        factor=None,
        precision_factor=12,
    ):
        """Return a PSF kernel at the given position.

        The PSF is returned in the form a WcsNDMap defined by the input Geom.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Target geometry to use.
        position : `~astropy.coordinates.SkyCoord`, optional
            Target position. Should be a single coordinate. By default, the
            center position is used.
        max_radius : `~astropy.coordinates.Angle`, optional
            Maximum angular size of the kernel map.
            Default is None and it will be computed for the `containment` fraction set.
        containment : float, optional
            Containment fraction to use as size of the kernel.
            The radius can be overwritten using the `max_radius` argument.
            Default is 0.999.
        factor : int, optional
            Oversampling factor to compute the PSF.
            Default is None and it will be computed automatically.
        precision_factor : int, optional
            Factor between the bin half-width of the geom and the median R68% containment radius.
            Used only for the oversampling method. Default is 10.

        Returns
        -------
        kernel : `~gammapy.irf.PSFKernel` or list of `PSFKernel`
            The resulting kernel.
        """

        if geom.is_region or geom.is_hpx:
            geom = geom.to_wcs_geom()

        if position is None:
            position = self.psf_map.geom.center_skydir

        position = self._get_nearest_valid_position(position)

        energy_axis = self.psf_map.geom.axes[self.energy_name]
        kwargs = {
            "fraction": containment,
            "position": position,
            self.energy_name: energy_axis.center,
        }
        radii = self.containment_radius(**kwargs)
        if max_radius is None:
            max_radius = np.max(radii)
        else:
            max_radius = u.Quantity(max_radius)
            radii[radii > max_radius] = max_radius

        n_radii = len(radii)
        if factor is None:  # TODO: this remove and the else once factor is deprecated
            factor = _psf_upsampling_factor(self, geom, position, precision_factor)
        else:
            factor = [factor] * n_radii
        geom = geom.to_odd_npix(max_radius=max_radius)
        kernel_map = Map.from_geom(geom=geom)
        for im, ind in zip(kernel_map.iter_by_image(keepdims=True), range(n_radii)):
            geom_image_cut = im.geom.to_odd_npix(max_radius=radii[ind]).upsample(
                factor=factor[ind]
            )

            coords = geom_image_cut.get_coord(sparse=True)
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
            kernel_image = Map.from_geom(
                geom=geom_image_cut, data=np.clip(data, 0, np.inf)
            )
            kernel_image = kernel_image.downsample(
                factor=factor[ind], preserve_counts=True
            )
            coords = kernel_image.geom.get_coord()
            im.fill_by_coord(coords, weights=kernel_image.data)
        return PSFKernel(kernel_map, normalize=True)

    def sample_coord(self, map_coord, random_state=0, chunk_size=10000):
        """Apply PSF corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`. Default is 0.
        chunk_size : int
            If set, this will slice the input MapCoord into smaller chunks of chunk_size elements.
            Default is 10000.

        Returns
        -------
        corr_coord : `~gammapy.maps.MapCoord`
            Sequence of PSF-corrected coordinates of the input map_coord map.
        """
        random_state = get_random_state(random_state)
        rad_axis = self.psf_map.geom.axes["rad"]

        position = map_coord.skycoord
        energy = map_coord[self.energy_name]

        size = position.size
        separation = np.ones(size) * u.deg
        chunk_size = size if chunk_size is None else chunk_size

        index = 0

        while index < size:
            chunk = slice(index, index + chunk_size, 1)
            coord = {
                "skycoord": position[chunk].reshape(-1, 1),
                self.energy_name: energy[chunk].reshape(-1, 1),
                "rad": rad_axis.center,
            }

            pdf = (
                self.psf_map.interp_by_coord(coord)
                * rad_axis.center.value
                * rad_axis.bin_width.value
            )

            sample_pdf = InverseCDFSampler(pdf, axis=1, random_state=random_state)
            pix_coord = sample_pdf.sample_axis()
            separation[chunk] = rad_axis.pix_to_coord(pix_coord)
            index += chunk_size

        position_angle = random_state.uniform(360, size=len(map_coord.lon)) * u.deg

        event_positions = map_coord.skycoord.directional_offset_by(
            position_angle=position_angle, separation=separation
        )
        return MapCoord.create({"skycoord": event_positions, self.energy_name: energy})

    @classmethod
    def from_gauss(cls, energy_axis_true, rad_axis=None, sigma=0.1 * u.deg, geom=None):
        """Create all-sky PSF map from Gaussian width.

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
            Image geometry. By default, an all-sky geometry is created.

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
        """Reduce to a 2D map after weighing with the associated exposure and a spectrum.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If True, the energy axis is kept with one bin.
            If False, the axis is removed.

        Returns
        -------
        psf_out : `PSFMap`
            `PSFMap` with the energy axis summed over.
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
        ax : `~matplotlib.pyplot.Axes`, optional
            Matplotlib axes. Default is None.
        fraction : list of float or `~numpy.ndarray`
            Containment fraction between 0 and 1.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Matplotlib axes.

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
        ax.set_ylabel(
            f"Containment radius [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]"
        )
        return ax

    def plot_psf_vs_rad(self, ax=None, energy_true=[0.1, 1, 10] * u.TeV, **kwargs):
        """Plot PSF vs radius.

        The method plots the profile at the center of the map.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`, optional
            Matplotlib axes. Default is None.
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Matplotlib axes.

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
            psf_value *= self.psf_map.unit
            with quantity_support():
                ax.plot(rad, psf_value, label=label, **kwargs)

        ax.set_yscale("log")
        ax.set_xlabel(f"Rad [{ax.xaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.set_ylabel(f"PSF [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]")
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
        if self.exposure_map is not None:
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
            Image geometry. By default, an all-sky geometry is created.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        return super().from_gauss(energy_axis, rad_axis, sigma, geom)

    def containment(self, rad, energy, position=None):
        """Containment at given coordinates.

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value.
        energy : `~astropy.units.Quantity`
            Energy value.
        position : `~astropy.coordinates.SkyCoord`, optional
            Sky position. By default, the center of the map is chosen.

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values.
        """
        return super().containment(rad, energy, position)

    def containment_radius(self, fraction, energy, position=None):
        """Containment at given coordinates.

        Parameters
        ----------
        fraction : float
            Containment fraction.
        energy : `~astropy.units.Quantity`
            Energy value.
        position : `~astropy.coordinates.SkyCoord`, optional
            Sky position. By default, the center of the map is chosen.

        Returns
        -------
        containment : `~astropy.units.Quantity`
            Containment values.
        """
        return super().containment_radius(fraction, energy, position)

    def containment_radius_map(self, energy, fraction=0.68):
        """Containment radius map.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to compute the containment radius
        fraction : float, optional
            Containment fraction (range: 0 to 1).
            Default is 0.68.

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map.
        """
        return super().containment_radius_map(energy, fraction=0.68)

    def plot_psf_vs_rad(self, ax=None, energy=[0.1, 1, 10] * u.TeV, **kwargs):
        """Plot PSF vs radius.

        The method plots the profile at the center of the map.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`, optional
            Matplotlib axes. Default is None.
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Matplotlib axes.

        """
        return super().plot_psf_vs_rad(ax, energy_true=energy, **kwargs)

    def stack(self, other, weights=None, nan_to_num=True):
        """Stack IRF map with another one in place."""
        raise NotImplementedError(
            "Stacking is not supported for PSF in reconstructed energy."
        )
