# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.interpolate import interp1d
from gammapy.maps import Map, MapAxis, MapCoord, RegionGeom, WcsGeom
from gammapy.utils.random import InverseCDFSampler, get_random_state
from .edisp_kernel import EDispKernel
from .irf_map import IRFMap

__all__ = ["EDispMap", "EDispKernelMap"]


def get_overlap_fraction(energy_axis, energy_axis_true):
    a_min = energy_axis.edges[:-1]
    a_max = energy_axis.edges[1:]

    b_min = energy_axis_true.edges[:-1][:, np.newaxis]
    b_max = energy_axis_true.edges[1:][:, np.newaxis]

    xmin = np.fmin(a_max, b_max)
    xmax = np.fmax(a_min, b_min)
    return np.clip(xmin - xmax, 0, np.inf) / (b_max - b_min)


class EDispMap(IRFMap):
    """Energy dispersion map.

    Parameters
    ----------
    edisp_map : `~gammapy.maps.Map`
        the input Energy Dispersion Map. Should be a Map with 2 non spatial axes.
        migra and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    ::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.maps import WcsGeom, MapAxis
        from gammapy.irf import EnergyDispersion2D, EffectiveAreaTable2D
        from gammapy.makers.utils import make_edisp_map, make_map_exposure_true_energy

        # Define energy dispersion map geometry
        energy_axis = MapAxis.from_edges(np.logspace(-1, 1, 4), unit="TeV", name="energy")
        migra_axis = MapAxis.from_edges(np.linspace(0, 3, 100), name="migra")
        pointing = SkyCoord(0, 0, unit="deg")
        max_offset = 4 * u.deg
        geom = WcsGeom.create(
            binsz=0.25 * u.deg,
            width=10 * u.deg,
            skydir=pointing,
            axes=[migra_axis, energy_axis],
        )

        # Extract EnergyDispersion2D from CTA 1DC IRF
        filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        edisp2D = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
        aeff2d = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

        # Create the exposure map
        exposure_geom = geom.to_image().to_cube([energy_axis])
        exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

        # create the EDispMap for the specified pointing
        edisp_map = make_edisp_map(edisp2D, pointing, geom, max_offset, exposure_map)

        # Get an Energy Dispersion (1D) at any position in the image
        pos = SkyCoord(2.0, 2.5, unit="deg")
        energy = np.logspace(-1.0, 1.0, 10) * u.TeV
        edisp = edisp_map.get_edisp_kernel(pos=pos, energy=energy)

        # Write map to disk
        edisp_map.write("edisp_map.fits")
    """

    _hdu_name = "edisp"

    def __init__(self, edisp_map, exposure_map=None):
        if edisp_map.geom.axes[1].name.upper() != "ENERGY_TRUE":
            raise ValueError("Incorrect energy axis position in input Map")

        if edisp_map.geom.axes[0].name.upper() != "MIGRA":
            raise ValueError("Incorrect migra axis position in input Map")

        super().__init__(irf_map=edisp_map, exposure_map=exposure_map)

    @property
    def edisp_map(self):
        return self._irf_map

    @edisp_map.setter
    def edisp_map(self, value):
        self._irf_map = value

    def get_edisp_kernel(self, position, energy_axis):
        """Get energy dispersion at a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinates
        energy_axis : `MapAxis`
            Reconstructed energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            the energy dispersion (i.e. rmf object)
        """
        if position is None:
            position = self.edisp_map.geom.center_skydir

        if position.size != 1:
            raise ValueError(
                "EnergyDispersion can be extracted at one single position only."
            )

        energy_axis_true = self.edisp_map.geom.axes["energy_true"]
        migra_axis = self.edisp_map.geom.axes["migra"]

        coords = {
            "skycoord": position,
            "migra": migra_axis.center.reshape((-1, 1, 1, 1)),
            "energy_true": energy_axis_true.center.reshape((1, -1, 1, 1)),
        }

        # Interpolate in the EDisp map. Squeeze to remove dimensions of length 1
        values = self.edisp_map.interp_by_coord(coords) * self.edisp_map.unit
        edisp_values = values[:, :, 0, 0]

        data = []

        for idx, energy_true in enumerate(energy_axis_true.center):
            # migration value of energy bounds
            migra = energy_axis.edges / energy_true

            cumsum = np.insert(edisp_values[:, idx], 0, 0).cumsum()
            with np.errstate(invalid="ignore", divide="ignore"):
                cumsum = np.nan_to_num(cumsum / cumsum[-1])

            f = interp1d(
                migra_axis.edges.value,
                cumsum,
                kind="linear",
                bounds_error=False,
                fill_value=(0, 1),
            )

            # We compute the difference between 2 successive bounds in energy
            # to get integral over reco energy bin
            integral = np.diff(np.clip(f(migra), a_min=0, a_max=1))
            data.append(integral)

        return EDispKernel(
            energy_axis_true=energy_axis_true, energy_axis=energy_axis, data=data
        )

    @classmethod
    def from_geom(cls, geom):
        """Create edisp map from geom.

        By default a diagonal edisp matrix is created.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Edisp map geometry.

        Returns
        -------
        edisp_map : `~gammapy.maps.EDispMap`
            Energy dispersion map.
        """
        if "energy_true" not in [ax.name for ax in geom.axes]:
            raise ValueError("EDispMap requires true energy axis")

        geom_exposure_edisp = geom.squash(axis_name="migra")
        exposure_edisp = Map.from_geom(geom_exposure_edisp, unit="m2 s")

        migra_axis = geom.axes["migra"]
        edisp_map = Map.from_geom(geom, unit="")
        migra_0 = migra_axis.coord_to_pix(1)

        # distribute over two pixels
        migra = geom.get_idx()[2]
        data = np.abs(migra - migra_0)
        data = np.where(data < 1, 1 - data, 0)
        edisp_map.quantity = data
        return cls(edisp_map, exposure_edisp)

    def sample_coord(self, map_coord, random_state=0):
        """Apply the energy dispersion corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        `~gammapy.maps.MapCoord`.
            Sequence of Edisp-corrected coordinates of the input map_coord map.
        """
        random_state = get_random_state(random_state)
        migra_axis = self.edisp_map.geom.axes["migra"]

        coord = {
            "skycoord": map_coord.skycoord.reshape(-1, 1),
            "energy_true": map_coord["energy_true"].reshape(-1, 1),
            "migra": migra_axis.center,
        }

        pdf_edisp = self.edisp_map.interp_by_coord(coord)

        sample_edisp = InverseCDFSampler(pdf_edisp, axis=1, random_state=random_state)
        pix_edisp = sample_edisp.sample_axis()
        migra = migra_axis.pix_to_coord(pix_edisp)

        energy_reco = map_coord["energy_true"] * migra

        return MapCoord.create({"skycoord": map_coord.skycoord, "energy": energy_reco})

    @classmethod
    def from_diagonal_response(cls, energy_axis_true, migra_axis=None):
        """Create an allsky EDisp map with diagonal response.

        Parameters
        ----------
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis
        migra_axis : `~gammapy.maps.MapAxis`
            Migra axis

        Returns
        -------
        edisp_map : `~gammapy.maps.EDispMap`
            Energy dispersion map.
        """
        migra_res = 1e-5
        migra_axis_default = MapAxis.from_bounds(
            1 - migra_res, 1 + migra_res, nbin=3, name="migra", node_type="edges"
        )

        migra_axis = migra_axis or migra_axis_default

        geom = WcsGeom.create(
            npix=(2, 1), proj="CAR", binsz=180, axes=[migra_axis, energy_axis_true]
        )

        return cls.from_geom(geom)

    def to_edisp_kernel_map(self, energy_axis):
        """Convert to map with edisp kernels

        Parameters
        ----------
        energy : `~gammapy.maps.MapAxis`
            Reconstructed enrgy axis.

        Returns
        -------
        edisp : `~gammapy.maps.EDispKernelMap`
            Energy dispersion kernel map.
        """
        axis = 0
        energy_axis_true = self.edisp_map.geom.axes["energy_true"]
        migra_axis = self.edisp_map.geom.axes["migra"]

        data = []

        for idx, energy_true in enumerate(energy_axis_true.center):
            # migration value of energy bounds
            migra = energy_axis.edges / energy_true

            edisp_energy_true = self.edisp_map.slice_by_idx({"energy_true": idx})

            cumsum = np.insert(edisp_energy_true.data, 0, 0, axis=axis).cumsum(
                axis=axis
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                cumsum = np.nan_to_num(cumsum / cumsum[slice(-2, -1)])

            f = interp1d(
                migra_axis.edges.value,
                cumsum,
                kind="linear",
                bounds_error=False,
                fill_value=(0, 1),
                axis=axis,
            )

            integral = np.diff(np.clip(f(migra), a_min=0, a_max=1), axis=axis)
            data.append(integral)

        data = np.stack(data)

        geom_image = self.edisp_map.geom.to_image()
        geom = geom_image.to_cube([energy_axis, energy_axis_true])
        edisp_kernel_map = Map.from_geom(geom=geom, data=data)
        exposure_map = None
        if self.exposure_map is not None:
            exposure_map = Map.from_geom(
                geom.squash(axis_name=energy_axis.name),
                data=self.exposure_map.data,
                unit=self.exposure_map.unit,
                meta=self.exposure_map.meta,
            )

        return EDispKernelMap(
            edisp_kernel_map=edisp_kernel_map, exposure_map=exposure_map
        )


class EDispKernelMap(IRFMap):
    """Energy dispersion kernel map.

    Parameters
    ----------
    edisp_kernel_map : `~gammapy.maps.Map`
        The input energy dispersion kernel map. Should be a Map with 2 non spatial axes.
        Reconstructed and and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        Associated exposure map. Needs to have a consistent map geometry.

    """

    tag = "edisp_kernel_map"
    _hdu_name = "edisp"

    def __init__(self, edisp_kernel_map, exposure_map):
        if edisp_kernel_map.geom.axes[1].name.upper() != "ENERGY_TRUE":
            raise ValueError("Incorrect energy axis position in input Map")

        if edisp_kernel_map.geom.axes[0].name.upper() != "ENERGY":
            raise ValueError("Incorrect migra axis position in input Map")

        super().__init__(irf_map=edisp_kernel_map, exposure_map=exposure_map)

    @property
    def edisp_map(self):
        return self._irf_map

    @edisp_map.setter
    def edisp_map(self, value):
        self._irf_map = value

    @classmethod
    def from_geom(cls, geom):
        """Create edisp map from geom.

        By default a diagonal edisp matrix is created.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Edisp map geometry.

        Returns
        -------
        edisp_map : `EDispKernelMap`
            Energy dispersion kernel map.
        """
        axis_names = [ax.name for ax in geom.axes]

        if "energy_true" not in axis_names:
            raise ValueError("EDispKernelMap requires true energy axis")

        if "energy" not in axis_names:
            raise ValueError("EDispKernelMap requires energy axis")

        geom_exposure = geom.squash(axis_name="energy")
        exposure = Map.from_geom(geom_exposure, unit="m2 s")

        energy_axis = geom.axes["energy"]
        energy_axis_true = geom.axes["energy_true"]

        data = get_overlap_fraction(energy_axis, energy_axis_true)

        edisp_kernel_map = Map.from_geom(geom, unit="")
        edisp_kernel_map.quantity += data[:, :, np.newaxis, np.newaxis]
        return cls(edisp_kernel_map=edisp_kernel_map, exposure_map=exposure)

    def get_edisp_kernel(self, position=None, energy_axis=None):
        """Get energy dispersion at a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord` or `~regions.SkyRegion`
            The target position. Should be a single coordinates
        energy_axis : `MapAxis`
            Reconstructed energy axis, only used for checking.

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            the energy dispersion (i.e. rmf object)
        """
        if energy_axis:
            assert energy_axis == self.edisp_map.geom.axes["energy"]

        if isinstance(self.edisp_map.geom, RegionGeom):
            kernel_map = self.edisp_map
        else:
            if position is None:
                position = self.edisp_map.geom.center_skydir

            kernel_map = self.edisp_map.to_region_nd_map(region=position)

        return EDispKernel(
            energy_axis_true=kernel_map.geom.axes["energy_true"],
            energy_axis=kernel_map.geom.axes["energy"],
            data=kernel_map.data[..., 0, 0],
        )

    @classmethod
    def from_diagonal_response(cls, energy_axis, energy_axis_true, geom=None):
        """Create an energy dispersion map with diagonal response.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            Energy axis.
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis
        geom : `~gammapy.maps.Geom`
            The (2D) geom object to use. Default creates an all sky geometry with 2 bins.

        Returns
        -------
        edisp_map : `EDispKernelMap`
            Energy dispersion kernel map.
        """
        if geom is None:
            geom = WcsGeom.create(
                npix=(2, 1), proj="CAR", binsz=180, axes=[energy_axis, energy_axis_true]
            )
        else:
            geom = geom.to_image().to_cube([energy_axis, energy_axis_true])

        return cls.from_geom(geom)

    @classmethod
    def from_edisp_kernel(cls, edisp, geom=None):
        """Create an energy dispersion map from the input 1D kernel.

        The kernel will be duplicated over all spatial bins.

        Parameters
        ----------
        edisp : `~gammapy.irfs.EDispKernel`
            the input 1D kernel.
        geom : `~gammapy.maps.Geom`
            The (2D) geom object to use. Default creates an all sky geometry with 2 bins.

        Returns
        -------
        edisp_map : `EDispKernelMap`
            Energy dispersion kernel map.
        """
        edisp_map = cls.from_diagonal_response(
            edisp.energy_axis, edisp.energy_axis_true, geom=geom
        )
        edisp_map.edisp_map.data *= 0
        edisp_map.edisp_map.data[:, :, ...] = edisp.pdf_matrix[
            :, :, np.newaxis, np.newaxis
        ]
        return edisp_map

    @classmethod
    def from_gauss(
        cls, energy_axis, energy_axis_true, sigma, bias, pdf_threshold=1e-6, geom=None
    ):
        """Create an energy dispersion map from the input 1D kernel.

        The kernel will be duplicated over all spatial bins.

        Parameters
        ----------
        edisp : `~gammapy.irfs.EDispKernel`
            the input 1D kernel.
        geom : `~gammapy.maps.Geom`
            The (2D) geom object to use. Default creates an all sky geometry with 2 bins.

        Returns
        -------
        edisp_map : `EDispKernelMap`
            Energy dispersion kernel map.
        """
        kernel = EDispKernel.from_gauss(
            energy=energy_axis.edges,
            energy_true=energy_axis_true.edges,
            sigma=sigma,
            bias=bias,
            pdf_threshold=pdf_threshold,
        )
        return cls.from_edisp_kernel(kernel, geom=geom)

    def to_image(self, weights=None):
        """"Return a 2D EdispKernelMap by summing over the reconstructed energy axis.

        Parameters
        ----------
        weights: `~gammapy.maps.Map`, optional
            Weights to be applied

        Returns
        -------
        edisp : `EDispKernelMap`
            Edisp kernel map
        """

        edisp = self.edisp_map.data
        if weights:
            edisp = edisp * weights.data

        data = np.sum(edisp, axis=1, keepdims=True)
        geom = self.edisp_map.geom.squash(axis_name="energy")
        edisp_map = Map.from_geom(geom=geom, data=data)
        return self.__class__(
            edisp_kernel_map=edisp_map, exposure_map=self.exposure_map
        )

    def resample_energy_axis(self, energy_axis, weights=None):
        """Returns a resampled EdispKernelMap

        Bins are grouped according to the edges of the reconstructed energy axis provided.
        The true energy is left unchanged.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            The reco energy axis to use for the reco energy grouping
        weights: `~gammapy.maps.Map`, optional
            Weights to be applied

        Returns
        -------
        edisp : `EDispKernelMap`
            Edisp kernel map
        """
        new_edisp_map = self.edisp_map.resample_axis(axis=energy_axis, weights=weights)
        return self.__class__(
            edisp_kernel_map=new_edisp_map, exposure_map=self.exposure_map
        )
