# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation."""
import itertools
import numpy as np
from astropy.convolution import Ring2DKernel, Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.cube.fit import MapDatasetOnOff
from gammapy.maps import Map, scale_cube

__all__ = ["AdaptiveRingBackgroundMaker", "RingBackgroundMaker"]


class AdaptiveRingBackgroundMaker:
    """Adaptive ring background algorithm.

    This algorithm extends the `RingBackgroundMaker` method by adapting the
    size of the ring to achieve a minimum on / off exposure ratio (alpha) in regions
    where the area to estimate the background from is limited.

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner radius of the ring.
    r_out_max : `~astropy.units.Quantity`
        Maximal outer radius of the ring.
    width : `~astropy.units.Quantity`
        Width of the ring.
    stepsize : `~astropy.units.Quantity`
        Stepsize used for increasing the radius.
    threshold_alpha : float
        Threshold on alpha above which the adaptive ring takes action.
    theta : `~astropy.units.Quantity`
        Integration radius used for alpha computation.
    method : {'fixed_width', 'fixed_r_in'}
        Adaptive ring method.
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask

    See Also
    --------
    RingBackgroundMaker, gammapy.detect.KernelBackgroundEstimator
    """

    def __init__(
        self,
        r_in,
        r_out_max,
        width,
        stepsize="0.02 deg",
        threshold_alpha=0.1,
        theta="0.22 deg",
        method="fixed_width",
        exclusion_mask=None,
    ):
        if method not in ["fixed_width", "fixed_r_in"]:
            raise ValueError("Not a valid adaptive ring method.")

        self.r_in = Angle(r_in)
        self.r_out_max = Angle(r_out_max)
        self.width = Angle(width)
        self.stepsize = Angle(stepsize)
        self.threshold_alpha = threshold_alpha
        self.theta = Angle(theta)
        self.method = method
        self.exclusion_mask = exclusion_mask

    def kernels(self, image):
        """Ring kernels according to the specified method.

        Parameters
        ----------
        image : `~gammapy.maps.WcsNDMap`
            Map specifying the WCS information.

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Ring2DKernel`
        """
        scale = image.geom.pixel_scales[0]
        r_in = (self.r_in / scale).to_value("")
        r_out_max = (self.r_out_max / scale).to_value("")
        width = (self.width / scale).to_value("")
        stepsize = (self.stepsize / scale).to_value("")

        if self.method == "fixed_width":
            r_ins = np.arange(r_in, (r_out_max - width), stepsize)
            widths = [width]
        elif self.method == "fixed_r_in":
            widths = np.arange(width, (r_out_max - r_in), stepsize)
            r_ins = [r_in]
        else:
            raise ValueError(f"Invalid method: {self.method!r}")

        kernels = []
        for r_in, width in itertools.product(r_ins, widths):
            kernel = Ring2DKernel(r_in, width)
            kernel.normalize("peak")
            kernels.append(kernel)

        return kernels

    @staticmethod
    def _alpha_approx_cube(cubes):
        """Compute alpha as acceptance / acceptance_off.
        Where acceptance_off < 0, alpha is set to infinity.
        """
        acceptance = cubes["acceptance"]
        acceptance_off = cubes["acceptance_off"]
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha_approx = np.where(
                acceptance_off > 0, acceptance / acceptance_off, np.inf
            )

        return alpha_approx

    def _reduce_cubes(self, cubes, dataset):
        """Compute off and off acceptance map.

        Calulated by reducing the cubes. The data is
        iterated along the third axis (i.e. increasing ring sizes), the value
        with the first approximate alpha < threshold is taken.
        """
        threshold = self.threshold_alpha

        alpha_approx_cube = self._alpha_approx_cube(cubes)
        counts_off_cube = cubes["counts_off"]
        acceptance_off_cube = cubes["acceptance_off"]
        acceptance_cube = cubes["acceptance"]

        shape = alpha_approx_cube.shape[:2]
        counts_off = np.tile(np.nan, shape)
        acceptance_off = np.tile(np.nan, shape)
        acceptance = np.tile(np.nan, shape)

        for idx in np.arange(alpha_approx_cube.shape[-1]):
            mask = (alpha_approx_cube[:, :, idx] <= threshold) & np.isnan(counts_off)
            counts_off[mask] = counts_off_cube[:, :, idx][mask]
            acceptance_off[mask] = acceptance_off_cube[:, :, idx][mask]
            acceptance[mask] = acceptance_cube[:, :, idx][mask]

        counts = dataset.counts
        acceptance = counts.copy(data=acceptance[np.newaxis, Ellipsis])
        acceptance_off = counts.copy(data=acceptance_off[np.newaxis, Ellipsis])
        counts_off = counts.copy(data=counts_off[np.newaxis, Ellipsis])

        return acceptance, acceptance_off, counts_off

    def make_cubes(self, dataset):
        """Make acceptance, off acceptance, off counts cubes

        Parameters
        ----------
        dataset : `~gammapy.cube.fit.MapDataset`
            Input map dataset.

        Returns
        -------
        cubes : dict of `~gammapy.maps.WcsNDMap`
            Dictionary containing `counts_off`, `acceptance` and `acceptance_off` cubes.
        """

        counts = dataset.counts
        background = dataset.background_model.map
        kernels = self.kernels(counts)

        if self.exclusion_mask is not None:
            # reproject exclusion mask
            coords = counts.geom.get_coord()
            data = self.exclusion_mask.get_by_coord(coords)
            exclusion = Map.from_geom(geom=counts.geom, data=data)
        else:
            data = np.ones(counts.geom.data_shape, dtype=bool)
            exclusion = Map.from_geom(geom=counts.geom, data=data)

        cubes = {}
        cubes["counts_off"] = scale_cube(
            (counts.data * exclusion.data)[0, Ellipsis], kernels
        )
        cubes["acceptance_off"] = scale_cube(
            (background.data * exclusion.data)[0, Ellipsis], kernels
        )

        scale = background.geom.pixel_scales[0].to("deg")
        theta = self.theta * scale
        tophat = Tophat2DKernel(theta.value)
        tophat.normalize("peak")
        acceptance = background.convolve(tophat.array)
        acceptance_data = acceptance.data[0, Ellipsis]
        cubes["acceptance"] = np.repeat(
            acceptance_data[Ellipsis, np.newaxis], len(kernels), axis=2
        )

        return cubes

    def run(self, dataset):
        """Run adaptive ring background maker

        Parameters
        ----------
        dataset : `~gammapy.cube.fit.MapDataset`
            Input map dataset.
        exclusion : `~gammapy.maps.WcsNDMap`
            Exclusion mask for regions with known gamma-ray emission.

        Returns
        -------
        dataset_on_off : `~gammapy.cube.fit.MapDatasetOnOff`
            On off dataset.
        """
        cubes = self.make_cubes(dataset)
        acceptance, acceptance_off, counts_off = self._reduce_cubes(cubes, dataset)

        mask_safe = dataset.mask_safe.copy()
        not_has_off_acceptance = acceptance_off.data <= 0
        mask_safe.data[not_has_off_acceptance] = 0

        return MapDatasetOnOff(
            counts=dataset.counts,
            counts_off=counts_off,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            exposure=dataset.exposure,
            psf=dataset.psf,
            name=dataset.name,
            mask_safe=mask_safe,
            gti=dataset.gti,
        )


class RingBackgroundMaker:
    """Ring background method for cartesian coordinates.

    - Step 1: apply exclusion mask
    - Step 2: ring-correlate

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner ring radius
    width : `~astropy.units.Quantity`
        Ring width
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask

    See Also
    --------
    gammapy.detect.KernelBackgroundEstimator, AdaptiveRingBackgroundEstimator
    """

    def __init__(self, r_in, width, exclusion_mask=None):
        self.r_in = Angle(r_in)
        self.width = Angle(width)
        self.exclusion_mask = exclusion_mask

    def kernel(self, image):
        """Ring kernel.

        Parameters
        ----------
        image : `~gammapy.maps.WcsNDMap`
            Input Map

        Returns
        -------
        ring : `~astropy.convolution.Ring2DKernel`
            Ring kernel.
        """
        scale = image.geom.pixel_scales[0].to("deg")
        r_in = self.r_in.to("deg") / scale
        width = self.width.to("deg") / scale

        ring = Ring2DKernel(r_in.value, width.value)
        ring.normalize("peak")
        return ring

    def make_maps_off(self, dataset):
        """Make off maps

        Parameters
        ----------
        dataset : `~gammapy.cube.fit.MapDataset`
            Input map dataset.

        Returns
        -------
        maps_off : dict of `~gammapy.maps.WcsNDMap`
            Dictionary containing `counts_off` and `acceptance_off` maps.
        """
        counts = dataset.counts
        background = dataset.background_model.map

        if self.exclusion_mask is not None:
            # reproject exclusion mask
            coords = counts.geom.get_coord()
            data = self.exclusion_mask.get_by_coord(coords)
            exclusion = Map.from_geom(geom=counts.geom, data=data)
        else:
            data = np.ones(counts.geom.data_shape, dtype=bool)
            exclusion = Map.from_geom(geom=counts.geom, data=data)

        maps_off = {}
        ring = self.kernel(counts)

        counts_excluded = counts * exclusion
        maps_off["counts_off"] = counts_excluded.convolve(ring.array)

        background_excluded = background * exclusion
        maps_off["acceptance_off"] = background_excluded.convolve(ring.array)

        return maps_off

    def run(self, dataset):
        """Run ring background maker

        Parameters
        ----------
        dataset : `~gammapy.cube.fit.MapDataset`
            Input map dataset.

        Returns
        -------
        dataset_on_off : `~gammapy.cube.fit.MapDatasetOnOff`
            On off dataset.
        """
        maps_off = self.make_maps_off(dataset)
        acceptance = dataset.background_model.map

        mask_safe = dataset.mask_safe.copy()
        not_has_off_acceptance = maps_off["acceptance_off"].data <= 0
        mask_safe.data[not_has_off_acceptance] = 0

        return MapDatasetOnOff(
            counts=dataset.counts,
            counts_off=maps_off["counts_off"],
            acceptance=acceptance,
            acceptance_off=maps_off["acceptance_off"],
            exposure=dataset.exposure,
            psf=dataset.psf,
            name=dataset.name,
            mask_safe=dataset.mask_safe,
            gti=dataset.gti,
        )

    def __str__(self):
        return (
            "RingBackground parameters: \n"
            f"r_in : {self.parameters['r_in']}\n"
            f"width: {self.parameters['width']}\n"
            f"Exclusion mask: {self.exclusion_mask}"
        )
