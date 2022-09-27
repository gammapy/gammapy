# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.maps import HpxNDMap, Map, RegionNDMap, WcsNDMap
from gammapy.modeling.models import PointSpatialModel, TemplateNPredModel

PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg

log = logging.getLogger(__name__)


class MapEvaluator:
    """Sky model evaluation on maps.

    Evaluates a sky model on a 3D map and returns a map of the predicted counts.
    Convolution with IRFs will be performed as defined in the sky_model

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask : `~gammapy.maps.Map`
        Mask to apply to the likelihood for fitting.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    use_cache : bool
        Use npred caching.
    """

    def __init__(
        self,
        model,
        exposure=None,
        psf=None,
        edisp=None,
        gti=None,
        mask=None,
        evaluation_mode="local",
        use_cache=True,
    ):

        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.mask = mask
        self.gti = gti
        self.use_cache = use_cache
        self._init_position = None
        self.contributes = True
        self.psf_containment = None

        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

        # TODO: this is preliminary solution until we have further unified the model handling
        if (
            isinstance(self.model, TemplateNPredModel)
            or self.model.spatial_model is None
            or self.model.evaluation_radius is None
        ):
            self.evaluation_mode = "global"

        # define cached computations
        self._cached_parameter_values = None
        self._cached_parameter_values_previous = None
        self._cached_parameter_values_spatial = None
        self._cached_position = (0, 0)
        self._computation_cache = None
        self._neval = 0  # for debugging
        self._renorm = 1
        self._spatial_oversampling_factor = 1
        if self.exposure is not None:
            if not self.geom.is_region or self.geom.region is not None:
                self.update_spatial_oversampling_factor(self.geom)

    def reset_cache_properties(self):
        """Reset cached properties."""
        del self._compute_npred
        del self._compute_flux_spatial

    @property
    def geom(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.exposure.geom

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        # TODO: simplify and clean up
        if isinstance(self.model, TemplateNPredModel):
            return False
        elif not self.contributes:
            return False
        elif self.exposure is None:
            return True
        elif self.geom.is_region:
            return False
        elif self.evaluation_mode == "global" or self.model.evaluation_radius is None:
            return False
        elif not self.parameters_spatial_changed(reset=False):
            return False
        else:
            return self.irf_position_changed

    @property
    def psf_width(self):
        """Width of the PSF"""
        if self.psf is not None:
            psf_width = np.max(self.psf.psf_kernel_map.geom.width)
        else:
            psf_width = 0 * u.deg
        return psf_width

    def use_psf_containment(self, geom):
        """Use psf containment for point sources and circular regions"""
        if not geom.is_region:
            return False

        is_point_model = isinstance(self.model.spatial_model, PointSpatialModel)
        is_circle_region = isinstance(geom.region, CircleSkyRegion)
        return is_point_model & is_circle_region

    @property
    def cutout_width(self):
        """Cutout width for the model component"""
        return self.psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)

    def update(self, exposure, psf, edisp, geom, mask):
        """Update MapEvaluator, based on the current position of the model component.

        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        psf : `gammapy.irf.PSFMap`
            PSF map.
        edisp : `gammapy.irf.EDispMap`
            Edisp map.
        geom : `WcsGeom`
            Counts geom
        mask : `~gammapy.maps.Map`
            Mask to apply to the likelihood for fitting.
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")

        # lookup edisp
        if edisp:
            energy_axis = geom.axes["energy"]
            self.edisp = edisp.get_edisp_kernel(
                position=self.model.position, energy_axis=energy_axis
            )

        # lookup psf
        if psf and self.model.spatial_model:
            energy_name = psf.energy_name
            if energy_name == "energy":
                geom_psf = geom
            else:
                geom_psf = exposure.geom

            if self.use_psf_containment(geom=geom_psf):
                energy_values = geom_psf.axes[energy_name].center.reshape((-1, 1, 1))
                kwargs = {energy_name: energy_values, "rad": geom.region.radius}
                self.psf_containment = psf.containment(**kwargs)
            else:
                if geom_psf.is_region or geom_psf.is_hpx:
                    geom_psf = geom_psf.to_wcs_geom()

                self.psf = psf.get_psf_kernel(
                    position=self.model.position,
                    geom=geom_psf,
                    containment=PSF_CONTAINMENT,
                )

        if self.evaluation_mode == "local":
            self.contributes = self.model.contributes(mask=mask, margin=self.psf_width)

            if self.contributes:
                self.exposure = exposure.cutout(
                    position=self.model.position, width=self.cutout_width, odd_npix=True
                )
        else:
            self.exposure = exposure

        if self.contributes:
            if not self.geom.is_region or self.geom.region is not None:
                self.update_spatial_oversampling_factor(self.geom)

        self.reset_cache_properties()
        self._computation_cache = None
        self._cached_parameter_previous = None

    def update_spatial_oversampling_factor(self, geom):
        """Update spatial oversampling_factor for model evaluation"""
        res_scale = self.model.evaluation_bin_size_min

        res_scale = res_scale.to_value("deg") if res_scale is not None else 0

        if geom.is_region or geom.is_hpx:
            geom = geom.to_wcs_geom()
        if res_scale != 0:
            factor = int(np.ceil(np.max(geom.pixel_scales.deg) / res_scale))
            self._spatial_oversampling_factor = factor

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.maps.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``
        """
        return self.model.evaluate_geom(self.geom, self.gti)

    def compute_flux(self, *arg):
        """Compute flux"""
        return self.model.integrate_geom(self.geom, self.gti)

    def compute_flux_psf_convolved(self, *arg):
        """Compute psf convolved and temporal model corrected flux."""
        value = self.compute_flux_spectral()

        if self.model.spatial_model:
            if self.psf_containment is not None:
                value = value * self.psf_containment
            else:
                value = value * self.compute_flux_spatial()

        if self.model.temporal_model:
            value *= self.compute_temporal_norm()

        return Map.from_geom(geom=self.geom, data=value.value, unit=value.unit)

    def compute_flux_spatial(self):
        """Compute spatial flux using caching"""
        if self.parameters_spatial_changed() or not self.use_cache:
            del self._compute_flux_spatial
        return self._compute_flux_spatial

    @lazyproperty
    def _compute_flux_spatial(self):
        """Compute spatial flux

        Returns
        ----------
        value: `~astropy.units.Quantity`
            Psf-corrected, integrated flux over a given region.
        """
        if self.geom.is_region:
            # We don't estimate spatial contributions if no psf are defined
            if self.geom.region is None or self.psf is None:
                return 1

            wcs_geom = self.geom.to_wcs_geom(width_min=self.cutout_width).to_image()

            if self.psf and self.model.apply_irf["psf"]:
                values = self._compute_flux_spatial_geom(wcs_geom)
            else:
                values = self.model.spatial_model.integrate_geom(
                    wcs_geom, oversampling_factor=1
                )
                axes = [self.geom.axes["energy_true"].squash()]
                values = values.to_cube(axes=axes)

            weights = wcs_geom.region_weights(regions=[self.geom.region])
            value = (values.quantity * weights).sum(axis=(1, 2), keepdims=True)
        else:
            value = self._compute_flux_spatial_geom(self.geom)

        return value

    def _compute_flux_spatial_geom(self, geom):
        """Compute spatial flux oversampling geom if necessary"""
        if not self.model.spatial_model.is_energy_dependent:
            geom = geom.to_image()
        value = self.model.spatial_model.integrate_geom(geom)

        if self.psf and self.model.apply_irf["psf"]:
            value = self.apply_psf(value)

        return value

    def compute_flux_spectral(self):
        """Compute spectral flux"""
        energy = self.geom.axes["energy_true"].edges
        value = self.model.spectral_model.integral(
            energy[:-1],
            energy[1:],
        )
        if self.geom.is_hpx:
            return value.reshape((-1, 1))
        else:
            return value.reshape((-1, 1, 1))

    def compute_temporal_norm(self):
        """Compute temporal norm"""
        integral = self.model.temporal_model.integral(
            self.gti.time_start, self.gti.time_stop
        )
        return np.sum(integral)

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = (flux.quantity * self.exposure.quantity).to_value("")
        return Map.from_geom(self.geom, data=npred, unit="")

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        tmp = npred.convolve(self.psf)
        return tmp

    def apply_edisp(self, npred):
        """Convolve map data with energy dispersion.

        Parameters
        ----------
        npred : `~gammapy.maps.Map`
            Predicted counts in true energy bins

        Returns
        -------
        npred_reco : `~gammapy.maps.Map`
            Predicted counts in reco energy bins
        """
        return npred.apply_edisp(self.edisp)

    @lazyproperty
    def _compute_npred(self):
        """Compute npred"""
        if isinstance(self.model, TemplateNPredModel):
            npred = self.model.evaluate()
        else:
            if not self.parameter_norm_only_changed:
                for method in self.methods_sequence:
                    values = method(self._computation_cache)
                    self._computation_cache = values
                npred = self._computation_cache
            else:
                npred = self._computation_cache * self.renorm()
        return npred

    @property
    def apply_psf_after_edisp(self):
        return (
            self.psf is not None and "energy" in self.psf.psf_kernel_map.geom.axes.names
        )

    def compute_npred(self):
        """Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        if self.parameters_changed or not self.use_cache:
            del self._compute_npred

        return self._compute_npred

    @property
    def parameters_changed(self):
        """Parameters changed"""
        values = self.model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values == values)

        if changed:
            self._cached_parameter_values = values

        return changed

    @property
    def parameter_norm_only_changed(self):
        """Only norm parameter changed"""
        norm_only_changed = False
        idx = self._norm_idx
        values = self.model.parameters.value
        if idx and self._computation_cache is not None:
            changed = self._cached_parameter_values_previous == values
            norm_only_changed = sum(changed) == 1 and changed[idx]

        if not norm_only_changed:
            self._cached_parameter_values_previous = values
        return norm_only_changed

    def parameters_spatial_changed(self, reset=True):
        """Parameters changed

        Parameters
        ----------
        reset : bool
            Reset cached values

        Returns
        -------
        changed : bool
            Whether spatial parameters changed.
        """
        values = self.model.spatial_model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values_spatial == values)

        if changed and reset:
            self._cached_parameter_values_spatial = values

        return changed

    @property
    def irf_position_changed(self):
        """Position for IRF changed"""

        # Here we do not use SkyCoord.separation to improve performance
        # (it avoids equivalence comparisons for frame and units)
        lon_cached, lat_cached = self._cached_position
        lon, lat = self.model.position_lonlat

        separation = angular_separation(lon, lat, lon_cached, lat_cached)
        changed = separation > (self.model.evaluation_radius + CUTOUT_MARGIN).to_value(
            u.rad
        )

        if changed:
            self._cached_position = lon, lat

        return changed

    @lazyproperty
    def _norm_idx(self):
        """norm index"""
        names = self.model.parameters.names
        ind = [idx for idx, name in enumerate(names) if name in ["norm", "amplitude"]]
        if len(ind) == 1:
            return ind[0]
        else:
            return None

    def renorm(self):
        value = self.model.parameters.value[self._norm_idx]
        value_cached = self._cached_parameter_values_previous[self._norm_idx]
        return value / value_cached

    @lazyproperty
    def methods_sequence(self):
        """order to apply irf"""

        if self.apply_psf_after_edisp:
            methods = [
                self.compute_flux,
                self.apply_exposure,
                self.apply_edisp,
                self.apply_psf,
            ]
            if not self.psf or not self.model.apply_irf["psf"]:
                methods.remove(self.apply_psf)
        else:
            methods = [
                self.compute_flux_psf_convolved,
                self.apply_exposure,
                self.apply_edisp,
            ]
        if not self.model.apply_irf["exposure"]:
            methods.remove(self.apply_exposure)
        if not self.model.apply_irf["edisp"]:
            methods.remove(self.apply_edisp)
        return methods

    def peek(self, figsize=(12, 15)):
        """Quick-look summary plots.
        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        if self.needs_update:
            raise AttributeError(
                "The evaluator needs to be updated first. Execute "
                "`MapDataset.npred_signal(model_name=...)` before calling this method."
            )

        nrows = 1
        if self.psf:
            nrows += 1
        if self.edisp:
            nrows += 1

        fig, axes = plt.subplots(
            ncols=2,
            nrows=nrows,
            subplot_kw={"projection": self.exposure.geom.wcs},
            figsize=figsize,
            gridspec_kw={"hspace": 0.2, "wspace": 0.3},
        )

        axes = axes.flat

        exposure = self.exposure
        if isinstance(exposure, WcsNDMap) or isinstance(exposure, HpxNDMap):
            axes[0].set_title("Predicted counts")
            self.compute_npred().sum_over_axes().plot(ax=axes[0], add_cbar=True)

            axes[1].set_title("Exposure")
            self.exposure.sum_over_axes().plot(ax=axes[1], add_cbar=True)
        elif isinstance(exposure, RegionNDMap):
            axes[0].remove()
            ax0 = fig.add_subplot(nrows, 2, 1)
            ax0.set_title("Predicted counts")
            self.compute_npred().plot(ax=ax0)

            axes[1].remove()
            ax1 = fig.add_subplot(nrows, 2, 2)
            ax1.set_title("Exposure")
            self.exposure.plot(ax=ax1)

        idx = 3
        if self.psf:
            axes[2].set_title("Energy-integrated PSF kernel")
            self.psf.plot_kernel(ax=axes[2], add_cbar=True)

            axes[3].set_title("PSF kernel at 1 TeV")
            self.psf.plot_kernel(ax=axes[3], add_cbar=True, energy=1 * u.TeV)

            idx += 2

        if self.edisp:
            axes[idx - 1].remove()
            ax = fig.add_subplot(nrows, 2, idx)
            ax.set_title("Energy bias")
            self.edisp.plot_bias(ax=ax)

            axes[idx].remove()
            ax = fig.add_subplot(nrows, 2, idx + 1)
            ax.set_title("Energy dispersion matrix")
            self.edisp.plot_matrix(ax=ax)
