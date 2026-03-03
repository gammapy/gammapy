import html
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import angular_separation
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.irf import EDispKernel, PSFKernel
from gammapy.maps import Map
from gammapy.modeling.models import PointSpatialModel, TemplateNPredModel
from gammapy.utils.integrate import integrate_histogram
from .utils import apply_edisp

PSF_MAX_RADIUS = None
PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg

log = logging.getLogger(__name__)


class UnbinnedEvaluator:
    """Sky model evaluation on maps.

    Evaluates a sky model on a 3D map and returns a map of the predicted counts.
    The convolution with IRFs will be performed as defined in the sky_model. To do so, IRF kernels
    are extracted at the position closest to the position of the model.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Sky model.
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel.
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion.
    mask : `~gammapy.maps.Map`
        Mask to apply to the likelihood for fitting.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation.
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
        geom,
        geom_normalization,
        exposure=None,
        # exposure_original_irf=None,
        psf=None,
        edisp=None,
        edisp_e_reco_binned=None,
        gti=None,
        mask=None,
        evaluation_mode="local",
        use_cache=True,
    ):
        self.model = model
        self.exposure = exposure
        self._geom = geom
        self._geom_normalization = geom_normalization
        self.mask = mask
        self.gti = gti
        self.use_cache = use_cache
        self.contributes = True
        self.psf_containment = None

        self._geom_reco_axis = None

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

        self.update(
            exposure,
            psf,
            edisp,
            geom,
            mask,
            edisp_e_reco_binned=edisp_e_reco_binned,
            # exposure_original_irf=exposure_original_irf,
        )

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def reset_cache_properties(self):
        """Reset cached properties."""
        del self._compute_flux_spatial
        self._computation_cache = None
        self._cached_parameter_previous = None

    @property
    def geom(self):
        """True energy map geometry (`~gammapy.maps.Geom`)."""
        if self.exposure is not None:
            return self.exposure.geom
        else:
            return None

    @property
    def _geom_reco(self):
        if self.edisp is not None:
            energy_axis = self.edisp.axes["energy"].copy()
        elif self._geom_reco_axis is not None:
            energy_axis = self._geom_reco_axis
        else:
            energy_axis = self.geom.axes["energy_true"].copy(name="energy")
        geom = self.geom.to_image().to_cube(axes=[energy_axis])
        return geom

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        if isinstance(self.model, TemplateNPredModel):
            return False
        elif not self.contributes:
            return False
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
        """Width of the PSF."""
        if self.psf is not None:
            psf_width = np.max(self.psf.psf_kernel_map.geom.width)
        else:
            psf_width = 0 * u.deg
        return psf_width

    def use_psf_containment(self, geom):
        """Use PSF containment for point sources and circular regions."""
        if not geom.is_region:
            return False

        is_point_model = isinstance(self.model.spatial_model, PointSpatialModel)
        is_circle_region = isinstance(geom.region, CircleSkyRegion)
        return is_point_model & is_circle_region

    @lazyproperty
    def position(self):
        """Latest evaluation position."""
        return self.model.position

    @lazyproperty
    def cutout_width(self):
        """Cutout width for the model component."""
        return self.psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)

    def update(
        self,
        exposure,
        psf,
        edisp,
        geom,
        mask,
        edisp_e_reco_binned=None,
        # exposure_original_irf=None,
    ):
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
            Counts geom.
        mask : `~gammapy.maps.Map`
            Mask to apply to the likelihood for fitting.
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")

        del self.position
        del self.cutout_width

        self._geom_reco_axis = geom.axes["energy"]

        # lookup edisp
        del self._edisp_diagonal
        if edisp:
            energy_axis = geom.axes["energy"]
            self.edisp = edisp.get_edisp_kernel(
                position=self.position, energy_axis=energy_axis
            )
            del self._edisp_diagonal

        else:
            self.edisp = None
        if edisp_e_reco_binned is not None:
            energy_axis = edisp_e_reco_binned.axes["energy"]
            self.edisp_e_reco_binned = edisp_e_reco_binned.get_edisp_kernel(
                position=self.position, energy_axis=energy_axis
            )
        else:
            self.edisp_e_reco_binned = None

        # lookup psf
        if (
            psf
            and self.model.spatial_model
            and not (isinstance(self.psf, PSFKernel) and psf.has_single_spatial_bin)
        ):
            energy_name = psf.energy_name
            geom_psf = geom if energy_name == "energy" else exposure.geom

            if self.use_psf_containment(geom=geom_psf):
                energy_values = geom_psf.axes[energy_name].center.reshape((-1, 1, 1))
                kwargs = {energy_name: energy_values, "rad": geom.region.radius}
                self.psf_containment = psf.containment(**kwargs)
            else:
                self.psf = psf.get_psf_kernel(
                    position=self.position,
                    geom=geom_psf,
                    containment=PSF_CONTAINMENT,
                    max_radius=PSF_MAX_RADIUS,
                )
        else:
            self.psf = None
            self.psf_containment = None

        self.exposure = exposure
        if self.evaluation_mode == "local":
            self.contributes = self.model.contributes(mask=mask, margin=self.psf_width)
            if self.contributes and not self.model.contributes(mask=mask):
                log.warning(
                    f"Model {self.model.name} is outside the target geom but contributes inside through the psf."
                    "This contribution cannot be estimated precisely."
                    "Consider extending the dataset geom and/or the masked margin in the mask_fit."
                )
            if self.contributes and not self.geom.is_region:
                self.exposure = exposure._cutout_view(
                    position=self.position, width=self.cutout_width, odd_npix=True
                )

        self.reset_cache_properties()

    @lazyproperty
    def _edisp_diagonal(self):  # TO BE MODIFIED
        return EDispKernel.from_diagonal_response(
            energy_axis_true=self.geom.axes["energy_true"],
            energy_axis=self._geom_reco.axes["energy"],
        )

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.maps.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``.
        """
        value = self.model.evaluate_geom(self.geom, self.gti)
        return Map.from_geom(geom=self.geom, data=value.value, unit=value.unit)

    def compute_flux(self, geom=None):
        """Compute flux."""
        if geom is None:
            geom = self.geom
        return self.model.integrate_geom(geom, self.gti)

    def compute_flux_psf_convolved(self, *arg):
        """Compute PSF convolved and temporal model corrected flux."""
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
        """Compute spatial flux using caching."""
        if self.parameters_spatial_changed() or not self.use_cache:
            del self._compute_flux_spatial
        return self._compute_flux_spatial

    @lazyproperty
    def _compute_flux_spatial(self):
        """Compute spatial flux.

        Returns
        ----------
        value: `~astropy.units.Quantity`
            PSF-corrected, integrated flux over a given region.
        """
        if self.geom.is_region:
            # We don't estimate spatial contributions if no psf are defined
            if self.geom.region is None or self.psf is None:
                return 1

            wcs_geom = self.geom.to_wcs_geom(width_min=self.cutout_width)
            values = self._compute_flux_spatial_geom(wcs_geom)

            if not values.geom.has_energy_axis:
                axes = [self.geom.axes["energy_true"].squash()]
                values = values.to_cube(axes=axes)

            weights = wcs_geom.region_weights(regions=[self.geom.region])
            value = (values.quantity * weights).sum(axis=(1, 2), keepdims=True)
        else:
            value = self._compute_flux_spatial_geom(self.geom)

        return value

    def _compute_flux_spatial_geom(self, geom):
        """Compute spatial flux oversampling geom if necessary."""
        if not self.model.spatial_model.is_energy_dependent:
            geom = geom.to_image()
        value = self.model.spatial_model.integrate_geom(geom)

        if self.psf and self.model.apply_irf["psf"]:
            value = self.apply_psf(value)

        return value

    def compute_flux_spectral(self):
        """Compute spectral flux."""
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
        """Compute temporal norm."""
        integral = self.model.temporal_model.integral(
            self.gti.time_start, self.gti.time_stop
        )
        return np.sum(integral)

    def apply_psf(self, npred):
        """Convolve npred cube with PSF."""
        return npred.convolve(self.psf)

    def apply_edisp(self, npred, edisp=None):
        """Convolve map data with energy dispersion.

        Parameters
        ----------
        npred : `~gammapy.maps.Map`
            Predicted counts in true energy bins.

        Returns
        -------
        npred_reco : `~gammapy.maps.Map`
            Predicted counts in reconstructed energy bins.
        """
        if edisp is None:
            edisp = self.edisp

        if self.model.apply_irf["edisp"] and edisp:
            return apply_edisp(npred, edisp)
        else:
            if "energy_true" in npred.geom.axes.names:
                return apply_edisp(npred, self._edisp_diagonal)
            else:
                return npred

    @property
    def apply_psf_after_edisp(self):
        return (
            self.psf is not None and "energy" in self.psf.psf_kernel_map.geom.axes.names
        )

    def apply_exposure(self, flux, original_irf=False):
        """Compute npred cube.

        For now just divide flux cube by exposure.
        """
        # if original_irf:
        #    geom = self._geom_normalization
        #    exposure = self.exposure_original_irf
        # else:
        #    geom = self.geom
        #    exposure = self.exposure
        npred = (flux.quantity * self.exposure.quantity).to_value("")
        return Map.from_geom(self.geom, data=npred, unit="")

    def compute_npred(self):
        """Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reconstructed energy bins).
        """
        flux_from_binned = self.compute_flux()
        flux_from_binned = self.apply_exposure(flux_from_binned)
        flux_from_binned = self.apply_edisp(
            flux_from_binned, edisp=self.edisp_e_reco_binned
        )
        return flux_from_binned

    def _get_factor_to_normalize_edisp(self, energy_min, energy_max):
        """Normalize the energy dispersion matrix to 1 in each true energy bin."""
        # mask = self.edisp.axes['energy'].center >= energy_min
        # mask &= self.edisp.axes['energy'].center <= energy_max
        # args = np.argsort(self.edisp.axes['energy'].center[mask])
        # energy_reco_sorted = self.edisp.axes['energy'].center[mask][args]
        # factor = simpson(self.edisp.data.T[mask][args].T, energy_reco_sorted, axis=1)
        differencial_edisp_map_e_reco_binned = (
            self.edisp_e_reco_binned.divide_bin_width("energy")
        )

        factor = (
            integrate_histogram(
                differencial_edisp_map_e_reco_binned.data,
                differencial_edisp_map_e_reco_binned.axes["energy"].edges,
                energy_min,
                energy_max,
                axis=differencial_edisp_map_e_reco_binned.axes.index("energy"),
            )[0]
            * differencial_edisp_map_e_reco_binned.unit
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.nan_to_num(factor**-1, posinf=0, neginf=0, nan=0)
        return factor

    def compute_signal_pdf(
        self, energy_min=None, energy_max=None, return_normalization_factor=False
    ):
        """Compute signal probability density function (PDF) of the model over the map energy range.
        This implementation is based on the independence of exposure and model to E_reco.

        Returns
        -------
        pdf : `~numpy.ndarray`
            Probability density function (PDF) of the model.
        """

        flux = self.compute_flux()
        flux = flux * self.exposure
        if energy_min is not None or energy_max is not None:
            if energy_min is None:
                energy_min = self.edisp.axes["energy"].center.min()
            if energy_max is None:
                energy_max = self.edisp.axes["energy"].center.max()
            # mask = np.logical_and(
            #    self.edisp.axes['energy'].center >= energy_min,
            #    self.edisp.axes['energy'].center <= energy_max
            # )
            # args = np.argsort(self.edisp.axes['energy'].center[mask])
            # energy_reco_sorted = self.edisp.axes['energy'].center[mask][args]

            npred_binned = self.apply_edisp(
                flux, edisp=self.edisp_e_reco_binned.divide_bin_width("energy")
            )
            normalization_factor = (
                integrate_histogram(
                    npred_binned.data,
                    npred_binned.geom.axes["energy"].edges,
                    energy_min,
                    energy_max,
                    axis=npred_binned.geom.axes.index_data("energy"),
                )[0]
                * npred_binned.unit
            )

            # factor = self._get_factor_to_normalize_edisp(energy_min, energy_max)
            # flux *= factor.reshape(flux.data.shape)
            npred = self.apply_edisp(flux)

        else:
            npred = self.apply_edisp(flux)
            normalization_factor = flux.quantity.sum(axis=0)
        pdf = npred / normalization_factor
        # pdf.data = np.clip(pdf.data, 0, None)#to avoid negative pdf values
        if return_normalization_factor:
            return pdf, normalization_factor
        else:
            return pdf

    @property
    def parameters_changed(self):
        """Parameters changed."""
        values = self.model.parameters.value
        changed = ~np.all(self._cached_parameter_values == values)

        if changed:
            self._cached_parameter_values = values

        return changed

    @property
    def parameter_norm_only_changed(self):
        """Only norm parameter changed."""
        norm_only_changed = False
        idx = self._norm_idx
        values = self.model.parameters.value
        if idx is not None and self._computation_cache is not None:
            changed = self._cached_parameter_values_previous != values
            norm_only_changed = np.count_nonzero(changed) == 1 and changed[idx]

        if not norm_only_changed:
            self._cached_parameter_values_previous = values
        return norm_only_changed

    def parameters_spatial_changed(self, reset=True):
        """Parameters changed.

        Parameters
        ----------
        reset : bool
            Reset cached values. Default is True.

        Returns
        -------
        changed : bool
            Whether spatial parameters changed.
        """
        values = self.model.spatial_model.parameters.value
        changed = ~np.all(self._cached_parameter_values_spatial == values)

        if changed and reset:
            self._cached_parameter_values_spatial = values

        return changed

    @property
    def irf_position_changed(self):
        """Position for IRF changed."""

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
        """Norm index."""
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
        """Order to apply the IRFs."""

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
            if not self.psf or not self.model.apply_irf["psf"]:
                methods = [
                    self.compute_flux,
                    self.apply_exposure,
                    self.apply_edisp,
                ]
            else:
                methods = [
                    self.compute_flux_psf_convolved,
                    self.apply_exposure,
                    self.apply_edisp,
                ]
        # methods.append(self.apply_normalization_factor)
        return methods
