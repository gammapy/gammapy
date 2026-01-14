# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import logging
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import (
    DatasetModels,
    FoVBackgroundModel,
    NormBackgroundSpectralModel,
    Models,
)
from gammapy.utils.scripts import make_name
from gammapy.utils.fits import LazyFitsData, HDULocation
from gammapy.utils.integrate import integrate_histogram
from gammapy.irf import EDispMap, EDispKernelMap, PSFMap, RecoPSFMap
from gammapy.maps import Map, MapAxes, LabelMapAxis
from gammapy.data import GTI
from .unbinned_evaluator import UnbinnedEvaluator
from .core import Dataset
from .spectrum import PlotMixin
from .map import BINSZ_IRF_DEFAULT, RAD_AXIS_DEFAULT, MIGRA_AXIS_DEFAULT

log = logging.getLogger(__name__)

EVALUATION_MODE = "local"
USE_NPRED_CACHE = True


class EventDataset(Dataset, PlotMixin):
    """ """

    tag = "EventDataset"
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
    background = LazyFitsData(cache=True)
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = [
        "background",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "mask_safe",
    ]

    gti = None
    meta_table = None

    def __init__(
        self,
        events=None,
        geom=None,
        geom_normalization=None,
        models=None,
        background=None,
        exposure=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        name=None,
        reference_time="2000-01-01",
        gti=None,
        meta=None,
        edisp_e_reco_binned=None,
        # exposure_original_irf=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}
        # self.position = position
        self.geom = geom
        self.geom_normalization = geom_normalization

        self.events = events
        self.exposure = exposure
        if not isinstance(self, EventDatasetOnOff):
            self.background = background
        # self._background_cached = None
        # self._background_parameters_cached = None

        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

        self.reference_time = reference_time
        self.gti = gti
        self.models = models
        self.meta_table = meta_table

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )
        self.psf = psf

        if edisp is not None and not isinstance(
            edisp, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )
        if edisp_e_reco_binned is not None and not isinstance(
            edisp_e_reco_binned, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp_e_reco_binned' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp_e_reco_binned)}` instead."
            )

        self.edisp = edisp
        self.meta = meta
        self.edisp_e_reco_binned = edisp_e_reco_binned
        # self.exposure_original_irf = exposure_original_irf

    @property
    def _geom(self):
        """Main analysis geometry."""
        return self.geom

    def __str__(self):
        pass

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators

    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models

    @models.setter
    def models(self, models):
        """Models setter."""
        self._evaluators = {}
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = UnbinnedEvaluator(
                        model=model,
                        geom=self.geom,
                        geom_normalization=self.geom_normalization,
                        psf=self.psf,
                        edisp=self.edisp,
                        edisp_e_reco_binned=self.edisp_e_reco_binned,
                        exposure=self.exposure,
                        # exposure_original_irf=self.exposure_original_irf,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                    )
                    self._evaluators[model.name] = evaluator
        self._models = models

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = value

    @property
    def mask_event(self):
        """Entry for each event whether it is inside the mask or not"""
        if self.mask is None:
            return np.ones(len(self.events.table), dtype=bool)
        coords = self.events.map_coord(self.mask.geom)
        return self.mask.get_by_coord(coords) == 1

    def info_dict(self):
        pass

    def stat_array(self):
        pass

    def peek(self, figsize=(16, 4)):
        """Quick-look summary plots.

        This method creates a figure displaying the elements of your `SpectrumDataset`.
        For example:

        * Exposure map
        * Energy dispersion matrix at the geometry center

        Parameters
        ----------
        figsize : tuple
            Size of the figure. Default is (16, 4).

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.set_title("Exposure")
        self.exposure.plot(ax1, ls="-", markersize=0, xerr=None)

        ax2.set_title("Energy Dispersion")

        if self.edisp_e_reco_binned is not None:
            kernel = self.edisp_e_reco_binned.get_edisp_kernel()
            kernel.plot_matrix(ax=ax2, add_cbar=True)

    @property
    def events_safe(self):
        return self.events.select_row_subset(self.mask_safe.data.astype(bool).flatten())

    @property
    def events_fit(self):
        return self.events.select_row_subset(self.mask_safe.data.astype(bool).flatten())

    @property
    def mask_image(self):
        """Reduced mask."""
        if self.mask is None:
            mask = Map.from_geom(self._geom.to_image(), dtype=bool)
            mask.data |= True
            return mask

        return self.mask.reduce_over_axes(func=np.logical_or)

    @classmethod
    def create(
        cls,
        geom,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=BINSZ_IRF_DEFAULT,
        reco_psf=False,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
        **kwargs,
    ):
        """Create an empty event dataset."""
        geoms = create_event_dataset_geoms(
            geom=geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            rad_axis=rad_axis,
            binsz_irf=binsz_irf,
            reco_psf=reco_psf,
        )
        kwargs.update(geoms)
        return cls.from_geoms(
            name=name, reference_time=reference_time, meta_table=meta_table, **kwargs
        )

    @classmethod
    def from_geoms(
        cls,
        geom,
        geom_exposure=None,
        geom_psf=None,
        geom_edisp=None,
        reference_time="2000-01-01",
        name=None,
        **kwargs,
    ):
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name

        if geom_exposure:
            kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")

        if geom_edisp:
            if "energy" in geom_edisp.axes.names:
                kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
            else:
                kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

        if geom_psf:
            if "energy_true" in geom_psf.axes.names:
                kwargs["psf"] = PSFMap.from_geom(geom_psf)
            elif "energy" in geom_psf.axes.names:
                kwargs["psf"] = RecoPSFMap.from_geom(geom_psf)

        kwargs.setdefault(
            "gti", GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        )
        kwargs["mask_safe"] = Map.from_geom(geom, unit="", dtype=bool)
        return cls(geom=geom, **kwargs)

    def npred(self):
        """Total predicted source and background counts.

        Returns
        -------
        npred : `Map`
            Total predicted counts.
        """
        npred_total = self.npred_signal()

        npred_total += self.npred_background()
        npred_total.data[npred_total.data < 0.0] = 0
        return npred_total

    def npred_background(self):
        raise NotImplementedError(
            "The method npred_background() is not implemented for EventDataset."
        )

    def npred_signal(self, model_names=None, stack=True):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        Parameters
        ----------
        model_names : list of str
            List of name of  SkyModel for which to compute the npred.
            If none, all the SkyModel predicted counts are computed.
        stack : bool
            Whether to stack the npred maps upon each other.

        Returns
        -------
        npred_sig : `gammapy.maps.Map`
            Map of the predicted signal counts.
        """
        npred_total = Map.from_geom(self._geom.squash("energy"), dtype=float)

        evaluators = self.evaluators
        if model_names is not None:
            if isinstance(model_names, str):
                model_names = [model_names]
            evaluators = {name: self.evaluators[name] for name in model_names}

        npred_list = []
        labels = []
        for evaluator_name, evaluator in evaluators.items():
            if evaluator.needs_update:
                evaluator.update(
                    self.exposure,
                    self.psf,
                    self.edisp,
                    self._geom,
                    self.mask_image,
                )
            if evaluator.contributes:
                npred = evaluator.compute_npred()
                shape_expected = list(npred.data.shape)
                shape_expected[npred.geom.axes.index_data("energy")] = -1
                npred.data = npred.data.reshape(shape_expected)
                bin_width = npred.geom.axes["energy"].bin_width.value.reshape(
                    shape_expected
                )
                data = integrate_histogram(
                    npred.data / bin_width,
                    npred.geom.axes["energy"].edges.value,
                    self.events_safe.energy.value.min(),
                    self.events_safe.energy.value.max(),
                )
                npred = Map.from_geom(npred.geom.squash("energy"), data=data, unit="")

                if stack:
                    npred_total.stack(npred)
                else:
                    npred_geom = Map.from_geom(self._geom, dtype=float)
                    npred_geom.stack(npred)
                    labels.append(evaluator_name)
                    npred_list.append(npred_geom)
                if not USE_NPRED_CACHE:
                    evaluator.reset_cache_properties()

        if npred_list != []:
            label_axis = LabelMapAxis(labels=labels, name="models")
            npred_total = Map.from_stack(npred_list, axis=label_axis)

        return npred_total


def create_event_dataset_geoms(
    geom,
    energy_axis_true=None,
    migra_axis=None,
    rad_axis=None,
    binsz_irf=BINSZ_IRF_DEFAULT,
    reco_psf=False,
):
    """Create geometries needed for event dataset.
    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom` or `~gammapy.maps.RegionGeom`
        Reference geometry.
    energy_axis_true : `~gammapy.maps.MapAxis`
        True energy axis.
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis.
    rad_axis : `~gammapy.maps.MapAxis`
        Offset axis.
    binsz_irf : float
        Bin size for IRF maps in deg.
    reco_psf : bool
        Use reconstructed energy axis for PSF map.
    Returns
    -------
    dict
        Dictionary of geometries.
    """
    rad_axis = rad_axis or RAD_AXIS_DEFAULT
    migra_axis = migra_axis or MIGRA_AXIS_DEFAULT

    if energy_axis_true is not None:
        energy_axis_true.assert_name("energy_true")
    else:
        energy_axis_true = geom.axes["energy_true"].copy(name="energy_true")

    external_axes = geom.axes.drop("energy_true")
    geom_image = geom.to_image()
    geom_exposure = geom_image.to_cube(MapAxes([energy_axis_true]) + external_axes)
    geom_irf = geom_image.to_binsz(binsz=binsz_irf)

    if reco_psf:
        raise NotImplementedError(
            "PSF map with reco energy axis not implemented yet for event dataset."
        )
    geom_psf = geom_irf.to_cube(MapAxes([rad_axis, energy_axis_true]) + external_axes)
    geom_edisp = geom_irf.to_cube(
        MapAxes([migra_axis, energy_axis_true]) + external_axes
    )
    return {
        "geom": geom,
        "geom_exposure": geom_exposure,
        "geom_psf": geom_psf,
        "geom_edisp": geom_edisp,
    }


class EventDatasetOnOff(EventDataset):
    """Event dataset for on-off analysis."""

    def __init__(
        self,
        events=None,
        events_off=None,
        acceptance=None,
        acceptance_off=None,
        stat_type="unbinned_onoff",
        *args,
        **kwargs,
    ):
        super().__init__(events=events, *args, **kwargs)
        self.events_off = events_off
        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self.stat_type = stat_type
        self.background_model = NormBackgroundSpectralModel(
            energy_events=self.events_off.energy,
            alpha=self.alpha.data.flatten(),
        )

        bkg_model = FoVBackgroundModel(
            dataset_name=self.name, spectral_model=self.background_model
        )
        if self.models is not None:
            _models = Models(self.models)
            _models.append(bkg_model)
        else:
            _models = [bkg_model]
        self.models = DatasetModels(_models)

    @classmethod
    def from_eventdataset(
        cls,
        dataset,
        acceptance,
        acceptance_off,
        events_off,
        name,
    ):
        if np.isscalar(acceptance):
            acceptance = Map.from_geom(dataset._geom, data=acceptance)
        if np.isscalar(acceptance_off):
            acceptance_off = Map.from_geom(dataset._geom, data=acceptance_off)

        return cls(
            events=dataset.events,
            geom=dataset.geom,
            geom_normalization=dataset.geom_normalization,
            models=dataset.models,
            exposure=dataset.exposure,
            psf=dataset.psf,
            edisp=dataset.edisp,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            meta_table=dataset.meta_table,
            name=name,
            reference_time=dataset.reference_time,
            gti=dataset.gti,
            meta=dataset.meta,
            edisp_e_reco_binned=dataset.edisp_e_reco_binned,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            events_off=events_off,
        )

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions.

        See :ref:`wstat`.

        Returns
        -------
        alpha : `Map`
            Alpha map.
        """
        # WARNING : ALPHA IS NOT BINNED AND WE HAVE TO CHANGE THAT
        # log.warning("Alpha is not binned, this should be changed.")
        with np.errstate(invalid="ignore", divide="ignore"):
            data = self.acceptance.quantity / self.acceptance_off.quantity
        data = np.nan_to_num(data)

        return Map.from_geom(self.acceptance.geom, data=data.to_value(""), unit="")

    def npred_background(self):
        """Predicted background total count from the background model interpolated on off counts.

        See :ref:`wstat`.

        Returns
        -------
        npred_background : `Map`
            Predicted background counts.
        """
        # x = np.logspace(
        #    np.log10(self.events_safe.energy.min().value),
        #    np.log10(self.events_safe.energy.max().value),
        #    100
        # ) * self.events_safe.energy.unit
        # mu_bkg = simpson(
        #    self.background_model(x),
        #    x
        # )
        mu_bkg = self.background_model.integral(
            energy_min=self.events_safe.energy.min(),
            energy_max=self.events_safe.energy.max(),
        )
        mu_bkg = np.nan_to_num(mu_bkg)
        return Map.from_geom(
            geom=self.geom.squash("energy"), data=mu_bkg.value, unit=mu_bkg.unit
        )

    @property
    def background(self):
        """Computed as alpha * n_off.

        See :ref:`wstat`.

        Returns
        -------
        background : `Map`
            Background map.
        """
        if self.background_model is None:
            return None
        background = self.background_model(self.events.energy)
        return Map.from_geom(self._geom, data=background.value, unit=background.unit)

    def signal_pdf(
        self, energy_min=None, energy_max=None, return_normalization_factor=False
    ):
        """Signal PDF evaluated at the event energies.

        Returns
        -------
        signal_pdf : `Map`
            Signal PDF map.
        """
        if not self.evaluators:
            raise ValueError("No model defined.")
        signal_pdf = None
        if return_normalization_factor:
            normalization_factor = []
        n = 0
        for evaluator in self.evaluators.values():
            if evaluator.needs_update:
                evaluator.update(
                    self.exposure,
                    self.psf,
                    self.edisp,
                    self._geom,
                    self.mask_image,
                )

            if evaluator.contributes:
                n += 1
                _output = evaluator.compute_signal_pdf(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    return_normalization_factor=return_normalization_factor,
                )
                if signal_pdf is None:
                    if isinstance(_output, tuple):
                        signal_pdf, normalization_factor = _output
                    else:
                        signal_pdf = _output
                else:
                    if isinstance(_output, tuple):
                        signal_pdf += _output[0]
                        normalization_factor.append(_output[1])
                    else:
                        signal_pdf += _output
                if not USE_NPRED_CACHE:
                    evaluator.reset_cache_properties()
        output = Map.from_geom(
            self._geom, data=signal_pdf.data / n, unit=signal_pdf.unit
        )
        if return_normalization_factor:
            output = (output, normalization_factor)
        return output

    def background_pdf(
        self, energy_min=None, energy_max=None, return_normalization_factor=False
    ):
        """Compute background probability of the model over the map energy range.

        Returns
        -------
        prob : `~numpy.ndarray`
            Background probability of the model.
        """
        energy_reco_axis = self.edisp.axes[
            "energy"
        ]  # extremely important to stay consistent with the normalization computed for signal_pdf, the integration is done between energy_reco limits that are the same as the ones used in the
        data = self.background_model(energy_reco_axis.center)
        flux = Map.from_geom(geom=self.geom, data=data.value, unit=data.unit)

        # energy_for_normalization = np.logspace(
        #    np.log10(energy_reco_axis.center.min().value) if energy_min is None else np.log10(energy_min.to(energy_reco_axis.unit).value),
        #    np.log10(energy_reco_axis.center.max().value) if energy_max is None else np.log10(energy_max.to(energy_reco_axis.unit).value),
        #    num=100,
        # ) * energy_reco_axis.unit
        # normalization_factor = simpson(
        #    self.background_model(energy_for_normalization), energy_for_normalization
        # )
        normalization_factor = self.background_model.integral(
            energy_min=energy_reco_axis.center.min()
            if energy_min is None
            else energy_min,
            energy_max=energy_reco_axis.center.max()
            if energy_max is None
            else energy_max,
        )
        pdf = flux / normalization_factor
        # pdf.data = np.clip(pdf.data, 0, None) #to avoid negative pdf values
        if return_normalization_factor:
            return pdf, normalization_factor
        else:
            return pdf

    @property
    def events_off_safe(self):
        mask = np.logical_and(
            self.events_off.energy >= self.events_safe.energy.min(),
            self.events_off.energy <= self.events_safe.energy.max(),
        )
        return self.events_off.select_row_subset(mask)

    def plot_results(self, figsize=(10, 6)):
        """Plot the results of the fit."""
        names = []
        evaluators = []
        for item in self.evaluators.items():
            names.append(item[0])
            evaluators.append(item[1])
        ncols = len(names)
        fig, axs = plt.subplots(ncols, 1, figsize=figsize)
        axs = np.asarray(axs).flatten()
        for i, (name, _evaluator) in enumerate(zip(names, evaluators)):
            ax = axs[i]
            ax.set_title(f"Model : {name}")
            flux = _evaluator.compute_flux()
            flux = _evaluator.apply_exposure(flux)
            flux = _evaluator.apply_edisp(flux, edisp=_evaluator.edisp_e_reco_binned)

            flux.data = flux.data / flux.geom.axes["energy"].bin_width.value.reshape(
                -1, 1, 1
            )
            ax = flux.plot_hist(ax=ax, label="Unbinned fit")

            hist, _ = np.histogram(
                self.events.energy, bins=flux.geom.axes["energy"].edges
            )
            ax.plot(
                flux.geom.axes["energy"].center.value,
                hist / flux.geom.axes["energy"].bin_width,
                lw=2,
                label="On events for unbinned",
                drawstyle="steps-mid",
            )

            hist, _ = np.histogram(
                self.events_off.energy,
                bins=flux.geom.axes["energy"].edges,
                weights=self.alpha.data.flatten() if self.alpha is not None else None,
            )
            ax.plot(
                flux.geom.axes["energy"].center.value,
                hist / flux.geom.axes["energy"].bin_width,
                drawstyle="steps-mid",
                lw=2,
                label="Off events for unbinned",
            )

            ax.plot(
                flux.geom.axes["energy"].center.value,
                1
                / self.background_model.norm.value
                * self.background_model(flux.geom.axes["energy"].center),
                label="bkg model initial for unbinned spline",
                ls="--",
            )

            ax.plot(
                flux.geom.axes["energy"].center.value,
                self.background_model(flux.geom.axes["energy"].center),
                label="bkg model fitted for unbinned spline",
                ls="-.",
            )

            # ax.set_xlim(0.01, 1e2)
            # ax.set_ylim(1e-2, 1e6)
            ax.legend()
        fig.tight_layout()
        return fig, ax
