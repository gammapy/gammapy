# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import ScaleSpectralModel
from gammapy.spectrum import FluxPoints, SpectrumDatasetOnOff
from gammapy.time import LightCurve
from gammapy.utils.table import table_from_row_data

__all__ = ["LightCurveEstimator"]

log = logging.getLogger(__name__)


class LightCurveEstimator:
    """Estimate flux points for a given list of datasets, each per time bin.

    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDataset` or `~gammapy.cube.MapDataset`
        Spectrum or Map datasets.
    source : str
        For which source in the model to compute the flux points. Default is ''
    norm_min : float
        Minimum value for the norm used for the likelihood profile evaluation.
    norm_max : float
        Maximum value for the norm used for the likelihood profile evaluation.
    norm_n_values : int
        Number of norm values used for the likelihood profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the likelihood profile.
    sigma : int
        Sigma to use for asymmetric error computation.
    sigma_ul : int
        Sigma to use for upper limit computation.
    reoptimize : bool
        reoptimize other parameters during likelihod scan
    """

    def __init__(
        self,
        datasets,
        source="",
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        sigma=1,
        sigma_ul=2,
        reoptimize=False,
    ):

        if not isinstance(datasets, Datasets):
            datasets = Datasets(datasets)

        self.datasets = datasets

        if not datasets.is_all_same_type and datasets.is_all_same_shape:
            raise ValueError(
                "Light Curve estimation requires a list of datasets"
                " of the same type and data shape."
            )

        dataset = self.datasets[0]

        if isinstance(dataset, SpectrumDatasetOnOff):
            model = dataset.model
        else:
            model = dataset.model[source].spectral_model

        self.model = ScaleSpectralModel(model)
        self.model.norm.min = 0
        self.model.norm.max = 1e5

        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )

        self.norm_values = norm_values

        self.sigma = sigma
        self.sigma_ul = sigma_ul
        self.reoptimize = reoptimize
        self.source = source

        self._set_scale_model()

    def _set_scale_model(self):
        # set the model on all datasets
        for dataset in self.datasets:
            if isinstance(dataset, SpectrumDatasetOnOff):
                dataset.model = self.model
            else:
                dataset.model[self.source].spectral_model = self.model

    @property
    def ref_model(self):
        return self.model.model

    def run(self, e_ref, e_min, e_max, steps="all"):
        """Run light curve extraction.

        Normalize integral and energy flux between emin and emax.

        Parameters
        ----------
        e_ref : `~astropy.units.Quantity`
            reference energy of dnde flux normalization
        e_min : `~astropy.units.Quantity`
            minimum energy of integral and energy flux normalization interval
        e_max : `~astropy.units.Quantity`
            minimum energy of integral and energy flux normalization interval
        steps : list of str
            Which steps to execute. Available options are:

                * "err": estimate symmetric error.
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "ts": estimate ts and sqrt(ts) values.
                * "norm-scan": estimate likelihood profiles.

            By default all steps are executed.

        Returns
        -------
        lightcurve : `~gammapy.time.LightCurve`
            the Light Curve object
        """
        self.e_ref = e_ref
        self.e_min = e_min
        self.e_max = e_max

        rows = []
        for dataset in self.datasets:
            row = {
                "time_min": dataset.counts.meta["t_start"].mjd,
                "time_max": dataset.counts.meta["t_stop"].mjd,
            }
            row.update(self.estimate_time_bin_flux(dataset, steps))
            rows.append(row)

        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})
        table = FluxPoints(table).to_sed_type("flux").table
        return LightCurve(table)

    def estimate_time_bin_flux(self, dataset, steps="all"):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        steps : list of str
            Which steps to execute. Available options are:

                * "err": estimate symmetric error.
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "ts": estimate ts and sqrt(ts) values.
                * "norm-scan": estimate likelihood profiles.

            By default all steps are executed.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        self.fit = Fit(dataset)

        result = {
            "e_ref": self.e_ref,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "ref_dnde": self.ref_model(self.e_ref),
            "ref_flux": self.ref_model.integral(self.e_min, self.e_max),
            "ref_eflux": self.ref_model.energy_flux(self.e_min, self.e_max),
            "ref_e2dnde": self.ref_model(self.e_ref) * self.e_ref ** 2,
        }

        result.update(self.estimate_norm())

        if not result.pop("success"):
            log.warning(
                "Fit failed for time bin between {t_min} and {t_max},"
                " setting NaN.".format(
                    t_min=dataset.counts.meta["t_start"],
                    t_max=dataset.counts.meta["t_stop"],
                )
            )

        if steps == "all":
            steps = ["err", "counts", "errp-errn", "ul", "ts", "norm-scan"]

        if "err" in steps:
            result.update(self.estimate_norm_err())

        if "counts" in steps:
            result.update(self.estimate_counts(dataset))

        if "errp-errn" in steps:
            result.update(self.estimate_norm_errn_errp())

        if "ul" in steps:
            result.update(self.estimate_norm_ul(dataset))

        if "ts" in steps:
            result.update(self.estimate_norm_ts())

        if "norm-scan" in steps:
            result.update(self.estimate_norm_scan())

        return result

    # TODO : most of the following code is copied from FluxPointsEstimator, can it be restructured?
    def estimate_norm_errn_errp(self):
        """Estimate asymmetric errors for a flux point.

        Returns
        -------
        result : dict
            Dict with asymmetric errors for the flux point norm.
        """
        result = self.fit.confidence(parameter=self.model.norm, sigma=self.sigma)
        return {"norm_errp": result["errp"], "norm_errn": result["errn"]}

    def estimate_norm_err(self):
        """Estimate covariance errors for a flux point.

        Returns
        -------
        result : dict
            Dict with symmetric error for the flux point norm.
        """
        result = self.fit.covariance()
        norm_err = result.parameters.error(self.model.norm)
        return {"norm_err": norm_err}

    def estimate_counts(self, dataset):
        """Estimate counts for the flux point.

        Parameters
        ----------
        dataset : `~gammapy.modeling.Dataset`
            the dataset object

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with counts for the flux point.
        """
        # TODO : use e_min and e_max interval for counts calculation
        # TODO : add off counts and excess? for DatasetOnOff
        # TODO : this may require a loop once we support Datasets per time bin
        mask = dataset.mask
        if dataset.mask_safe is not None:
            mask &= dataset.mask_safe

        counts = dataset.counts.data[mask].sum()

        return {"counts": counts}

    def estimate_norm_ul(self, dataset):
        """Estimate upper limit for a flux point.

        Returns
        -------
        result : dict
            Dict with upper limit for the flux point norm.
        """
        norm = self.model.norm

        # TODO: the minuit backend has convergence problems when the likelihood is not
        #  of parabolic shape, which is the case, when there are zero counts in the
        #  bin. For this case we change to the scipy backend.
        counts = self.estimate_counts(dataset)["counts"]

        if np.all(counts == 0):
            result = self.fit.confidence(
                parameter=norm,
                sigma=self.sigma_ul,
                backend="scipy",
                reoptimize=self.reoptimize,
            )
        else:
            result = self.fit.confidence(parameter=norm, sigma=self.sigma_ul)

        return {"norm_ul": result["errp"] + norm.value}

    def estimate_norm_ts(self):
        """Estimate ts and sqrt(ts) for the flux point.

        Returns
        -------
        result : dict
            Dict with ts and sqrt(ts) for the flux point.
        """
        loglike = self.datasets.likelihood()

        # store best fit amplitude, set amplitude of fit model to zero
        self.model.norm.value = 0
        self.model.norm.frozen = True

        if self.reoptimize:
            _ = self.fit.optimize()

        loglike_null = self.datasets.likelihood()

        # compute sqrt TS
        ts = np.abs(loglike_null - loglike)
        sqrt_ts = np.sqrt(ts)
        return {"sqrt_ts": sqrt_ts, "ts": ts}

    def estimate_norm_scan(self):
        """Estimate likelihood profile for the norm parameter.

        Returns
        -------
        result : dict
            Dict with norm_scan and dloglike_scan for the flux point.
        """
        result = self.fit.likelihood_profile(
            self.model.norm, values=self.norm_values, reoptimize=self.reoptimize
        )
        dloglike_scan = result["likelihood"]
        return {"norm_scan": result["values"], "dloglike_scan": dloglike_scan}

    def estimate_norm(self):
        """Fit norm of the flux point.

        Returns
        -------
        result : dict
            Dict with "norm" and "loglike" for the flux point.
        """
        # start optimization with norm=1
        self.model.norm.value = 1.0
        self.model.norm.frozen = False

        result = self.fit.optimize()

        if result.success:
            norm = self.model.norm.value
        else:
            norm = np.nan

        return {"norm": norm, "loglike": result.total_stat, "success": result.success}
