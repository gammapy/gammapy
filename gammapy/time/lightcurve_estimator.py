# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.table import QTable, Column
from astropy.time import Time
from scipy.interpolate import interp1d
from scipy.stats import chisquare
from ..utils.fitting import Fit
from ..utils.interpolation import interpolate_likelihood_profile
from ..time import LightCurve

__all__ = [ "LightCurveEstimator3D"]




class LightCurveEstimator3D:
    """Flux Points estimated for each time bin.

    Estimates flux points for a given list of datasets.

    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDatatset` or `~gammapy.cube.MapDataset`
        Spectrum or Map datasets.
    reoptimize : bool
        reoptimize other parameters during likelihod scan
    return_scan : bool
        return the likelihood scan profile for the amplitude parameter
    min_ts : float
        minimum TS to decide as upper limit. Default 9.0
    """

    def __init__(self, datasets,  reoptimize=False, return_scan=True, min_ts=9.0):
        self.datasets = datasets

        if not datasets.is_all_same_type and datasets.is_all_same_shape:
            raise ValueError(
                "Flux point estimation requires a list of datasets"
                " of the same type and data shape."
            )

        self.reoptimize = reoptimize
        self.return_scan = return_scan
        self.min_ts = min_ts



    def _get_free_parameters(self):
        """Get the inidces of free parameters in the model"""
        dataset = self.datasets[0] #should not be fixed at 0
        i = 0
        indices = []
        for apar in dataset.model.parameters.parameters:
            if apar.frozen == False:
                indices.append(i)
            i = i + 1
        return indices


    def t_start(self):
        """Return the start times present in the counts meta"""
        t_start = []
        for dataset in self.datasets:
            t_start.append(dataset.counts.meta["t_start"])
        return t_start


    def t_stop(self):
        """Return the stop times present in the counts meta"""
        t_stop = []
        for dataset in self.datasets:
            t_stop.append(dataset.counts.meta["t_stop"])
        return t_stop


    def make_names(self):
        row = []
        indices = self._get_free_parameters()
        for ind in indices:
            name = self.datasets[0].model.parameters.parameters[ind].name
            err = name + "_err"
            row = row + [name, err]
            if self.return_scan:
                row = row + [name + "_likelihood_profile"]
        return row

    def set_units(self, lc):
        #Useless fuction because code badly written
        indices = self._get_free_parameters()
        for ind in indices:
            name = self.datasets[0].model.parameters.parameters[ind].name
            err = name + "_err"
            lc[name].unit = self.datasets[0].model.parameters.parameters[ind].unit
            lc[err].unit = self.datasets[0].model.parameters.parameters[ind].unit
        return lc


    def run(self, bounds=6):
        """Run light Curve extraction"""
        rows = []
        indices = self._get_free_parameters()
        col_names =  self.make_names()

        for dataset in self.datasets:
            fit = Fit(dataset)
            result = fit.run()
            pars = []
            for i in indices:
                pars.append(result.parameters.parameters[i].value)
                pars.append(result.parameters.error(i))
                if self.return_scan:
                    scan = self.estimate_likelihood_scan(fit, result.parameters.parameters[i].name, bounds)
                    pars = pars + [scan]
            rows.append(pars)

        lc = QTable(rows=rows, names=col_names)
        t_start = Column(data=self.t_start(), name='time_min')
        t_stop = Column(data=self.t_stop(), name='time_max')
        lc.add_columns([t_start,t_stop], [0,0])
        ts = self.compute_ts()
        is_ul = np.array(ts) < self.min_ts
        lc["TS"] = ts
        lc["is_ul"] = is_ul
        lc = self.set_units(lc)
        return LightCurve(lc)


    def estimate_likelihood_scan(self, fit, par_name="amplitude", bounds=6):
        """Estimate likelihood profile for the amplitude parameter.

        Returns
        -------
        result : dict
            Dict with norm_scan and dloglike_scan for the flux point.
        """
        result = fit.likelihood_profile(
            par_name, bounds=bounds, reoptimize=self.reoptimize, nvalues=31,
        )
        dloglike_scan = result["likelihood"]

        return {par_name+"_scan": result["values"], "dloglike_scan": dloglike_scan}


    def compute_ts(self):
        ts = []
        for dataset in self.datasets:
            loglike = dataset.likelihood()

            ds = dataset.copy()
            # Assuming the first the model corresponds to the source.
            # Finding the TS for this model only
            for apar in ds.model.parameters.parameters:
                if apar.name == "amplitude":
                    apar.value = 0.0
                    apar.frozen = True
                    break

            loglike_null = ds.likelihood()

            # compute TS
            ts.append(np.abs(loglike_null - loglike))
        return ts

    def estimate_time_bin_flux(self, datasets, steps="all"):
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

        result = OrderedDict(
            [
                ("e_ref", self.e_ref),
                ("e_min", self.e_min),
                ("e_max", self.e_max),
                ("ref_dnde", self.ref_model(self.e_ref)),
                ("ref_flux", self.ref_model.integral(self.e_min, self.e_max)),
                ("ref_eflux", self.ref_model.energy_flux(self.e_min, self.e_max)),
                ("ref_e2dnde", self.ref_model(self.e_ref) * self.e_ref ** 2),
            ]
        )

        result.update(self.estimate_norm())

        if not result.pop("success"):
            log.warning(
                    "Fit failed for time bin between {t_min} and {t_max},"
                    " setting NaN.".format(t_min=tmin, t_max=tmax)
            )

        if steps == "all":
            steps = ["err", "counts", "errp-errn", "ul", "ts", "norm-scan"]

        if "err" in steps:
            result.update(self.estimate_norm_err())

        if "counts" in steps:
            result.update(self.estimate_counts())

        if "errp-errn" in steps:
            result.update(self.estimate_norm_errn_errp())

        if "ul" in steps:
            result.update(self.estimate_norm_ul())

        if "ts" in steps:
            result.update(self.estimate_norm_ts())

        if "norm-scan" in steps:
            result.update(self.estimate_norm_scan())

        return result


#TODO : the following code is copied from FluxPointsEstimator, can it be restructured?
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

    def estimate_counts(self):
        """Estimate counts for the flux point.

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with counts for the flux point.
        """
        counts = []

        for dataset in self.datasets.datasets:
            mask = dataset.mask_fit
            if dataset.mask_safe is not None:
                mask &= dataset.mask_safe

            counts.append(dataset.counts.data[mask].sum())

        return {"counts": np.array(counts, dtype=int)}

    def estimate_norm_ul(self):
        """Estimate upper limit for a flux point.

        Returns
        -------
        result : dict
            Dict with upper limit for the flux point norm.
        """
        norm = self.model.norm

        # TODO: the minuit backend has convergence problems when the likelihood is not
        #  of parabolic shape, which is the case, when there are zero counts in the
        #  energy bin. For this case we change to the scipy backend.
        counts = self.estimate_counts()["counts"]

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
