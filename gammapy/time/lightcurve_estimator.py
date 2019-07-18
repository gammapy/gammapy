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
