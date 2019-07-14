from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.table import QTable, Column
from astropy.time import Time
from scipy.interpolate import interp1d
from scipy.stats import chisquare
from ..utils.fitting import Fit
from ..utils.interpolation import interpolate_likelihood_profile

__all__ = ["LightCurve", "LightCurveEstimator"]


class LightCurve:
    """Lightcurve container.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with lightcurve data.
    """

    def __init__(self, table):
        self.table = table


    def __repr__(self):
        return "{}(len={})".format(self.__class__.__name__, len(self.table))


    @property
    def time_bin_start(self):
        return self.table['time_bin_start']


    @property
    def time_bin_end(self):
        return self.table['time_bin_end']


    @property
    def time_bin_center(self):
        centers = []
        for i in range(len(self.table)):
            t = (self.time_bin_start[i].mjd + self.time_bin_end[i].mjd)/2.0
            centers.append(Time(t, format='mjd'))
        return centers


    def chisq(self, col_name='amplitude', ignore_ul=True):
        """Chi square test for variability.
        :param col_name: The column to test for variability
        :param ignore_ul: bool, optional
            Ignore the upper limit values while doing computation
        :return: The chisq of a straight line fit and the pval
        """
        value = self.table[col_name].value
        if ignore_ul:
            value = value[~self.table["is_ul"]]
        chi2, pval = chisquare(value, np.mean(value))
        return chi2, pval


    def fvar(self, col_name='amplitude', ignore_ul=False):
        """Compute fractional variability
         The fractional excess variance :math:`F_{var}`, an intrinsic
        variability estimator, is given by

        .. math::
            F_{var} = \sqrt{\frac{S^{2} - \bar{\sigma^{2}}}{\bar{x}^{2}}}.

        It is the excess variance after accounting for the measurement errors
        on the light curve :math:`\sigma`. :math:`S` is the variance.

        Parameters
        ---------
        col_name: The column for which to compute fvar
        ignore_ul: bool, optional
            Ignore the upper limit values

        Returns
        -------
        fvar, fvar_err : `~numpy.ndarray`
            Fractional excess variance.

        References
        ----------
        .. [Vaughan2003] "On characterizing the variability properties of X-ray light
           curves from active galaxies", Vaughan et al. (2003)
           https://ui.adsabs.harvard.edu/abs/2003MNRAS.345.1271V
        """


        col_err = col_name + "_err"
        value = self.table[col_name].value
        value_err = self.table[col_err].value

        if ignore_ul:
            value = value[self.table["is_ul"]]
            value_err = value_err[self.table["is_ul"]]


        value_mean = np.mean(value)
        n_points = np.count_nonzero(~self.table["is_ul"])

        s_square = np.nansum((value - value_mean) ** 2) / (n_points - 1)
        sig_square = np.sum(value_err ** 2) / n_points
        fvar = np.sqrt(np.abs(s_square - sig_square)) / value_mean

        sigxserr_a = np.sqrt(2 / n_points) * (sig_square / value_mean) ** 2
        sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / value_mean)
        sigxserr = np.sqrt(sigxserr_a ** 2 + sigxserr_b ** 2)
        fvar_err = sigxserr / (2 * fvar)

        return fvar, fvar_err


    def compute_ul(self, col_name="amplitude", sigma_limit=5):
        """
        To compute upper limits from the likelihood scans.
        Adds a column on the Table
        """
        ul = []
        for arow in self.table:
            l_profile = arow[col_name+'_likelihood_profile']
            if arow['is_ul']:
                lim = sigma_limit * 0.5 + np.min(l_profile['dloglike_scan'])
                arr = np.where(l_profile['dloglike_scan'] < lim)
                func = interp1d(l_profile['dloglike_scan'][arr[-1], arr[-1]+1],
                                l_profile["amplitude_scan"][arr[-1], arr[-1]+1])
                ul.append(func(lim))
            else:
                ul.append(np.nan)
        self.table[str(sigma_limit)+"_upper_limits"]=ul


    def plot(self, col_name, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if col_name == "likelihood_profile":
            self.plot_likelihood()
            return

        col_err = col_name + "_err"
        #TODO : implement the plotting upper limits
        y = self.table[col_name].value
        y_err = self.table[col_err].value
        # optional y_err asymmetric
        x = [_.mjd for _ in self.time_bin_center]
        x_low = np.subtract(x, [_.mjd for _ in self.time_bin_start])
        x_high = np.subtract([_.mjd  for _ in self.time_bin_end], x)

        if ax is None:
            ax = plt.gca()

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "None")

        ax.errorbar(x=x, y=y, xerr=[x_low, x_high], yerr=y_err, **kwargs)

        return ax

    def plot_likelihood(self, col_name='amplitude', ax=None, add_cbar=True):
        """To plot the likelihood profile as a density plot

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axis object to plot on.
        add_cbar : bool
            Whether to add a colorbar to the plot.
        kwargs : dict
            Keyword arguments passed to :func:`matplotlib.pyplot.pcolormesh`
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        y_unit = self.table[col_name].unit
        y_values = self.table[col_name+'_likelihood_profile'][0]['dloglike_scan']

        x = self.time_bin_start

        # Compute likelihood "image" one time bin at a time

        z = np.empty((len(self.table), len(y_values)))
        for idx, row in enumerate(self.table):
            y_ref = self.table["ref_" + self.sed_type].quantity[idx]
            norm = (y_values / y_ref).to_value("")
            norm_scan = row[col_name+'_scan']
            loglike_min = np.min(row["dloglike_scan"])
            dloglike_scan = row["dloglike_scan"] - loglike_min
            interp = interpolate_likelihood_profile(norm_scan, dloglike_scan)
            z[idx] = interp((norm,))

        kwargs.setdefault("vmax", 0)
        kwargs.setdefault("vmin", -4)
        kwargs.setdefault("zorder", 0)
        kwargs.setdefault("cmap", "Blues")
        kwargs.setdefault("linewidths", 0)

        # clipped values are set to NaN so that they appear white on the plot
        z[-z < kwargs["vmin"]] = np.nan
        caxes = ax.pcolormesh(x, y_values, -z.T, **kwargs)
        ax.set_xscale("linear", nonposx="clip")
        ax.set_yscale("linear", nonposy="clip")
        ax.set_xlabel("Time ({})")
        ax.set_ylabel("{} ({})")

        if add_cbar:
            label = "delta log-likelihood"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax



    def _get_fluxes_and_errors(self, unit="cm-2 s-1"):
        """Extract fluxes and corresponding errors

        Helper function for the plot method.

        Parameters
        ----------
        unit : str, `~astropy.units.Unit`, optional
            Unit of the returned flux and errors values

        Returns
        -------
        y : `numpy.ndarray`
            Flux values
        (yn, yp) : tuple of `numpy.ndarray`
            Flux error values
        """
        y = self.table["flux"].quantity.to(unit)

        if all(k in self.table.colnames for k in ["flux_errp", "flux_errn"]):
            yp = self.table["flux_errp"].quantity.to(unit)
            yn = self.table["flux_errn"].quantity.to(unit)
        elif "flux_err" in self.table.colnames:
            yp = self.table["flux_err"].quantity.to(unit)
            yn = self.table["flux_err"].quantity.to(unit)
        else:
            yp, yn = np.zeros_like(y), np.zeros_like(y)

        return y.value, (yn.value, yp.value)

    def _get_uls(self, unit="cm-2 s-1"):
        """Extract upper limits for the given column

        Helper function for the plot method.

        Parameters
        ----------
        unit : str, `~astropy.units.Unit`, optional
            Unit of the returned flux upper limit values

        Returns
        -------
        is_ul : `numpy.ndarray`
            Is flux point is an upper limit? (boolean array)
        yul : `numpy.ndarray`
            Flux upper limit values
        """
        try:
            is_ul = self.table["is_ul"].data.astype("bool")
        except KeyError:
            is_ul = np.zeros_like(self.table["flux"]).data.astype("bool")

        if is_ul.any():
            yul = self.table["flux_ul"].quantity.to(unit)
        else:
            yul = np.zeros_like(self.table["flux"]).quantity
            yul[:] = np.nan

        return is_ul, yul.value



class LightCurveEstimator:
    """Flux Points estimated for each time bin.

    Estimates flux points for a given list of datasets.


    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDatatset` or `~gammapy.cube.MapDatatset`
        Spectrum or Map datasets.
    reoptimize : reoptimize other parameters during likelihod scan
    return_scan : return the likelihood scan profile for the amplitude parameter
    min_ts : minimum TS to decide as upper limit.


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


    def get_lc(self, bounds=6):

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
        t_start = Column(data=self.t_start(), name='time_bin_start')
        t_stop = Column(data=self.t_stop(), name='time_bin_end')
        lc.add_columns([t_start,t_stop], [0,0])
        ts = self.compute_ts()
        is_ul = np.array(ts) < self.min_ts
        lc["TS"] = ts
        lc["is_ul"] = is_ul
        lc = self.set_units(lc)
        #forcefully setting the units on the parameters. To be cleaned up!
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






