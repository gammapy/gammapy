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
                func = interp1d([l_profile['dloglike_scan'][arr[-1]], l_profile['dloglike_scan'][arr[-1]+1]],
                    [l_profile["amplitude_scan"][arr[-1]], l_profile["amplitude_scan"][arr[-1]+1]])
                ul.append(func(lim))
            else:
                ul.append(np.nan)
        self.table[col_name+"_upper_limits_"]=ul


    def compute_asymmetric_errors(self, col_name="amplitude", sigma=1):
        """
        :param col_name="amplitude",
        :param sigma: the confidence limit to compute errors
        :return: the lower and upper limits of the parameter
        """

        y_low = []
        y_high = []
        for arow in self.table:
            l_profile = arow[col_name+'_likelihood_profile']
            lim = 0.5 + np.min(l_profile['dloglike_scan'])
            arr = np.where(l_profile['dloglike_scan'] < lim)
            func1 = interp1d([l_profile['dloglike_scan'][arr[-1]], l_profile['dloglike_scan'][arr[-1]+1]],
                             [l_profile["amplitude_scan"][arr[-1]], l_profile["amplitude_scan"][arr[-1]+1]])
            y_high.append(func1(lim))

            func2 = interp1d([l_profile['dloglike_scan'][arr[0]], l_profile['dloglike_scan'][arr[0]-1]],
                        [l_profile["amplitude_scan"][arr[0]], l_profile["amplitude_scan"][arr[0]-1]])
            y_low.append(func2(lim))

        self.table[col_name+"_asymmetric_error"] = [y_low,y_high]
        return y_low, y_high


    def rebin(self, nbins, min_sig):
        """To group the bins using the likelihood profile scan

        Parameters:
        -------
        nbins: int
            The maximum number of bins to group
        min_sig: float
            The min sig to achieve for each group
        Either of the 2 must be specified

        Returns a new binned table
        """

        pass


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

        pass


    def plot(self, col_name, error_asymmetric = False, plot_upperlimit=False, ax=None, **kwargs):
        """

        :param col_name: The parameter to be plotted
        :param error_asymmetric: bool,
            If the error on the parameter is asymmetric.
            If True, asymmetric errors will be computed from the likelihood profile
        :param ax: matplotlib axes
        :param kwargs: matplotlib kwargs
        :return:
        """
        import matplotlib.pyplot as plt

        col_err = col_name + "_err"
        y = self.table[col_name].value

        if error_asymmetric:
            colas = col_name+"_asymmetric_error"
            if colas.issubset(self.table.columns): #check if the values are already computed
                y_low, y_high = self.table[colas]
            else:
                y_low, y_high = self.compute_asymmetric_errors()
                y_low = np.subtract(y, y_low)
                y_high = np.subtract(y_high, y)

        else:
            y_err = self.table[col_err].value
            y_low = y_err
            y_high = y_err

        x = [_.mjd for _ in self.time_bin_center]
        x_low = np.subtract(x, [_.mjd for _ in self.time_bin_start])
        x_high = np.subtract([_.mjd for _ in self.time_bin_end], x)

        is_ul = self.table['is_ul']
        if plot_upperlimit:
            col = col_name+"_upper_limits"
            if colas.issubset(self.table.columns):  # check if the values are already computed
                y[is_ul] = self.table[col][is_ul]
            else:
                raise ValueError("Upper limits not computed")

        if ax is None:
            ax = plt.gca()

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "None")

        ax.errorbar(x=x, y=y, xerr=[x_low, x_high], yerr=[y_low, y_high], uplims=is_ul, **kwargs)

        return ax


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






