from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time

__all__ = ["LightCurve", "LightCurveEstimator"]


class LightCurve:
    """Lightcurve container.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with lightcurve data.
    """

    def __init__(self, table, **kwargs):
        self.table = table


    def __repr__(self):
        return "{}(len={})".format(self.__class__.__name__, len(self.table))


    @property
    def time_bin_end(self):
        return self.table.time_bin_end


    @property
    def time_bin_start(self):
        return self.table.time_bin_start


    @property
    def time_bin_center(self):
        return self.time_bin_end - self.time_bin_start


    def chisq(self, col_name, ignore_ul=True):
        """Chi square test for variability.
        :param col_name: The column to test for variability
        :param ignore_ul: bool, optional
            Ignore the upper limit value
        :return: The chisq of a straight line fit and the pval
        """
        value = self.table[col_name].value
        if ignore_ul:
            value = value[self.table["is_ul"]]

        chi2, pval = stats.chisquare(value, np.mean(value))
        return chi2, pval


    def fvar(self, col_name, ignore_ul=False):
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

        n_points = np.count_nonzero(~np.isnan(self[col_name]))

        s_square = np.nansum((value - value_mean) ** 2) / (n_points - 1)
        sig_square = np.sum(value_err ** 2) / n_points
        fvar = np.sqrt(np.abs(s_square - sig_square)) / value_mean

        sigxserr_a = np.sqrt(2 / n_points) * (sig_square / value_mean) ** 2
        sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / value_mean)
        sigxserr = np.sqrt(sigxserr_a ** 2 + sigxserr_b ** 2)
        fvar_err = sigxserr / (2 * fvar)

        return fvar, fvar_err

    def plot(self, col_name, ax=None, **kwargs):
        col_err = col_name + "_err"
        y = self[col_name].value
        y_err = self[col_err].value
        # optional y_err asymmetric
        x = self.time_bin_center.mjd
        x_low = x - self.time_bin_start.mjd
        x_high = self.time_bin_end.mjd - x

        if ax is None:
            ax = plt.gca()

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "None")

        ax.errorbar(x=x, y=y, xerr=[x_low, x_high], yerr=y_err, **kwargs)

        return ax





class LightCurveEstimator:
    """Flux Points estimated for each time bin.

    Estimates flux points for a given list of datasets.


    Parameters
    ----------
    datasets : list of `~gammapy.spectrum.SpectrumDatatset` or `~gammapy.cube.MapDatatset`
        Spectrum or Map datasets.
    compute_ul: bool, optional
        Compute upper limits in case of sigma < sigma_ul. True by default
    sigma_ul : int, optional
        Sigma to use for upper limit computation. Default value 3.0


    """

    def __init__(self, datasets, compute_ul=True, sigma_ul=3):
        self.datasets = datasets
        self.compute_ul = compute_ul
        self.sigma_ul = sigma_ul



    def _get_free_parameters(self, dataset):
        """Get the free parameters in the model"""
        i = 0
        indices = []
        names = []
        for apar in dataset.model.parameters.parameters:
            i = i + 1
            if apar.frozen == False:
                indices.append(i)
                names.append(i)
        return indices, names


    def get_times(self, dataset):
        """Return the times present in the counts meta"""
        t_start = dataset.counts.meta["t_start"]
        t_stop = dataset.counts.meta["t_start"]
        return t_start, t_stop


    def get_lc(self):
        """

        :param dataset:
        :return:
        """
        rows = []
        for dataset in datasets:
            t_start, t_stop = self.get_times(dataset)
            fit = Fit(dataset)
            result = fit.run()
            ts = compute_ts(dataset)
            free_indices, free_names = self._get_free_parameters(dataset)

            rows.append()


    def compute_ts(self, dataset):
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

        # compute sqrt TS
        ts = np.abs(loglike_null - loglike)
        return ts




