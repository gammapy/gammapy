# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to store profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
import logging

__all__ = ["ImageProfile"]

log = logging.getLogger(__name__)


class ImageProfile:
    """Image profile class.

    The image profile data is stored in `~astropy.table.Table` object, with the
    following columns:

        * `x_ref` Center of the bin, given the distance to a reference position (required).
        * `x_min` Bin minimum value (optional).
        * `x_max` Bin maximum value (optional).
        * `energy_edge` Edges of the energy bands (required).
        * `counts` Counts profile data (required).
        * `background` Estimated background counts (optional).
        * `excess` Excess counts (required).
        * `alpha` Bin acceptance_on/acceptance_off ratio (optional).
        * `ts` Bin statistics (optional).
        * `sqrt_ts` square-root of TS (optional).
        * `err` Excess profile data error (optional).
        * `errn` Excess profile data lower error (optional).
        * `errp` Excess profile data upper error (optional).
        * `ul` Excess profile data upper limit (optional).
        * `flux` Flux within the energy range (required).
        * `flux_err` Flux error (optional).
        * `flux_errn` Flux lower error (optional).
        * `flux_errp` Flux upper error (optional).
        * `flux_ul` Flux upper limit (optional).
        * `solid_angle` Region solid angle (required).

    The stored metadata are:

        * `PROFILE_TYPE` orthogonal_rectangle
        * `SPECTRAL_MODEL` Spectral Model used to compute fluxes

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table instance with the columns specified as above.
    """

    def __init__(self, table):
        self.table = table

    def profile(self, method='counts', energy_band=None):
        """Image profile quantity.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
           Compute counts, excess, fluxes or the brightness within profile bins.
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used

        Returns
        -------
        value: `astropy.units.quantity.Quantity`
            Data computed for the required method of the image profile

        """
        try:
            if method == 'brightness':
                y = self.table['flux'].quantity / self.table['solid_angle'].quantity
            else:
                y = self.table[method].quantity
                # if method != 'excess' and self.table["excess"].unit == '':
                #     fact = fact.value
            mask = self._get_energy_mask(energy_band)
            yy = np.sum(y, axis=1, where=mask, keepdims=True)
            yy.name = method
            return yy
        except AttributeError:
            raise AttributeError(f"The method [{method}] is not supported")

    def profile_err(self, method='counts', energy_band=None):
        """Mean error quantity of the image profile

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
           Compute the mean error of counts, excess, fluxes or the brightness within profile bins.
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        mask = self._get_energy_mask(energy_band)

        try:
            if method == 'counts':
                y = np.sum(self.table['counts'].quantity, axis=1, where=mask, keepdims=True)
                return np.sqrt(y)
            elif method == 'excess':
                if "errn" in self.table.colnames and "errp" in self.table.colnames:
                    ymin = self._quadratic_sum(self.table["errn"].quantity, axis=1, where=mask, keepdims=True)
                    ymax = self._quadratic_sum(self.table["errp"].quantity, axis=1, where=mask, keepdims=True)
                elif "err" in self.table.colnames:
                    ymin = ymax = self._quadratic_sum(self.table["err"].quantity, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return (ymin+ymax)/2.
            else:
                fact = 1.
                if method == 'brightness':
                    fact = 1. / self.table['solid_angle'].quantity[:, 0]
                # if method != 'excess' and self.table["excess"].unit == '':
                #     fact = fact.value
                if "flux_errn" in self.table.colnames and "flux_errp" in self.table.colnames:
                    ymin = self._quadratic_sum(self.table["flux_errn"].quantity*fact, axis=1, where=mask, keepdims=True)
                    ymax = self._quadratic_sum(self.table["flux_errp"].quantity*fact, axis=1, where=mask, keepdims=True)
                elif "flux_err" in self.table.colnames:
                    ymin = ymax = \
                        self._quadratic_sum(self.table["flux_err"].quantity*fact, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return (ymin+ymax)/2.
        except KeyError:
            return None

    def profile_err_p(self, method='counts', energy_band=None):
        """Positive error quantity of the image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
           Compute the positive error of counts, excess, fluxes or the brightness within profile bins.
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        mask = self._get_energy_mask(energy_band)

        try:
            if method == 'counts':
                y = np.sum(self.table['counts'].quantity, axis=1, where=mask, keepdims=True)
                return np.sqrt(y)
            elif method == 'excess':
                if "errp" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["errp"].quantity, axis=1, where=mask, keepdims=True)
                elif "err" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["err"].quantity, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return yerr
            else:
                fact = 1.
                if method == 'brightness':
                    fact = 1. / self.table['solid_angle'].quantity
                # if method != 'excess' and self.table["excess"].unit == '':
                #     fact = fact.value
                if "flux_errp" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["flux_errp"].quantity*fact, axis=1, where=mask, keepdims=True)
                elif "flux_err" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["flux_err"].quantity*fact, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return yerr
        except KeyError:
            return None

    def profile_err_n(self, method='counts', energy_band=None):
        """Negative error quantity of the image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
           Compute the negative error of counts, excess, fluxes or the brightness within profile bins.
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        mask = self._get_energy_mask(energy_band)
        try:
            if method == 'counts':
                y = np.sum(self.table['counts'].quantity, axis=1, where=mask, keepdims=True) * -1.
                return np.sqrt(y)
            elif method == 'excess':
                if "errn" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["errn"].quantity, axis=1, where=mask, keepdims=True)
                elif "err" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["err"].quantity, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return yerr * -1.
            else:
                fact = 1.
                if method == 'brightness':
                    fact = 1. / self.table['solid_angle'].quantity
                # if method != 'excess' and self.table["excess"].unit == '':
                #     fact = fact.value
                if "flux_errn" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["flux_errn"].quantity*fact, axis=1, where=mask, keepdims=True)
                elif "flux_err" in self.table.colnames:
                    yerr = self._quadratic_sum(self.table["flux_err"].quantity*fact, axis=1, where=mask, keepdims=True)
                else:
                    return None
                return yerr * -1.
        except KeyError:
            return None

    def profile_ul(self, method='counts', energy_band=None):
        """ quantity of the image profile upper limits.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
           Compute the upper limit within profile bins.
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Upper limits computed for the required method of the image profile

        """
        mask = self._get_energy_mask(energy_band)
        try:
            if method == 'counts':
                y = np.sum(self.table['counts'].quantity, axis=1, where=mask, keepdims=True)
                return y + np.sqrt(y)
            elif method == 'excess':
                if "ul" in self.table.colnames:
                    return self._quadratic_sum(self.table["ul"].quantity, axis=1, where=mask, keepdims=True)
            else:
                if "flux_ul" not in self.table.colnames:
                    return None
                fact = 1.
                if method == 'brightness':
                    fact = 1. / self.table['solid_angle'].quantity
                # if method != 'excess' and self.table["excess"].unit == '':
                #     fact = fact.value
                return self._quadratic_sum(self.table["flux_ul"].quantity*fact, axis=1, where=mask, keepdims=True)
        except KeyError:
            return None

    def _quadratic_sum(self, aa, axis=None, where=None, keepdims=False):
        """Compute the quadratic sum of 1D-array elements

        Parameters
        ----------
        aa : `array_like`
            Elements to sum
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default, axis=None, will sum all of the elements of the \
            input array. If axis is negative it counts from the last to the first axis.
        where : array_like of bool, optional
            Elements to include in the sum
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one. \
            With this option, the result will broadcast correctly against the input array.
            If the default value is passed, then keepdims will not be passed through to the sum method of sub-classes \
            of ndarray, however any non-default value will be. If the sub-class’ method does not implement keepdims any\
             exceptions will be raised.


        Returns
        -------
         sum : `float`
            return value
        """
        return np.sqrt(np.sum(np.square(aa), axis=axis, where=where, keepdims=keepdims))

    def _get_energy_mask(self, energy_band):
        mask = np.full(self.table['counts'].quantity.shape, True)
        if 'energy_edge' not in self.table.colnames:
            return mask
        if energy_band is None or self.table['energy_edge'].quantity[0].size <= 2:
            return mask

        e_reco_lo = self.table['energy_edge'].quantity[:, :-1]
        e_reco_hi = self.table['energy_edge'].quantity[:, 1:]
        e_center = (e_reco_lo + e_reco_hi) / 2.
        mask = np.logical_and(energy_band[0] <= e_center, e_center <= energy_band[1])
        return mask

    def plot(self, method, n_sigma=3., energy_band=None, ax=None, **kwargs):
        """Plot image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'brightness']
            Compute counts, excess of the fluxes within profile bins.
        n_sigma : float
            Minimum number of sigma for which upper limits are plotted
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used
        ax : `~matplotlib.axes.Axes`
            Axes object
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.plot`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if energy_band is None:
            if 'energy_edge' in self.table.colnames:
                log.info(f" Used energy band : {self.table['energy_edge'][0]}\n")
            else:
                log.info(f" Using the full energy band \n")
        else:
            log.info(f" Used energy band : {energy_band}\n")
        y = self.profile(method, energy_band)
        errn = self.profile_err_n(method, energy_band)
        errp = self.profile_err_p(method, energy_band)
        if errn is None or errp is None:
            errn = errp = np.full(self.table['counts'].quantity.shape, 0.)
        yerr = [np.abs(errn), errp]
        y_ul = []
        x = self.table['x_ref']
        if n_sigma is not None and ("ul" not in self.table.colnames or "sqrt_ts" not in self.table.colnames):
            log.warning(f"No UL or SQRT_TS stored in the table")
        elif n_sigma is not None and "sqrt_ts" in self.table.colnames and "ul" in self.table.colnames and \
                "flux_ul" in self.table.colnames:
            en_mask = self._get_energy_mask(energy_band)
            sqrt_ts = self._quadratic_sum(self.table["sqrt_ts"].quantity, axis=1, where=en_mask, keepdims=True)
            ulmask = [val[0] < n_sigma for val in sqrt_ts]
            x_ul = x[ulmask]
            dx = np.zeros_like(x_ul)
            y_ul = self.profile_ul(method, energy_band)[ulmask]
            dy = y_ul * -0.05
            if ax.get_yaxis().get_scale() == "log":
                dy = y_ul * -0.3
            mask = [val[0] >= n_sigma for val in sqrt_ts]
            y = y[mask]
            x = x[mask]
            yerr = [np.abs(errn[mask]), errp[mask]]

        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='blue', **kwargs)
        ax.set_xlabel(f"Distance [{self.table['x_ref'].unit.to_string()}]")
        ax.set_ylabel(f"{method} profile [{y.unit.to_string()}]")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if len(y_ul) > 0:
            # Case of only ULs
            if len(x) < 1:
                xmin = ymin = 1.e10
                xmax = ymax = -1.e10
            arrow_width = 0.03
            if len(x_ul) >= 1:
                arrow_width = (x_ul[1]-x_ul[0])/3.
            for i, _ in enumerate(x_ul):
                ax.arrow(x=x_ul[i], y=y_ul[i].value[0], dx=dx[i], dy=dy[i].value[0],
                         color='blue', head_width=arrow_width, head_length=np.abs(dy[i].value[0]*0.4),
                         **kwargs)
            if np.max(y_ul).value > ymax:
                ymax = np.max(y_ul).value*1.1
            if np.min(y_ul+dy).value < ymin:
                ymin = np.min(y_ul+dy).value*0.8
            if np.max(x_ul) > xmax:
                xmax = np.max(x_ul) + np.abs(np.max(x_ul)*0.1)
            if np.min(x_ul) < xmin:
                xmin = np.min(x_ul) - np.abs(np.min(x_ul)*0.1)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)

        return ax

    def peek(self, method='excess', n_sigma=None, energy_band=None, figsize=(8, 4.5), **kwargs):
        """Show image profile and error.

        Parameters
        ----------
         method : ['counts', 'excess', 'flux', 'brightness']
            Compute counts, excess of the fluxes within profile bins.
         n_sigma : float
            Minimum number of sigma for which upper limits are plotted
        energy_band : `~astropy.units.Quantity`, e.g. Quantity([1, 20], 'TeV')
            User energy band. If None, the sum is used
         **kwargs : dict
            Keyword arguments passed to `ImageProfile.plot_profile()`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax = self.plot(method, n_sigma=n_sigma, energy_band=energy_band, ax=ax, **kwargs)

        return ax
