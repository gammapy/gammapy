# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u
import logging

__all__ = ["ImageProfile"]

log = logging.getLogger(__name__)


class ImageProfile:
    """Image profile class.

    The image profile data is stored in `~astropy.table.Table` object, with the
    following columns:

        * `x_ref` Coordinate bin center (required).
        * `x_min` Coordinate bin minimum (optional).
        * `x_max` Coordinate bin maximum (optional).
        * `counts` Counts profile data (required).
        * `background` Estimated background profile data (required).
        * `excess` Excess profile data (required).
        * `alpha` Bin exposure_on/Exposure_off ratio (optional).
        * `ts` Bin statistics (optional).
        * `sqrt_ts` square-root of TS (optional).
        * `err` Excess profile data error (optional).
        * `errn` Excess profile data lower error (optional).
        * `errp` Excess profile data upper error (optional).
        * `ul` Excess profile data upper limit (optional).
        * `exposure` Bin exposure (required).
        * `solid_angle` Region solid angle (required).

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table instance with the columns specified as above.
    """

    def __init__(self, table):
        self.table = table

    def profile(self, method='counts'):
        """Image profile quantity.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute counts, excess, fluxes or the radiance within profile bins.

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Data computed for the required method of the image profile

        """
        try:
            if method == 'counts':
                y = self.table['counts'].quantity.copy()
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                else:
                    raise AttributeError(f"The method [{method}] is not supported")
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value
                y = self.table["excess"].quantity.copy() * fact
                y.name = method
            return y
        except AttributeError:
            raise AttributeError(f"Missing data to compute [{method}]")

    def profile_err(self, method='counts'):
        """Mean error quantity of the image profile

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute the mean error of counts, excess, fluxes or the radiance within profile bins.

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        try:
            if method == 'counts':
                return np.sqrt(self.table['counts'].quantity.copy())
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                else:
                    raise AttributeError(f"The method [{method}] is not supported")
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                ymin = self.table["errn"].quantity.copy() * fact
                ymax = self.table["errp"].quantity.copy() * fact
                return (ymin+ymax)/2.
        except KeyError:
            return None

    def profile_err_p(self, method='counts'):
        """Positive error quantity of the image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute the positive error of counts, excess, fluxes or the radiance within profile bins.

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        try:
            if method == 'counts':
                return np.sqrt(self.table['counts'].quantity.copy())
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                return self.table["errp"].quantity.copy() * fact
        except KeyError:
            return None

    def profile_err_m(self, method='counts'):
        """Negative error quantity of the image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute the negative error of counts, excess, fluxes or the radiance within profile bins.

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Errors on the data computed for the required method of the image profile

        """
        try:
            if method == 'counts':
                return np.sqrt(self.table['counts'].quantity.copy())
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                return self.table["errn"].quantity.copy() * fact
        except KeyError:
            return None

    def profile_ul(self, method='counts'):
        """Negative error quantity of the image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute the upper limit within profile bins.

        Returns
        -------
        value: array of `astropy.units.quantity.Quantity`
            Upper limits computed for the required method of the image profile

        """
        try:
            if method == 'counts':
                return self.table['counts'].quantity.copy()
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                return self.table["ul"].quantity.copy() * fact
        except KeyError:
            return None

    def normalize(self, mode="peak", method="counts"):
        """Normalize profile to peak value or integral.

        Parameters
        ----------
        mode : ['integral', 'peak']
            Normalize image profile so that it integrates to unity ('integral')
            or the maximum value corresponds to one ('peak').
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute the negative error of counts, excess, fluxes or the radiance within profile bins.

        Returns
        -------
        profile : `ImageProfile`
            Normalized image profile.
        """

        table = self.table.copy()

        if method == "counts":
            y = self.profile('counts').copy()
            if mode == "peak":
                norm = np.nanmax(y)
            elif mode == "integral":
                norm = np.nansum(y)
            else:
                raise ValueError(f"Invalid normalization mode: {mode!r}")
            table['counts'] = [(ii/norm).value for ii in table['counts'].quantity]*u.dimensionless_unscaled
        else:
            y = self.profile(method).copy()
            if mode == "peak":
                norm = np.nanmax(y)
            elif mode == "integral":
                norm = np.nansum(y)
            else:
                raise ValueError(f"Invalid normalization mode: {mode!r}")
            table['excess'] = [(ii / norm).value for ii in table['excess'].quantity] * u.dimensionless_unscaled
            if "err" in table.colnames:
                table['err'] = [(ii / norm).value for ii in table['err'].quantity] * u.dimensionless_unscaled
            if "errn" in table.colnames:
                table['errn'] = [(ii / norm).value for ii in table['errn'].quantity] * u.dimensionless_unscaled
            if "errp" in table.colnames:
                table['errp'] = [(ii/norm).value for ii in table['errp'].quantity]*u.dimensionless_unscaled
            if "ul" in table.colnames:
                table['ul'] = [(ii / norm).value for ii in table['ul'].quantity] * u.dimensionless_unscaled

        return self.__class__(table)

    def plot(self, method, n_sigma=3., ax=None, **kwargs):
        """Plot image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
            Compute counts, excess of the fluxes within profile bins.
        n_sigma : float
            Minimum number of sigma for which upper limits are plotted
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

        y = self.profile(method)
        yerr = [self.profile_err_m(method).value * -1., self.profile_err_p(method).value]
        x = self.table['x_ref']
        if n_sigma is not None and ("ul" not in self.table.colnames or "sqrt_ts" not in self.table.colnames):
            log.warning(f"No UL stored or TS in the table")
        elif n_sigma is not None and "ul" in self.table.colnames and "sqrt_ts" in self.table.colnames:
            ulmask = [(ii.value < n_sigma) for ii in self.table["sqrt_ts"].quantity]
            x_ul = x[ulmask].copy()
            dx = np.zeros_like(x_ul)
            y_ul = self.profile_ul(method)[ulmask]
            dy = y_ul * -0.05
            if ax.get_yaxis().get_scale() == "log":
                dy = y_ul * -0.3
            mask = [(ii.value >= n_sigma) for ii in self.table["sqrt_ts"].quantity]
            y = y[mask]
            x = x[mask]
            yerr = [self.profile_err_m(method).value[mask] * -1., self.profile_err_p(method).value[mask]]

        ax.errorbar(x, y.data, yerr=yerr, fmt='o', ecolor='blue', **kwargs)
        ax.set_xlabel(f"Distance [{self.table['x_ref'].unit.to_string()}]")
        ax.set_ylabel(f"Profile [{y.unit.to_string()}]")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        try:
            y_ul
        except NameError:
            log.warning("Not plotting the ULs")
        else:
            if len(y_ul) > 0:
                arrow_width = 0.03
                if len(x) > 1:
                    arrow_width = (x[1]-x[0])/3.
                for i in range(len(x_ul)):
                    ax.arrow(x=x_ul[i], y=y_ul.data[i], dx=dx[i], dy=dy.data[i],
                             color='blue', head_width=arrow_width, head_length=np.abs(dy.data[i]*0.4),
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

    def peek(self, method='excess', n_sigma=None, figsize=(8, 4.5), **kwargs):
        """Show image profile and error.

        Parameters
        ----------
         method : ['counts', 'excess', 'flux', 'radiance']
            Compute counts, excess of the fluxes within profile bins.
         n_sigma : float
            Minimum number of sigma for which upper limits are plotted
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
        ax = self.plot(method, n_sigma=n_sigma, ax=ax, **kwargs)

        return ax
