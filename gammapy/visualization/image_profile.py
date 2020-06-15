# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
from astropy import units as u

__all__ = ["ImageProfile"]


class ImageProfile:
    """Image profile class.

    The image profile data is stored in `~astropy.table.Table` object, with the
    following columns:

        * `x_ref` Coordinate bin center (required).
        * `x_min` Coordinate bin minimum (optional).
        * `x_max` Coordinate bin maximum (optional).
        * `counts` Counts profile data (required).
        * `counts_err` Counts profile data error (optional).
        * `excess` Excess profile data (required).
        * `excess_err_m` Excess profile data lower error (optional).
        * `excess_err_p` Excess profile data upper error (optional).
        * `exposure` Bin exposure (optional).
        * `solid_angle` Region solid angle (optional).

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
                return self.table['counts_err'].quantity.copy()
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

                ymin = self.table["excess_err_m"].quantity.copy() * fact
                ymax = self.table["excess_err_p"].quantity.copy() * fact
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
                return self.table['counts_err'].quantity.copy()
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                return self.table["excess_err_p"].quantity.copy() * fact
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
                return self.table['counts_err'].quantity.copy()
            else:
                if method == 'excess':
                    fact = 1.
                elif method == 'radiance':
                    fact = 1. / self.table['exposure'].quantity.copy() / self.table['solid_angle'].quantity.copy()
                elif method == 'flux':
                    fact = 1. / self.table['exposure'].quantity.copy()
                if method != 'excess' and self.table["excess"].unit == '':
                    fact = fact.value

                return self.table["excess_err_m"].quantity.copy() * fact
        except KeyError:
            return None

    def normalize(self, mode="peak"):
        """Normalize profile to peak value or integral.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
           Compute counts, excess, fluxes or the radiance within profile bins.

        mode : ['integral', 'peak']
            Normalize image profile so that it integrates to unity ('integral')
            or the maximum value corresponds to one ('peak').

        Returns
        -------
        profile : `ImageProfile`
            Normalized image profile.
        """

        table = self.table.copy()

        y = self.profile('counts').copy()
        if mode == "peak":
            norm = np.nanmax(y)
        elif mode == "integral":
            norm = np.nansum(y)
        else:
            raise ValueError(f"Invalid normalization mode: {mode!r}")
        table['counts'] = [(ii/norm).value for ii in table['counts'].quantity]*u.dimensionless_unscaled
        if "counts_err" in table.colnames:
            table['counts_err'] = [(ii/norm).value for ii in table['counts_err'].quantity]*u.dimensionless_unscaled

        y = self.profile('excess').copy()
        if mode == "peak":
            norm = np.nanmax(y)
        elif mode == "integral":
            norm = np.nansum(y)
        else:
            raise ValueError(f"Invalid normalization mode: {mode!r}")
        table['excess'] = [(ii / norm).value for ii in table['excess'].quantity] * u.dimensionless_unscaled
        if "excess_err_m" in table.colnames:
            table['excess_err_m'] = [(ii / norm).value for ii in table['excess_err_m'].quantity] * u.dimensionless_unscaled
        if "excess_err_p" in table.colnames:
            table['excess_err_p'] = [(ii/norm).value for ii in table['excess_err_p'].quantity]*u.dimensionless_unscaled

        return self.__class__(table)

    def plot(self, method='counts', ax=None, **kwargs):
        """Plot image profile.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
            Compute counts, excess of the fluxes within profile bins.
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
        x = self.table['x_ref']
        ax.plot(x, y.data, **kwargs)
        ax.set_xlabel(f"Distance [{self.table['x_ref'].unit.to_string()}]")
        ax.set_ylabel(f"Profile [{y.unit.to_string()}]")
        return ax

    def plot_err(self, method='counts', ax=None, **kwargs):
        """Plot image profile error as band.

        Parameters
        ----------
        method : ['counts', 'excess', 'flux', 'radiance']
            Compute counts, excess of the fluxes within profile bins.
        ax : `~matplotlib.axes.Axes`
            Axes object
        **kwargs : dict
            Keyword arguments passed to plt.fill_between()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        y = self.profile(method)
        ymin = y - self.profile_err_m(method)
        ymax = y + self.profile_err_p(method)
        x = self.table['x_ref']

        # plotting defaults
        kwargs.setdefault("alpha", 0.5)

        ax.fill_between(x, ymin.data, ymax.data, **kwargs)
        ax.set_xlabel(f"Distance [{self.table['x_ref'].unit.to_string()}]")
        ax.set_ylabel(f"Profile [{y.unit.to_string()}]")
        return ax

    def peek(self, method='counts', figsize=(8, 4.5), **kwargs):
        """Show image profile and error.

        Parameters
        ----------
         method : ['counts', 'excess', 'flux', 'radiance']
            Compute counts, excess of the fluxes within profile bins.

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
        ax = self.plot(method, ax, **kwargs)
        ax = self.plot_err(method, ax, color=kwargs.get("c"))

        return ax
