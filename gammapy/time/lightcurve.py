# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.utils.scripts import make_path

__all__ = ["LightCurve"]


class LightCurve:
    """Lightcurve container.

    The lightcurve data is stored in ``table``.

    For now we only support times stored in MJD format!

    TODO: specification of format is work in progress
    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/pull/61

    Usage: :ref:`time-lc`

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table with lightcurve data
    """

    def __init__(self, table):
        self.table = table

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self.table)})"

    @property
    def time_scale(self):
        """Time scale (str).

        Taken from table "TIMESYS" header.
        Common values: "TT" or "UTC".
        Assumed default is "UTC".
        """
        return self.table.meta.get("TIMESYS", "utc")

    @property
    def time_format(self):
        """Time format (str)."""
        return "mjd"

    # @property
    # def time_ref(self):
    #     """Time reference (`~astropy.time.Time`)."""
    #     return time_ref_from_dict(self.table.meta)

    def _make_time(self, colname):
        val = self.table[colname].data
        scale = self.time_scale
        format = self.time_format
        return Time(val, scale=scale, format=format)

    @property
    def time(self):
        """Time (`~astropy.time.Time`)."""
        return self.time_mid

    @property
    def time_min(self):
        """Time bin start (`~astropy.time.Time`)."""
        return self._make_time("time_min")

    @property
    def time_max(self):
        """Time bin end (`~astropy.time.Time`)."""
        return self._make_time("time_max")

    @property
    def time_mid(self):
        """Time bin center (`~astropy.time.Time`)."""
        return self.time_min + 0.5 * self.time_delta

    @property
    def time_delta(self):
        """Time bin width (`~astropy.time.TimeDelta`)."""
        return self.time_max - self.time_min

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from file.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.read`.
        """
        table = Table.read(make_path(filename), **kwargs)
        return cls(table=table)

    def write(self, filename, **kwargs):
        """Write to file.

        Parameters
        ----------
        filename : str
            Filename
        kwargs : dict
            Keyword arguments passed to `astropy.table.Table.write`.
        """
        self.table.write(make_path(filename), **kwargs)

    def compute_fvar(self):
        r"""Calculate the fractional excess variance.

        This method accesses the the ``FLUX`` and ``FLUX_ERR`` columns
        from the lightcurve data.

        The fractional excess variance :math:`F_{var}`, an intrinsic
        variability estimator, is given by

        .. math::
            F_{var} = \sqrt{\frac{S^{2} - \bar{\sigma^{2}}}{\bar{x}^{2}}}.

        It is the excess variance after accounting for the measurement errors
        on the light curve :math:`\sigma`. :math:`S` is the variance.

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
        flux = self.table["flux"].data.astype("float64")
        flux_err = self.table["flux_err"].data.astype("float64")

        flux_mean = np.mean(flux)
        n_points = len(flux)

        s_square = np.sum((flux - flux_mean) ** 2) / (n_points - 1)
        sig_square = np.nansum(flux_err ** 2) / n_points
        fvar = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

        sigxserr_a = np.sqrt(2 / n_points) * (sig_square / flux_mean) ** 2
        sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
        sigxserr = np.sqrt(sigxserr_a ** 2 + sigxserr_b ** 2)
        fvar_err = sigxserr / (2 * fvar)

        return fvar, fvar_err

    def compute_chisq(self):
        """Calculate the chi-square test for `LightCurve`.

        Chisquare test is a variability estimator. It computes
        deviations from the expected value here mean value

        Returns
        -------
        ChiSq, P-value : tuple of float or `~numpy.ndarray`
            Tuple of Chi-square and P-value
        """
        import scipy.stats as stats

        flux = self.table["flux"]
        yexp = np.mean(flux)
        yobs = flux.data
        chi2, pval = stats.chisquare(yobs, yexp)
        return chi2, pval

    def plot(self, ax=None, time_format="mjd", flux_unit="cm-2 s-1", **kwargs):
        """Plot flux points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        time_format : {'mjd', 'iso'}, optional
            If 'iso', the x axis will contain Matplotlib dates.
            For formatting these dates see: https://matplotlib.org/gallery/ticks_and_spines/date_demo_rrule.html
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        kwargs : dict
            Keyword arguments passed to :func:`matplotlib.pyplot.errorbar`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis object
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter

        if ax is None:
            ax = plt.gca()

        x, xerr = self._get_times_and_errors(time_format)
        y, yerr = self._get_fluxes_and_errors(flux_unit)
        is_ul, yul = self._get_flux_uls(flux_unit)

        # length of the ul arrow
        ul_arr = (
            np.nanmax(np.concatenate((y[~is_ul], yul[is_ul])))
            - np.nanmin(np.concatenate((y[~is_ul], yul[is_ul])))
        ) * 0.1

        # join fluxes and upper limits for the plot
        y[is_ul] = yul[is_ul]
        yerr[0][is_ul] = ul_arr

        # set plotting defaults and plot
        kwargs.setdefault("marker", "+")
        kwargs.setdefault("ls", "None")

        ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, uplims=is_ul, **kwargs)
        ax.set_xlabel("Time ({})".format(time_format.upper()))
        ax.set_ylabel("Flux ({:FITS})".format(u.Unit(flux_unit)))
        if time_format == "iso":
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )

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

    def _get_flux_uls(self, unit="cm-2 s-1"):
        """Extract flux upper limits

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

    def _get_times_and_errors(self, time_format="mjd"):
        """Extract times and corresponding errors

        Helper function for the plot method.

        Parameters
        ----------
        time_format : {'mjd', 'iso'}, optional
            Time format of the times. If 'iso', times and errors will be
            returned as `~datetime.datetime` and `~datetime.timedelta` objects

        Returns
        -------
        x : `~numpy.ndarray` or of `~datetime.datetime`
            Time values or `~datetime.datetime` instances if 'iso' is chosen
            as time format
        (xn, xp) : tuple of `numpy.ndarray` of `~datetime.timedelta`
            Tuple of time error values or `~datetime.timedelta` instances if
            'iso' is chosen as time format
        """
        x = self.time

        try:
            xn, xp = x - self.time_min, self.time_max - x
        except KeyError:
            xn, xp = x - x, x - x

        if time_format == "iso":
            x = x.datetime
            xn = xn.to_datetime()
            xp = xp.to_datetime()
        elif time_format == "mjd":
            x = x.mjd
            xn = xn.to("day").value
            xp = xp.to("day").value
        else:
            raise ValueError(f"Invalid time_format: {time_format}")

        return x, (xn, xp)
