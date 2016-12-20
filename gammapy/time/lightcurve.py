# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import QTable
from astropy.units import Quantity
import astropy.units as u
import numpy as np

__all__ = [
    'LightCurve',
]


class LightCurve(QTable):
    """LightCurve class.

    Contains all data in the tabular form with columns
    tstart, tstop, flux, flux error, time bin (opt).
    Possesses functions allowing plotting data, saving as txt
    and elementary stats like mean & std dev.

    TODO: specification of format is work in progress
    See https://github.com/open-gamma-ray-astro/gamma-astro-data-formats/pull/61
    """

    def plot(self, ax=None):
        """Plot flux versus time.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes` or None, optional.
            The `~matplotlib.axes.Axes` object to be drawn on.
            If None, uses the current `~matplotlib.axes.Axes`.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        tstart = self['TIME_MIN'].to('s')
        tstop = self['TIME_MAX'].to('s')
        time = (tstart + tstop) / 2.0
        flux = self['FLUX'].to('cm-2 s-1')
        errors = self['FLUX_ERR'].to('cm-2 s-1')

        ax.errorbar(time.value, flux.value,
                    yerr=errors.value, linestyle="None")
        ax.scatter(time, flux)
        ax.set_xlabel("Time (secs)")
        ax.set_ylabel("Flux ($cm^{-2} sec^{-1}$)")

        return ax

    @classmethod
    def simulate_example(cls):
        """Simulate an example `LightCurve`.

        TODO: add options to simulate some more interesting lightcurves.
        """
        lc = cls()

        lc['TIME_MIN'] = [1, 4, 7, 9] * u.s
        lc['TIME_MAX'] = [1, 4, 7, 9] * u.s
        lc['FLUX'] = Quantity([1, 4, 7, 9], 'cm^-2 s^-1')
        lc['FLUX_ERR'] = Quantity([0.1, 0.4, 0.7, 0.9], 'cm^-2 s^-1')

        return lc

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
        fvar, fvar_err : `numpy.array`
            Fractional excess variance.

        References
        ----------
        .. [Vaughan2003] "On characterizing the variability properties of X-ray light
           curves from active galaxies", Vaughan et al. (2003)
           http://adsabs.harvard.edu/abs/2003MNRAS.345.1271V
        """
        flux = self['FLUX'].value.astype('float64')
        flux_err = self['FLUX_ERR'].value.astype('float64')
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
