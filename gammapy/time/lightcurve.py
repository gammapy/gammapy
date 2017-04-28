# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from collections import OrderedDict
from astropy.table import QTable
from astropy.units import Quantity
from astropy.time import Time
import astropy.units as u
import numpy as np

from ..spectrum.utils import CountsPredictor
from ..stats.poisson import excess_error

__all__ = [
    'LightCurve',
    'LightCurveEstimator',
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

        tstart = self['TIME_MIN']
        tstop = self['TIME_MAX']
        time = (tstart.value + tstop.value) / 2.0
        flux = self['FLUX'].to('cm-2 s-1')
        errors = self['FLUX_ERR'].to('cm-2 s-1')

        ax.errorbar(time, flux.value,
                    yerr=errors.value, linestyle="None")
        ax.scatter(time, flux)
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Flux ($cm^{-2} sec^{-1}$)")

        return ax

    @classmethod
    def simulate_example(cls):
        """Simulate an example `LightCurve`.

        TODO: add options to simulate some more interesting lightcurves.
        """
        lc = cls()

        lc['TIME_MIN'] = Time([1, 4, 7, 9], format='mjd')
        lc['TIME_MAX'] = Time([1, 4, 7, 9], format='mjd')
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
        flux = self['FLUX']
        ymean = np.mean(flux)
        yexp = ymean.value
        yobs = flux.value
        chi2, pval = stats.chisquare(yobs, yexp)
        return chi2, pval


class LightCurveEstimator(object):
    """
    Class producing light curve.

    Parameters
    ----------
    spec_extract : `~gammapy.spectrum.SpectrumExtraction`
       Contains statistics, IRF and event lists
    """

    def __init__(self, spec_extract):
        self.obs_list = spec_extract.obs_list
        self.obs_spec = spec_extract.observations
        self.off_evt_list = self._get_off_evt_list(spec_extract)
        self.on_evt_list = self._get_on_evt_list(spec_extract)

    @staticmethod
    def _get_off_evt_list(spec_extract):
        """
        Returns list of OFF events for each observations
        """
        off_evt_list = []
        for bg in spec_extract.bkg_estimate:
            off_evt_list.append(bg.off_events)
        return off_evt_list

    @staticmethod
    def _get_on_evt_list(spec_extract):
        """
        Returns list of OFF events for each observations
        """
        on_evt_list = []
        for obs in spec_extract.bkg_estimate:
            on_evt_list.append(obs.on_events)

        return on_evt_list

    def light_curve(self, time_intervals, spectral_model, energy_range):
        """Compute light curve.

        Implementation follows what is done in:
        http://adsabs.harvard.edu/abs/2010A%26A...520A..83H.

        To be discussed: assumption that threshold energy in the
        same in reco and true energy.

        TODO: For the moment there is an issue with the rebinning in reconstructed
        energy, do not use it: https://github.com/gammapy/gammapy/issues/953

        Parameters
        ----------
        time_intervals : `list` of `~astropy.time.Time`
            List of time intervals
        spectral_model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        energy_range : `~astropy.units.Quantity`
            True energy range to evaluate integrated flux (true energy)

        Returns
        -------
        lc : `~gammapy.time.LightCurve`
            Light curve
        """
        rows = []
        for time_interval in time_intervals:
            row = self.compute_flux_point(time_interval, spectral_model, energy_range)
            rows.append(row)

        return self._make_lc_from_row_data(rows)

    @staticmethod
    def _make_lc_from_row_data(rows):
        lc = LightCurve()
        lc['TIME_MIN'] = Time([_['TIME_MIN'].value for _ in rows], format='mjd')
        lc['TIME_MAX'] = Time([_['TIME_MAX'].value for _ in rows], format='mjd')

        lc['FLUX'] = [_['FLUX'].value for _ in rows] * u.Unit('1 / (s cm2)')
        lc['FLUX_ERR'] = [_['FLUX_ERR'].value for _ in rows] * u.Unit('1 / (s cm2)')

        lc['LIVETIME'] = [_['LIVETIME'].value for _ in rows] * u.s
        lc['N_ON'] = [_['N_ON'] for _ in rows]
        lc['N_OFF'] = [_['N_OFF'] for _ in rows]
        lc['ALPHA'] = [_['ALPHA'] for _ in rows]
        lc['MEASURED_EXCESS'] = [_['MEASURED_EXCESS'] for _ in rows]
        lc['EXPECTED_EXCESS'] = [_['EXPECTED_EXCESS'].value for _ in rows]

        return lc

    def compute_flux_point(self, time_interval, spectral_model, energy_range):
        """Compute one flux point for one time interval.

        Parameters
        ----------
        time_interval : `~astropy.time.Time`
            Time interval (2-element array, or a tuple of Time objects)
        spectral_model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        energy_range : `~astropy.units.Quantity`
            True energy range to evaluate integrated flux (true energy)

        Returns
        -------
        measurements : dict
            Dictionary with flux point measurement in the time interval
        """
        tmin, tmax = time_interval[0], time_interval[1]
        livetime = 0
        alpha_mean = 0.
        alpha_mean_backup = 0.
        measured_excess = 0
        predicted_excess = 0
        n_on = 0
        n_off = 0

        # Loop on observations
        for t_index, obs in enumerate(self.obs_list):

            spec = self.obs_spec[t_index]

            n_on_obs = 0
            n_off_obs = 0

            # discard observations not matching the time interval
            obs_start = obs.events.time[0]
            obs_stop = obs.events.time[-1]
            if (tmin < obs_start and tmax < obs_start) or (tmin > obs_stop):
                continue

            # get ON and OFF evt list
            off_evt = self.off_evt_list[t_index]
            on_evt = self.on_evt_list[t_index]

            # Loop on energy bins (default binning set to SpectrumObservation.e_reco)
            e_reco = spec.e_reco
            for e_index in range(len(e_reco) - 1):
                emin = e_reco[e_index]
                emax = e_reco[e_index + 1]

                # discard bins not matching the energy threshold of SpectrumObservation
                if emin < spec.lo_threshold or emax > spec.hi_threshold:
                    continue

                # Loop on ON evts (time and energy)
                on = on_evt.select_energy([emin, emax])  # evt in bin energy range
                on = on.select_energy(energy_range)  # evt in user energy range
                on = on.select_time([tmin, tmax])
                n_on_obs += len(on.table)

                # Loop on OFF evts
                off = off_evt.select_energy([emin, emax])  # evt in bin energy range
                off = off.select_energy(energy_range)  # evt in user energy range
                off = off.select_time([tmin, tmax])
                n_off_obs += len(off.table)

            # compute effective livetime (for the interval)
            livetime_to_add = 0.
            # interval included in obs
            if tmin >= obs_start and tmax <= obs_stop:
                livetime_to_add = (tmax - tmin).to('s')
            # interval min above tstart from obs
            elif tmin >= obs_start and tmax >= obs_stop:
                livetime_to_add = (obs_stop - tmin).to('s')
            # interval min below tstart from obs
            elif tmin <= obs_start and tmax <= obs_stop:
                livetime_to_add = (tmax - obs_start).to('s')
            # obs included in interval
            elif tmin <= obs_start and tmax >= obs_stop:
                livetime_to_add = (obs_stop - obs_start).to('s')
            else:
                pass

            # Take into account dead time
            livetime_to_add *= (1. - obs.observation_dead_time_fraction)

            # Compute excess
            obs_measured_excess = n_on_obs - spec.alpha * n_off_obs

            # Compute the expected excess in the range given by the user
            # but must respect the energy threshold of the observation
            # (to match the energy range of the measured excess)
            # We use the effective livetime and the right energy threshold
            e_idx = np.where(np.logical_and.reduce(
                (e_reco >= spec.lo_threshold,  # threshold
                 e_reco <= spec.hi_threshold,  # threshold
                 e_reco >= energy_range[0],  # user
                 e_reco <= energy_range[-1])  # user
            ))
            counts_predictor = CountsPredictor(
                livetime=livetime_to_add,
                aeff=spec.aeff,
                edisp=spec.edisp,
                model=spectral_model,
                # e_reco=e_reco[e_idx],
            )
            counts_predictor.run()
            counts_predicted_excess = counts_predictor.npred.data.data[e_idx[:-1]]

            obs_predicted_excess = np.sum(counts_predicted_excess)
            obs_predicted_excess /= obs.observation_live_time_duration.to('s').value
            obs_predicted_excess *= livetime_to_add.value

            # compute effective normalisation between ON/OFF (for the interval)
            livetime += livetime_to_add
            alpha_mean += spec.alpha * n_off_obs
            alpha_mean_backup += spec.alpha * livetime_to_add
            measured_excess += obs_measured_excess
            predicted_excess += obs_predicted_excess
            n_on += n_on_obs
            n_off += n_off_obs

        # Fill time interval information
        int_flux = spectral_model.integral(energy_range[0], energy_range[1])
        if n_off > 0.:
            alpha_mean /= n_off
        if livetime > 0.:
            alpha_mean_backup /= livetime
        if alpha_mean == 0.:  # use backup if necessary
            alpha_mean = alpha_mean_backup

        flux = measured_excess / predicted_excess.value
        flux *= int_flux
        flux_err = int_flux / measured_excess
        # Gaussian errors, TODO: should be improved
        flux_err *= excess_error(n_on=n_on, n_off=n_off, alpha=alpha_mean)

        # Store measurements in a dict and return that
        data = OrderedDict()
        data['TIME_MIN'] = Time(tmin, format='mjd')
        data['TIME_MAX'] = Time(tmax, format='mjd')

        data['FLUX'] = flux * u.Unit('1 / (s cm2)')
        data['FLUX_ERR'] = flux_err * u.Unit('1 / (s cm2)')

        data['LIVETIME'] = livetime * u.s
        data['ALPHA'] = alpha_mean
        data['N_ON'] = n_on
        data['N_OFF'] = n_off
        data['MEASURED_EXCESS'] = measured_excess
        data['EXPECTED_EXCESS'] = predicted_excess

        return data
