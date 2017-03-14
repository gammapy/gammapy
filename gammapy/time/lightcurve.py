# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Lightcurve and elementary temporal functions
"""
from astropy.table import QTable
from astropy.units import Quantity
from astropy.time import Time
import astropy.units as u
import numpy as np

from ..spectrum.utils import calculate_predicted_counts

__all__ = [
    'LightCurve',
    'LightCurveFactory',
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


class LightCurveFactory(object):
    """
    Class producing light curve

    Parameters
    ----------
    spectral_extraction: `~gammapy.spectrum.SpectrumExtraction`
       class storing statistics, IRF and event lists
    """

    def __init__(self, spectral_extraction):
        self.obs_list = spectral_extraction.obs
        self.obs_spec = spectral_extraction.observations
        self.off_evt_list = self._get_off_evt_list(spectral_extraction)
        self.on_evt_list = self._get_on_evt_list(spectral_extraction)

        
    def _get_off_evt_list(self, spectral_extraction):
        """
        Returns list of OFF events for each observations
        """
        off_evt_list = []
        for bg in spectral_extraction.background:
            off_evt_list.append(bg.off_events)
        return off_evt_list

    
    def _get_on_evt_list(self, spectral_extraction):
        """
        Returns list of OFF events for each observations
        """
        on_region = spectral_extraction.target.on_region
        on_evt_list = []
        for obs in spectral_extraction.obs:
            idx = on_region.contains(obs.events.radec)
            on_evt_list.append(obs.events.select_row_subset(idx))
            
        return on_evt_list
    
            
    def _get_time_intervals(self, t_start, t_stop, time_binning_type):
        """
        Compute time intervals needed for a light curve

        Parameters
        ----------
        time_start: `~astropy.units.Quantity`
            start time to integrate events
        time_stop: `~astropy.units.Quantity`
            start time to integrate events
        time_binning_type: `str`
            type of time binning (e.g. obs, minute, hour, night, day, month, user)
        """

        dt = None
        n_dt = None
        intervals = []
        if time_binning_type == 'obs':
            n_dt = len(self.obs_list)
            for obs in self.obs_list:
                # for now use time of first and last evt to define the interval
                # intervals.append([obs.tstart, obs.tstop])
                intervals.append([obs.events.time[0], obs.events.time[-1]])
        else:
            raise NotImplementedError

        return intervals

    
    def light_curve(self, t_start, t_stop, t_binning_type,
                    spectral_model, energy_range, binning=None):
        """
        Function returning light curve for given parameters
        
        Parameters
        ----------
        t_start: `~astropy.units.Quantity`
            start time to integrate events
        t_stop: `~astropy.units.Quantity`
            start time to integrate events
        t_binning_type: `str`
            type of time binning (e.g. obs, minute, hour, night, day, month, user)
        spectral_model: `~gammapy.spectrum.models.SpectralModel`
            spectral model
        energy_range: `~astropy.units.Quantity`
            true energy range to evaluate integrated flux (true energy)
        binning: `~astropy.units.Quantity` 
            bining 
        """

        # create time intervals
        if binning == 'user':
            t_intervals = binning
        else:
            t_intervals = self._get_time_intervals(t_start,
                                                   t_stop,
                                                   t_binning_type)

        lc_flux = []
        lc_flux_err = []
        lc_tmin = []
        lc_tmax = []
            
        # Loop on time intervals
        for t_bin in t_intervals:
            interval_tmin, interval_tmax = t_bin[0], t_bin[1]

            interval_livetime = 0
            interval_alpha_mean = 0.
            interval_alpha_mean_backup = 0.

            interval_measured_excess = 0
            interval_predicted_excess = 0
            interval_off = 0
            interval_on = 0
            
            # Loop on observations
            for t_index, obs in enumerate(self.obs_list):

                spec = self.obs_spec[t_index]

                obs_on = 0
                obs_off = 0
                obs_measured_excess = 0
                obs_predicted_excess = 0
                
                # discard observations not matching the time interval
                obs_start = obs.events.time[0]
                obs_stop = obs.events.time[-1]
                if ( (interval_tmin < obs_start and interval_tmax < obs_start)
                     or interval_tmin > obs_stop ):
                    continue

                # print('## t_index={}, obs={}'.format(t_index, obs.obs_id))
                
                # get ON and OFF evt list
                off_evt = self.off_evt_list[t_index]
                on_evt = self.on_evt_list[t_index]
                
                # Loop on energy bins
                for e_index in range(len(spec.e_reco)-1):
                    emin = spec.e_reco[e_index]
                    emax = spec.e_reco[e_index+1]

                    # discard bins not matching the energy threshold
                    if emin < spec.lo_threshold or emax > spec.hi_threshold:
                        continue

                    # Loop on ON evts (time and energy)
                    on = on_evt.select_energy([emin, emax])
                    on = on.select_time([interval_tmin, interval_tmax])
                    # print('on={}'.format(len(on.table)))
                    obs_on += len(on.table)

                    # Loop on OFF evts
                    off = off_evt.select_energy([emin, emax])
                    off = off.select_time([interval_tmin, interval_tmax])
                    # print('off={}'.format(len(off.table)))
                    obs_off += len(off.table)
                    
                # compute effective livetime (for the interval)
                livetime_to_add = 0.
                # interval included in obs
                if interval_tmin >= obs_start and interval_tmax <= obs_stop:
                    livetime_to_add = (interval_tmax - interval_tmin).to('s')
                # interval min above tstart from obs
                elif interval_tmin >= obs_start and interval_tmax >= obs_stop:
                    livetime_to_add = (obs_stop - interval_tmin).to('s')
                # interval min below tstart from obs
                elif interval_tmin <= obs_start and interval_tmax <= obs_stop:
                    livetime_to_add = (interval_tmax - obs_start).to('s')
                # obs included in interval
                elif interval_tmin <= obs_start and interval_tmax >= obs_stop:
                    livetime_to_add = (obs_stop - obs_start).to('s')
                else:
                    pass

                # Take into account dead time
                livetime_to_add *= (1. - obs.observation_dead_time_fraction)
                
                # Compute excess
                obs_measured_excess = obs_on - spec.alpha * obs_off

                # Compute expected excess
                # We use the effective livetime and the right energy threshold
                e_idx = np.where(np.logical_and(spec.e_reco>=spec.lo_threshold,
                                                spec.e_reco<=spec.hi_threshold))
                counts_predicted_excess = calculate_predicted_counts(livetime=livetime_to_add,
                                                                     aeff=spec.aeff,
                                                                     edisp=spec.edisp,
                                                                     model=spectral_model,
                                                                     e_reco=spec.e_reco[e_idx])
                obs_predicted_excess = np.sum(counts_predicted_excess.data.data)
                obs_predicted_excess /= obs.observation_live_time_duration.to('s').value
                obs_predicted_excess *= livetime_to_add.value

                # compute effective normalisation between ON/OFF (for the interval)
                interval_livetime += livetime_to_add
                interval_alpha_mean += spec.alpha * obs_off
                interval_alpha_mean_backup += spec.alpha * livetime_to_add
                interval_measured_excess += obs_measured_excess
                interval_predicted_excess += obs_predicted_excess
                interval_on += obs_on
                interval_off += obs_off
                
            # Fill time interval information
            int_flux = spectral_model.integral(energy_range[0], energy_range[1])
            if interval_off > 0.:
                interval_alpha_mean /= interval_off
            if interval_livetime > 0.:
                interval_alpha_mean_backup /= interval_livetime

            if interval_alpha_mean == 0.:  # use backup if necessary
                interval_alpha_mean = interval_alpha_mean_backup

            interval_flux = interval_measured_excess / interval_predicted_excess.value
            interval_flux *= int_flux
            interval_flux_error = int_flux / interval_measured_excess
            # Gaussian errors, ToDo: should be improved
            interval_flux_error *= np.sqrt(interval_on + interval_alpha_mean**2 * interval_off)

            print('### start:{}, stop:{}'.format(interval_tmin, interval_tmax))
            print('LIVETIME={}'.format(interval_livetime))
            print('ALPHA={}'.format(interval_alpha_mean))
            print('ON={}'.format(interval_on))
            print('OFF={}'.format(interval_off))
            print('EXCESS={}'.format(interval_measured_excess))
            print('EXP. EXCESS={}'.format(interval_predicted_excess))
            print('FLUX={}'.format(interval_flux))
            print('FLUX_ERR={}'.format(interval_flux_error))

            lc_flux.append(interval_flux.value)
            lc_flux_err.append(interval_flux_error.value)
            lc_tmin.append(interval_tmin.value)
            lc_tmax.append(interval_tmax.value)
            
        # Fill light curve data 
        lc = LightCurve()
        lc['FLUX'] = lc_flux * u.Unit('1 / (s cm2)')
        lc['FLUX_ERR'] = lc_flux_err * u.Unit('1 / (s cm2)')
        lc['TIME_MIN'] = Time(lc_tmin, format='mjd')
        lc['TIME_MAX'] = Time(lc_tmax, format='mjd')
        
        return lc
