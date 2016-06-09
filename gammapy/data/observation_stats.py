# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from ..stats import Stats
from gammapy.stats import significance_on_off

__all__ = [
    'ObservationStats',
]


class ObservationStats(Stats):
    """Observation statistics.

    Class allowing to summarize observation 
    (`~gammapy.data.DataStoreObservation) statistics

    Parameters
    ----------
    n_on : int
        number of on events
    n_off : int
        number of on events
    a_on : float
        number of on events
    a_off : float
        number of on events
    obs_id : int
        number of on events
    livetime : float
        number of on events
    alpha : float
        number of on events

    """

    def __init__(self,
                 n_on=None, n_off=None, a_on=None, a_off=None,
                 obs_id=None, livetime=None, alpha=None):
        super(ObservationStats, self).__init__(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off
        )

        self.obs_id = obs_id
        self.livetime = livetime
        self.alpha_obs = alpha

    @classmethod
    def from_target(cls, obs, target, bg_estimate):
        """
        Auxiliary constructor from an osbervation, a target
        a background estimate

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation index table
        target_pos : `~astropy.coordinates.SkyCoord`
            Target position
        bg_method : str
            Background estimation method
        """
        n_on = cls._get_on_events(obs, target)
        n_off = len(bg_estimate.off_events)
        a_on = bg_estimate.a_on
        a_off = bg_estimate.a_off

        obs_id = obs.obs_id
        livetime = obs.observation_live_time_duration
        alpha = a_on / a_off
        stats = cls(n_on=n_on,
                    n_off=n_off,
                    a_on=a_on,
                    a_off=a_off,
                    obs_id=obs_id,
                    livetime=livetime,
                    alpha=alpha)
        return stats

    @property
    def alpha(self):
        """Override member function from `~gammapy.stats.Stats`
        to take into account weighted alpha by number of Off events
        """
        return self.alpha_obs

    @property
    def sigma(self):
        """Li-Ma significance for observation statistics (`float`)
        """
        sigma = significance_on_off(
            self.n_on, self.n_off, self.alpha, method='lima')
        return sigma

    @staticmethod
    def _get_on_events(obs, target):
        """Number of ON events in the region of interest (`int`)
        """
        print(target)
        idx = target.on_region.contains(obs.events.radec)
        on_events = obs.events[idx]
        return len(on_events)

    @classmethod
    def stack(cls, stats_list):
        """Stack statistics from a list of 
        `~gammapy.data.ObservationStats` and returns a new instance 
        of `~gammapy.data.ObservationStats`

        Parameters
        ----------
        stats_list : list
            List of observation statistics `~gammapy.data.ObservationStats`
      
        Returns
        -------
        total_stats : `~gammapy.data.ObservationStats`
            Statistics for stacked observation 
        """
        n_on = 0
        n_off = 0
        a_on = 0
        a_on_backup = 0
        a_off = 0
        a_off_backup = 0
        alpha = 0
        alpha_backup = 0
        livetime = 0
        obs_id = list()
        for stats in stats_list:
            livetime += stats.livetime
            n_on += stats.n_on
            n_off += stats.n_off
            a_on += stats.a_on * stats.n_off
            a_on_backup += stats.a_on * stats.livetime.value
            a_off += stats.a_off * stats.n_off
            a_off_backup += stats.a_off * stats.livetime.value
            alpha += stats.alpha * stats.n_off
            alpha_backup += stats.alpha * stats.livetime.value
            obs_id.append(stats.obs_id)

        a_on /= n_off
        a_off /= n_off
        alpha /= n_off

        # if no off events the weighting of alpha is done
        # with the livetime
        if n_off == 0:
            alpha = alpha_backup / livetime.value
            a_on_backup = a_on_backup / livetime.value
            a_off_backup = a_off_backup / livetime.value

        total_stats = cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
            alpha=alpha
        )
        return total_stats

    def __add__(self, other):
        """Add statistics from two observations
        and returns new instance of `~gammapy.data.ObservationStats`
        """
        return ObservationStats.stack(self, other)

    def __str__(self):
        """Observation statistics report (`str`)
        """
        ss = '*** Observation summary report ***\n'
        ss += 'Observation Id: {}\n'.format(self.obs_id)
        ss += 'Livetime: {}\n'.format(self.livetime.to(u.h))
        ss += 'On events: {}\n'.format(self.n_on)
        ss += 'Off events: {}\n'.format(self.n_off)
        ss += 'Alpha: {}\n'.format(self.alpha)
        ss += 'Bkg events in On region: {}\n'.format(self.background)
        ss += 'Excess: {}\n'.format(self.excess)
        ss += 'Gamma rate: {}\n'.format(self.n_on / self.livetime.to(u.min))
        ss += 'Bkg rate rate: {}\n'.format(self.background /
                                           self.livetime.to(u.min))
        ss += 'Sigma: {}\n'.format(self.sigma)

        return ss
