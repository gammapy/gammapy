# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import astropy.units as u
from ..stats import Stats, significance_on_off

__all__ = [
    'ObservationStats',
]


class ObservationStats(Stats):
    """Observation statistics.

    Class allowing to summarize observation
    (`~gammapy.data.DataStoreObservation`) statistics

    Parameters
    ----------
    n_on : int
        Number of on events
    n_off : int
        Number of off events
    a_on : float
        Relative background exposure of the on region
    a_off : float
        Relative background exposure of the off region
    obs_id : int
        ID of the observation
    livetime : float
        Livetime of the observation
    alpha : float
        Normalisation between the on and the off exposure
    bg_rate : float
        Background rate (/min)
    gamma_rate : float
        Gamma rate (/min)
    """

    def __init__(self,
                 n_on=None, n_off=None, a_on=None, a_off=None,
                 obs_id=None, livetime=None, alpha=None,
                 gamma_rate=None, bg_rate=None):
        super(ObservationStats, self).__init__(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
        )

        self.obs_id = obs_id
        self.livetime = livetime
        self.alpha_obs = alpha or a_on / a_off
        self.gamma_rate = gamma_rate or n_on / livetime
        self.bg_rate = bg_rate or self.alpha_obs * n_off / livetime

    @classmethod
    def from_obs(cls, obs, bg_estimate):
        """Create from `~gammapy.data.DataStoreObservation`.

        Parameters
        ----------
        obs : `~gammapy.data.DataStoreObservation`
            IACT data store observation
        bg_estimate : `~gammapy.background.BackgroundEstimate`
            Background estimate
        """
        n_on = len(bg_estimate.on_events.table)
        n_off = len(bg_estimate.off_events.table)
        a_on = bg_estimate.a_on
        a_off = bg_estimate.a_off

        obs_id = obs.obs_id
        livetime = obs.observation_live_time_duration
        alpha = a_on / a_off

        gamma_rate = n_on / livetime.to(u.min)
        bg_rate = (alpha * n_off) / livetime.to(u.min)
        stats = cls(n_on=n_on,
                    n_off=n_off,
                    a_on=a_on,
                    a_off=a_off,
                    obs_id=obs_id,
                    livetime=livetime,
                    alpha=alpha,
                    gamma_rate=gamma_rate,
                    bg_rate=bg_rate)
        return stats

    @property
    def alpha(self):
        """Alpha (on / off exposure ratio)

        Override member function from `~gammapy.stats.Stats`
        to take into account weighted alpha by number of Off events
        """
        return self.alpha_obs

    @property
    def sigma(self):
        """Li-Ma significance for observation statistics (`float`).
        """
        sigma = significance_on_off(
            self.n_on, self.n_off, self.alpha, method='lima')
        return sigma

    @classmethod
    def stack(cls, stats_list):
        """Stack (concatenate) list of `~gammapy.data.ObservationStats`.

        Parameters
        ----------
        stats_list : list
            List of `~gammapy.data.ObservationStats`

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
        obs_id = []
        livetime = 0
        alpha = 0
        alpha_backup = 0
        gamma_rate = 0
        bg_rate = 0
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
            gamma_rate += stats.n_on - stats.alpha * stats.n_off
            bg_rate += stats.n_off * stats.alpha

        # if no off events the weighting of alpha is done
        # with the livetime
        if n_off == 0:
            alpha = alpha_backup / livetime.value
            a_on = a_on_backup / livetime.value
            a_off = a_off_backup / livetime.value
        else:
            a_on /= n_off
            a_off /= n_off
            alpha /= n_off

        gamma_rate /= livetime.to(u.min)
        bg_rate /= livetime.to(u.min)

        return cls(
            n_on=n_on,
            n_off=n_off,
            a_on=a_on,
            a_off=a_off,
            obs_id=obs_id,
            livetime=livetime,
            alpha=alpha,
            gamma_rate=gamma_rate,
            bg_rate=bg_rate
        )

    def to_dict(self):
        """Data as an `~collections.OrderedDict`.

        This is useful for serialisation or putting the info in a table.
        """
        data = OrderedDict()
        data['obs_id'] = self.obs_id
        data['livetime'] = self.livetime
        data['n_on'] = self.n_on
        data['n_off'] = self.n_off
        data['a_on'] = self.a_on
        data['a_off'] = self.a_off
        data['alpha'] = self.alpha
        data['background'] = self.background
        data['excess'] = self.excess
        data['sigma'] = self.sigma
        data['gamma_rate'] = self.gamma_rate
        data['bg_rate'] = self.bg_rate
        return data

    def __str__(self):
        """Observation statistics report (`str`)."""
        ss = '*** Observation summary report ***\n'
        if type(self.obs_id) is list:
            obs_str = '[{}-{}]'.format(self.obs_id[0], self.obs_id[-1])
        else:
            obs_str = '{}'.format(self.obs_id)
        ss += 'Observation Id: {}\n'.format(obs_str)
        ss += 'Livetime: {:.3f}\n'.format(self.livetime.to(u.h))
        ss += 'On events: {}\n'.format(self.n_on)
        ss += 'Off events: {}\n'.format(self.n_off)
        ss += 'Alpha: {:.3f}\n'.format(self.alpha)
        ss += 'Bkg events in On region: {:.2f}\n'.format(self.background)
        ss += 'Excess: {:.2f}\n'.format(self.excess)
        ss += 'Excess / Background: {:.2f}\n'.format(np.divide(self.excess,
                                                               self.background))
        ss += 'Gamma rate: {:.2e}\n'.format(self.gamma_rate)
        ss += 'Bkg rate: {:.2e}\n'.format(self.bg_rate)
        ss += 'Sigma: {:.2f}\n'.format(self.sigma)

        return ss
