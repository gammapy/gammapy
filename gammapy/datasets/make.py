# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.time import Time, TimeDelta
from astropy.table import Table
from astropy.coordinates import SkyCoord, AltAz, FK5, Angle
from ..irf import EnergyDependentMultiGaussPSF
from ..obs import ObservationTable, observatory_locations
from ..utils.random import sample_sphere
from ..utils.time import time_ref_from_dict, time_relative_to_ref

__all__ = ['make_test_psf',
           'make_test_observation_table',
           ]


def make_test_psf(energy_bins=15, theta_bins=12):
    """Create a test FITS PSF file.

    A log-linear dependency in energy is assumed, where the size of
    the PSF decreases by a factor of tow over tow decades. The
    theta dependency is a parabola where at theta = 2 deg the size
    of the PSF has increased by 30%.

    Parameters
    ----------
    energy_bins : int
        Number of energy bins
    theta_bins : int
        Number of theta bins

    Returns
    -------
    psf : `~gammapy.irf.EnergyDependentMultiGaussPSF`
        PSF
    """
    energies_all = np.logspace(-1, 2, energy_bins + 1)
    energies_lo = energies_all[:-1]
    energies_hi = energies_all[1:]
    theta_lo = theta_hi = np.linspace(0, 2.2, theta_bins)
    azimuth_lo = azimuth_hi = 0
    zenith_lo = zenith_hi = 0

    def sigma_energy_theta(energy, theta, sigma):
        # log-linear dependency of sigma with energy
        # m and b are choosen such, that at 100 TeV
        # we have sigma and at 0.1 TeV we have sigma/2
        m = -sigma / 6.
        b = sigma + m
        return (2 * b + m * np.log10(energy)) * (0.3 / 4 * theta ** 2 + 1)

    # Compute norms and sigmas values are taken from the psf.txt in
    # irf/test/data
    energies, thetas = np.meshgrid(energies_lo, theta_lo)

    sigmas = []
    for sigma in [0.0219206, 0.0905762, 0.0426358]:
        sigmas.append(sigma_energy_theta(energies, thetas, sigma))

    norms = []
    for norm in 302.654 * np.array([1, 0.0406003, 0.444632]):
        norms.append(norm * np.ones((theta_bins, energy_bins)))

    psf = EnergyDependentMultiGaussPSF(Quantity(energies_lo, 'TeV'),
                                       Quantity(energies_hi, 'TeV'),
                                       Quantity(theta_lo, 'deg'),
                                       sigmas, norms, azimuth=azimuth_hi,
                                       zenith=zenith_hi)

    return psf


def make_test_observation_table(observatory_name, n_obs, debug=False):
    """Make a test observation table.

    For the moment, only random observation tables are created.

    Parameters
    ----------
    observatory_name : string
        name of the observatory; a list of choices is given in `~gammapy.obs.observatory_locations`
    n_obs : integer
        number of observations for the obs table
    debug : bool
        show UTC times instead of seconds after the reference

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
        observation table
    """
    n_obs_start = 1

    obs_table = ObservationTable()

    # build a time reference as the start of 2010
    dateref = Time('2010-01-01 00:00:00', format='iso', scale='utc')
    dateref_mjd_fra, dateref_mjd_int = np.modf(dateref.mjd)

    # header
    header = {'OBSERVATORY_NAME': observatory_name,
              'MJDREFI': dateref_mjd_int, 'MJDREFF': dateref_mjd_fra}
    ObservationTable.meta = header

    # obs id
    obs_id = np.arange(n_obs_start, n_obs_start + n_obs)
    obs_table['OBS_ID'] = obs_id

    # obs time: 30 min
    time_observation = Quantity(
        30. * np.ones_like(obs_id), 'minute').to('second')
    obs_table['TIME_OBSERVATION'] = time_observation

    # livetime: 25 min
    time_live = Quantity(25. * np.ones_like(obs_id), 'minute').to('second')
    obs_table['TIME_LIVE'] = time_live

    # start time
    # random points between the start of 2010 and the end of 2014
    # using the start of 2010 as a reference time for the header of the table
    # observations restrict to night time
    # considering start of astronomical day at midday: implicit in setting the
    # start of the night, when generating random night hours
    datestart = Time('2010-01-01 00:00:00', format='iso', scale='utc')
    dateend = Time('2015-01-01 00:00:00', format='iso', scale='utc')
    time_start = Time((dateend.mjd - datestart.mjd) *
                      np.random.random(len(obs_id)) + datestart.mjd, format='mjd', scale='utc')

    # keep only the integer part (i.e. the day, not the fraction of the day)
    time_start_f, time_start_i = np.modf(time_start.mjd)
    time_start = Time(time_start_i, format='mjd', scale='utc')

    # random generation of night hours: 6 h (from 22 h to 4 h), leaving 1/2 h
    # time for the last run to finish
    night_start = Quantity(22., 'hour')
    night_duration = Quantity(5.5, 'hour')
    hour_start = night_start + night_duration * np.random.random(len(obs_id))

    # add night hour to integer part of MJD
    time_start += hour_start

    if debug:
        # show the observation times in UTC
        time_start = time_start.iso
    else:
        # show the observation times in seconds after the reference
        time_start = time_relative_to_ref(time_start, header)
        # converting to quantity (beter treatment of units)
        time_start = Quantity(time_start.sec, 'second')

    obs_table['TIME_START'] = time_start

    # stop time
    # calculated as TIME_START + TIME_OBSERVATION
    if debug:
        time_stop = Time(obs_table['TIME_START']) + \
            TimeDelta(obs_table['TIME_OBSERVATION'])
    else:
        time_stop = TimeDelta(
            obs_table['TIME_START']) + TimeDelta(obs_table['TIME_OBSERVATION'])
        # converting to quantity (beter treatment of units)
        time_stop = Quantity(time_stop.sec, 'second')

    obs_table['TIME_STOP'] = time_stop

    # az, alt
    # random points in a sphere above 45 deg altitude
    az, alt = sample_sphere(len(obs_id), (0, 360), (45, 90), 'degree')
    az = Angle(az, 'degree')
    alt = Angle(alt, 'degree')
    obs_table['AZ'] = az
    obs_table['ALT'] = alt

    # RA, dec
    # derive from az, alt taking into account that alt, az represent the values
    # at the middle of the observation, i.e. at time_ref + (TIME_START + TIME_STOP)/2
    # (or better: time_ref + TIME_START + (TIME_OBSERVATION/2))
    # in debug modus, the time_ref should not be added, since it's already included
    # in TIME_START and TIME_STOP
    az = Angle(obs_table['AZ'])
    alt = Angle(obs_table['ALT'])
    if debug:
        obstime = Time(obs_table['TIME_START']) + \
            TimeDelta(obs_table['TIME_OBSERVATION']) / 2.
    else:
        obstime = time_ref_from_dict(obs_table.meta) + TimeDelta(
            obs_table['TIME_START']) + TimeDelta(obs_table['TIME_OBSERVATION']) / 2.
    location = observatory_locations[observatory_name]
    alt_az_coord = AltAz(az=az, alt=alt, obstime=obstime, location=location)
    # optional: make it depend on other pars: temperature, pressure,
    # humidity,...
    sky_coord = alt_az_coord.transform_to(FK5)
    obs_table['RA'] = sky_coord.ra
    obs_table['DEC'] = sky_coord.dec

    # positions

    # number of telescopes
    # random integers between 3 and 4
    n_tels_min = 3
    n_tels_max = 4
    n_tels = np.random.randint(n_tels_min, n_tels_max + 1, len(obs_id))
    obs_table['N_TELS'] = n_tels

    # muon efficiency
    # random between 0.6 and 1.0
    muon_efficiency_min = 0.6
    muon_efficiency_max = 1.0
    muon_efficiency = np.random.random(
        len(obs_id)) * (muon_efficiency_max - muon_efficiency_min) + muon_efficiency_min
    obs_table['MUON_EFFICIENCY'] = muon_efficiency

    return obs_table
