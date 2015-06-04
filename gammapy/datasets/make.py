# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.time import Time
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, AltAz, FK5, Angle
from ..irf import EnergyDependentMultiGaussPSF
from ..obs import ObservationTable, observatory_locations
from ..utils.random import sample_sphere

__all__ = ['make_test_psf',
           'generate_observation_table',
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


def generate_observation_table(observatory, n_obs):
    """Generate an observation table.

    For the moment, only random observation tables are created.

    Parameters
    ----------
    observatory : string
    	name of the observatory
    n_obs : integer
    	number of observations for the obs table

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
    	observation table
    """
    n_obs_start = 1

    astro_table = Table()

    # obs id
    obs_id = np.arange(n_obs_start, n_obs_start + n_obs)
    col_obs_id = Column(name='OBS_ID', data=obs_id)
    astro_table.add_column(col_obs_id)

    # on time
    ontime = Quantity(30.*np.ones_like(col_obs_id.data), 'minute')
    col_ontime = Column(name='ONTIME', data=ontime)
    astro_table.add_column(col_ontime)

    # livetime
    livetime = Quantity(25.*np.ones_like(col_obs_id.data), 'minute')
    col_livetime = Column(name='LIVETIME', data=livetime)
    astro_table.add_column(col_livetime)

    # TODO: add columns for coordinates:
    #       alt az
    #       date -> done
    #       calculate pointing observation (ra dec), considering the observatory location, and alt az is at the middle of the observation
    #       then convert to alt az, considering the observatory location

    # TODO: is there a way to comment on the column names?!!!!!
    # TODO: store obs name as a column or as a header value?!!!
    # TODO: format times!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: alt az at mean time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: restrict to night time? (or dark time?)!!!!!!!!!!!!

    # start time
    # random points between the start of 2010 and the end of 2014
    # TODO: restrict to night time? (or dark time?)!!!
    #       if so: take into acount that enough time has to be permitted for the observation to finish (~ 30 min)
    datestart = Time('2010-01-01 00:00:00', format='iso', scale='utc')
    dateend = Time('2015-01-01 00:00:00', format='iso', scale='utc')
    time_start = Time((dateend.mjd - datestart.mjd)*np.random.random(len(col_obs_id)) + datestart.mjd, format='mjd', scale='utc').iso
    # TODO: using format "iso" for ~astropy.Time for now; eventually change it
    # to "fits" after the next astropy stable release (>1.0) is out.
    col_time_start = Column(name='TSTART', data=time_start)
    astro_table.add_column(col_time_start)

    # stop time
    # calculated as TSTART + ONTIME
    time_stop = Time(astro_table['TSTART']) + astro_table['ONTIME']
    col_time_stop = Column(name='TSTOP', data=time_stop)
    astro_table.add_column(col_time_stop)

    # az, alt
    # random points in a sphere above 45 deg altitude
    az, alt = sample_sphere(len(col_obs_id), (0, 360), (45, 90), 'degree')
    az = Angle(az, 'degree')
    alt = Angle(alt, 'degree')
    col_az = Column(name='AZ', data=az)
    astro_table.add_column(col_az)
    col_alt = Column(name='ALT', data=alt)
    astro_table.add_column(col_alt)

    # RA, dec
    # derive from az, alt taking into account that alt, az represent the values at the middle of the observation, i.e. at (TSTART + TSTOP)/2 (or TSTART + (ONTIME/2))
    az = Angle(astro_table['AZ'])
    alt = Angle(astro_table['ALT'])
    obstime = astro_table['TSTART']
    ##obstime = astro_table['TSTART'] + astro_table['TSTOP']
    ##obstime = Time(astro_table['TSTART']) + Time(astro_table['TSTOP'])
    location = observatory_locations[observatory]
    alt_az_coord = AltAz(az = az, alt = alt, obstime = obstime, location = location)
    # TODO: make it depend on other pars: temperature, pressure, humidity,...
    sky_coord = alt_az_coord.transform_to(FK5)
    ra = sky_coord.ra
    col_ra = Column(name='RA', data=ra)
    astro_table.add_column(col_ra)
    dec = sky_coord.dec
    col_dec = Column(name='DEC', data=dec)
    astro_table.add_column(col_dec)

    obs_table = ObservationTable(astro_table)

#    #t1 = Time('2010-01-01 00:00:00')
#    #t2 = Time('2010-02-01 00:00:00')
#    #dt = t2 - t1
#    #dt
#    #print(dt)
#    #print(dt.iso)
#
#
#    #from astropy.time import TimeDelta
###    from astropy.utils.data import download_file
###    from astropy.utils import iers
###    from astropy.coordinates import builtin_frames
###    iers.IERS.iers_table = iers.IERS_A.open(download_file(iers.IERS_A_URL, cache=True)) [builtin_frames.utils]
#
#    #dtss = astro_table['TSTART'] + astro_table['TSTOP']
#    #dtss = Time(astro_table['TSTART']) + Time(astro_table['TSTOP'])

    return obs_table
