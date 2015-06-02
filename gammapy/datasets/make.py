# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Make example datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.time import Time
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, AltAz, Angle
from ..irf import EnergyDependentMultiGaussPSF
from ..obs import ObservationTable, observatory_locations

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
    #       pointing observation -> done
    #       and date -> done
    #       then convert to alt az, considering the observatory location
    # TODO: add column(s) for offset, and take it into account in the coord. transformation!!!

    # RA, Dec
    # random points on a sphere ref: http://mathworld.wolfram.com/SpherePointPicking.html

    ra = Angle(360.*np.random.random(len(col_obs_id)), 'degree')
    col_ra = Column(name='RA', data=ra)
    astro_table.add_column(col_ra)

    dec = Angle(np.arccos(2.*np.random.random(len(col_obs_id)) - 1), 'radian').to('degree')
    # translate angles from [0, 180) deg to [-90, 90) deg
    dec = dec - Angle(90., 'degree')
    col_dec = Column(name='DEC', data=dec)
    astro_table.add_column(col_dec)

    # date
    # random points between the start of 2010 and the end of 2014
    #TODO: should this represent the time at the beginning of the run?
    #      this has consequences for the ra/dec -> alt/az conversion
    datestart = Time('2010-01-01T00:00:00', format='fits', scale='utc')
    dateend = Time('2015-01-01T00:00:00', format='fits', scale='utc')
    date = Time((dateend.mjd - datestart.mjd)*np.random.random(len(col_obs_id)) + datestart.mjd, format='mjd', scale='utc').fits
    col_date = Column(name='DATE', data=date)
    astro_table.add_column(col_date)


    # alt, az
    # TODO: should they be in the ObservationTable? (they are derived quantities, like dead time)
    # TODO: since I randomized RA/DEC without taking into account the observatory, I'm getting negative altitudes!!!
    #       maybe I should randomize alt az, then transform to RA/DEC!!!
    observatory_location = observatory_locations[observatory]

    # ref: http://astropy.readthedocs.org/en/latest/coordinates/observing-example.html
    sky_coord = SkyCoord(astro_table['RA'], astro_table['DEC'], frame='icrs')
    alt_az_coord = sky_coord.transform_to(AltAz(obstime=astro_table['DATE'], location=observatory_location))
    #print(alt_az_coord)
    #col_alt = alt_az_coord...


    # TODO: operate row by row (using Observation class) instead of by columns?
    # TODO: general methods for filling obs tables run by run; this function should call it.

    obs_table = ObservationTable(astro_table)

    return obs_table
