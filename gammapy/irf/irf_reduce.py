# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from .psf_table import EnergyDependentTablePSF

__all__ = ["make_psf", "make_mean_psf"]

log = logging.getLogger(__name__)


def make_psf(observation, position, energy=None, rad=None):
    """Make energy-dependent PSF for a given source position.

    Parameters
    ----------
    observation : `~gammapy.data.DataStoreObservation`
        Observation for which to compute the PSF
    position : `~astropy.coordinates.SkyCoord`
        Position at which to compute the PSF
    energy : `~astropy.units.Quantity`
        1-dim energy array for the output PSF.
        If none is given, the energy array of the PSF from the observation is used.
    rad : `~astropy.coordinates.Angle`
        1-dim offset wrt source position array for the output PSF.
        If none is given, the offset array of the PSF from the observation is used.

    Returns
    -------
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        Energy dependent psf table
    """
    offset = position.separation(observation.pointing_radec)

    if energy is None:
        energy = observation.psf.to_energy_dependent_table_psf(theta=offset).energy

    if rad is None:
        rad = observation.psf.to_energy_dependent_table_psf(theta=offset).rad

    psf_value = observation.psf.to_energy_dependent_table_psf(
        theta=offset, rad=rad
    ).evaluate(energy)

    arf = observation.aeff.data.evaluate(offset=offset, energy=energy)
    exposure = arf * observation.observation_live_time_duration

    psf = EnergyDependentTablePSF(
        energy=energy, rad=rad, exposure=exposure, psf_value=psf_value
    )
    return psf


def make_mean_psf(observations, position, energy=None, rad=None):
    """Compute mean energy-dependent PSF.

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
        Observations for which to compute the PSF
    position : `~astropy.coordinates.SkyCoord`
        Position at which to compute the PSF
    energy : `~astropy.units.Quantity`
        1-dim energy array for the output PSF.
        If none is given, the energy array of the PSF from the first
        observation is used.
    rad : `~astropy.coordinates.Angle`
        1-dim offset wrt source position array for the output PSF.
        If none is given, the energy array of the PSF from the first
        observation is used.

    Returns
    -------
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        Mean PSF
    """
    for idx, observation in enumerate(observations):
        psf = make_psf(observation, position, energy, rad)
        if idx == 0:
            stacked_psf = psf
        else:
            stacked_psf = stacked_psf.stack(psf)
    return stacked_psf
