# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from . import EnergyDependentTablePSF, IRFStacker, EffectiveAreaTable

__all__ = [
    "make_psf",
    "make_mean_psf",
    "make_mean_edisp",
    "apply_containment_fraction",
    "compute_energy_thresholds",
]

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


def make_mean_edisp(
    observations,
    position,
    e_true,
    e_reco,
    low_reco_threshold="0.002 TeV",
    high_reco_threshold="150 TeV",
):
    """Compute mean energy dispersion.

    Compute the mean edisp of a set of observations j at a given position

    The stacking is implemented in :func:`~gammapy.irf.IRFStacker.stack_edisp`

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
        Observations for which to compute the EDISP
    position : `~astropy.coordinates.SkyCoord`
        Position at which to compute the EDISP
    e_true : `~astropy.units.Quantity`
        True energy axis
    e_reco : `~astropy.units.Quantity`
        Reconstructed energy axis
    low_reco_threshold : `~astropy.units.Quantity`
        low energy threshold in reco energy
    high_reco_threshold : `~astropy.units.Quantity`
        high energy threshold in reco energy

    Returns
    -------
    stacked_edisp : `~gammapy.irf.EnergyDispersion`
        Stacked EDISP for a set of observation
    """
    low_reco_threshold = u.Quantity(low_reco_threshold)
    high_reco_threshold = u.Quantity(high_reco_threshold)

    list_aeff = []
    list_edisp = []
    list_livetime = []
    list_low_threshold = [low_reco_threshold] * len(observations)
    list_high_threshold = [high_reco_threshold] * len(observations)

    for obs in observations:
        offset = position.separation(obs.pointing_radec)
        list_aeff.append(obs.aeff.to_effective_area_table(offset, energy=e_true))
        list_edisp.append(
            obs.edisp.to_energy_dispersion(offset, e_reco=e_reco, e_true=e_true)
        )
        list_livetime.append(obs.observation_live_time_duration)

    irf_stack = IRFStacker(
        list_aeff=list_aeff,
        list_edisp=list_edisp,
        list_livetime=list_livetime,
        list_low_threshold=list_low_threshold,
        list_high_threshold=list_high_threshold,
    )
    irf_stack.stack_edisp()

    return irf_stack.stacked_edisp


def apply_containment_fraction(aeff, psf, radius):
    """Estimate PSF containment inside a given radius and correct effective area for leaking flux.

    The PSF and effective area must have the same binning in energy.

    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable`
        the input 1D effective area
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        the input 1D PSF
    radius : `~astropy.coordinates.Angle`
        the maximum angle

    Returns
    -------
    correct_aeff : `~gammapy.irf.EffectiveAreaTable`
        the output corrected 1D effective area
    """
    energy_center = aeff.energy.center
    energy_edges = aeff.energy.edges

    containment = psf.containment(energy_center, radius)

    corrected_aeff = EffectiveAreaTable(
        energy_lo=energy_edges[:-1],
        energy_hi=energy_edges[1:],
        data=aeff.data.data * np.squeeze(containment),
        meta=aeff.meta,
    )
    return corrected_aeff


def compute_energy_thresholds(
    aeff, edisp, method_lo="none", method_hi="none", **kwargs
):
    """Compute safe energy thresholds from 1D energy dispersion and effective area.

    Set the high and low energy threshold based on a chosen method.
    For now the methods return thresholds assuming true and reco energy are comparable.

    Available methods for setting the low energy threshold:

        * area_max : Set energy threshold at x percent of the maximum effective
          area (x given as kwargs['area_percent_lo'])

        * energy_bias : Set energy threshold at energy where the energy bias
          exceeds a value of x percent (given as kwargs['bias_percent_lo'])

        * none : Do not apply a lower threshold

    Available methods for setting the high energy threshold:

        * area_max : Set energy threshold at x percent of the maximum effective
          area (x given as kwargs['area_percent_hi']). The threshold is searched
          in the last true energy decade of the effective area.

        * energy_bias : Set energy threshold at energy where the energy bias
          exceeds a value of x percent (given as kwargs['bias_percent_hi']).
          The threshold is searched in the last true energy decade of the
          energy dispersion.

        * none : Do not apply a higher energy threshold

    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable`
        the 1D effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        the energy dispersion used
    method_lo : {'area_max', 'energy_bias', 'none'}
        Method for defining the low energy threshold
    method_hi : {'area_max', 'energy_bias', 'none'}
        Method for defining the high energy threshold
    """

    # Low threshold
    if method_lo == "area_max":
        aeff_thres = kwargs["area_percent_lo"] / 100 * aeff.max_area
        thres_lo = aeff.find_energy(aeff_thres)
    elif method_lo == "energy_bias":
        thres_lo = edisp.get_bias_energy(kwargs["bias_percent_lo"] / 100)
    elif method_lo == "none":
        thres_lo = aeff.energy.edges[0]
    else:
        raise ValueError("Invalid method_lo: {}".format(method_lo))

    # High threshold
    if method_hi == "area_max":
        aeff_thres = kwargs["area_percent_hi"] / 100 * aeff.max_area
        e_max = aeff.energy.edges[-1]
        try:
            thres_hi = aeff.find_energy(aeff_thres, emin=0.1 * e_max, emax=e_max)
        except ValueError:
            thres_hi = e_max
    elif method_hi == "energy_bias":
        e_max = aeff.energy.edges[-1]
        try:
            thres_hi = edisp.get_bias_energy(
                kwargs["bias_percent_hi"] / 100, emin=0.1 * e_max, emax=e_max
            )
        except ValueError:
            thres_hi = e_max
    elif method_hi == "none":
        thres_hi = aeff.energy.edges[-1]
    else:
        raise ValueError("Invalid method_hi: {}".format(method_hi))

    return thres_lo, thres_hi
