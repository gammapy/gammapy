# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
import astropy.units as u
from . import (
    CountsSpectrum,
    PHACountsSpectrum,
    SpectrumObservation,
    SpectrumObservationList,
)
from .results import SpectrumStats
from ..extern.bunch import Bunch
from ..extern.pathlib import Path
from ..image import ExclusionMask
from ..irf import EffectiveAreaTable, EnergyDispersion
from ..region import SkyCircleRegion, find_reflected_regions
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import make_path, write_yaml

__all__ = [
    'SpectrumExtraction',
]

log = logging.getLogger(__name__)


class SpectrumExtraction(object):
    """Class for creating input data to 1D spectrum fitting

    This class takes a `~gammapy.data.Target` as input and creates 1D counts on
    and off counts vectors as well as an effective area vector and an energy
    dispersion matrix.  For more info see :ref:`spectral_fitting`.

    Parameters
    ----------
    on_region : `~gammapy.extern.regions.SkyRegion`
        On region
    obs: `~gammapy.data.ObservationList`
        Observations to process
    background : `~gammapy.data.BackgroundEstimate`, dict
        Background estimate or dict of parameters
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning

    Examples
    --------
    """
    OGIP_FOLDER = 'ogip_data'
    """Folder that will contain the output ogip data"""

    def __init__(self, target, e_reco=None):
        self.target = target
        # This is the 14 bpd setup used in HAP Fitspectrum
        self.e_reco = e_reco or np.logspace(-2, 2, 96) * u.TeV
        self._observations = None

    def estimate_background(selfi, *args, **kwargs):
        self.background = BackgroundEstimate(...)

    @property
    def observations(self):
        """List of `~gammapy.spectrum.SpectrumObservation`

        This list is generated via
        :func:`~gammapy.spectrum.spectrum_extraction.extract_spectrum`
        when the property is first called and the result is cached.
        """
        if self._observations is None:
            self.extract_spectrum()
        return self._observations

    def run(self, outdir=None):
        """Run all steps

        Extract spectrum, update observation table, filter observations,
        write results to disk.

        Parameters
        ----------
        outdir : Path, str
            directory to write results files to
        """
        cwd = Path.cwd()
        outdir = cwd if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        os.chdir(str(outdir))
        self.write()
        os.chdir(str(cwd))

    def filter_observations(self):
        """Filter observations by number of reflected regions"""
        n_min = self.bkg_method['n_min']
        obs = self.observations
        mask = obs.filter_by_reflected_regions(n_min)
        inv_mask = np.where([_ not in mask for _ in np.arange(len(mask + 1))])
        excl_obs = self.obs_table[inv_mask[0]]['OBS_ID'].data
        log.info('Excluding obs {} : Found less than {} reflected '
                 'region(s)'.format(excl_obs, n_min))
        self._observations = SpectrumObservationList(np.asarray(obs)[mask])
        self.obs_table = self.obs_table[mask]

    def extract_spectrum(self):
        """Extract 1D spectral information

        The result can be obtained via
        :func:`~gammapy.spectrum.spectrum_extraction.observations`
        """
        spectrum_observations = []
        if self.target.background is None:
            raise ValueError("No background estimate for target {}".format(self.target))
        for obs, bkg in zip(self.target.obs, self.target.background):
            log.info('Extracting spectrum for observation {}'.format(obs))
            idx = self.target.on_region.contains(obs.events.radec)
            on_events = obs.events[idx]

            counts_kwargs = dict(energy=self.e_reco,
                                 exposure = obs.observation_live_time_duration,
                                 obs_id=obs.obs_id)

            on_vec = PHACountsSpectrum(backscal=bkg.a_on, **counts_kwargs)
            off_vec = PHACountsSpectrum(backscal=bkg.a_off, is_bkg=True, **counts_kwargs)

            on_vec.fill(on_events)
            off_vec.fill(bkg.off_events)

            offset = obs.pointing_radec.separation(self.target.position)
            arf = obs.aeff.to_effective_area_table(offset)
            rmf = obs.edisp.to_energy_dispersion(offset, e_reco=self.e_reco)

            temp = SpectrumObservation(on_vec, off_vec, arf, rmf)
            spectrum_observations.append(temp)

        self._observations = SpectrumObservationList(spectrum_observations)

    def write(self):
        """Write results to disk"""
        self.observations.write(self.OGIP_FOLDER)
        # TODO : add more debug plots etc. here

