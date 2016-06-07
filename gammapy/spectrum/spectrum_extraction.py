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
from ..data import Target
from ..background import (
    BackgroundEstimate,
    reflected_regions_background_estimate,
)
from ..extern.bunch import Bunch
from ..extern.pathlib import Path
from ..extern.regions.shapes import CircleSkyRegion
from ..image import ExclusionMask
from ..irf import EffectiveAreaTable, EnergyDispersion
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
    target : `~gammapy.data.Target` or `~gammapy.extern.regions.SkyRegion`
        Observation target
    obs: `~gammapy.data.ObservationList`
        Observations to process
    background : `~gammapy.data.BackgroundEstimate` or dict
        Background estimate or dict of parameters
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning

    Examples
    --------
    """
    OGIP_FOLDER = 'ogip_data'
    """Folder that will contain the output ogip data"""
    def __init__(self, target, obs, background, e_reco=None, e_true=None):
        if isinstance(target, CircleSkyRegion):
            target = Target(target)
        self.obs = obs
        self.background=background
        self.target = target
        # This is the 14 bpd setup used in HAP Fitspectrum
        self.e_reco = e_reco or np.logspace(-2, 2, 96) * u.TeV
        self.e_true = e_true or np.logspace(-2, 2.3, 250) * u.TeV
        self._observations = None

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
        if not isinstance(self.background, BackgroundEstimate):
            log.info('Estimate background with config {}'.format(self.background))
            self.estimate_background()
        self.extract_spectrum()
        self.write()
        os.chdir(str(cwd))

    def estimate_background(self):
        method = self.background.pop('method')
        if method == 'reflected':
            exclusion = self.background.pop('exclusion', None)
            bkg = [reflected_regions_background_estimate(
                self.target.on_region,
                _.pointing_radec,
                exclusion,
                _.events) for _ in self.obs]
        else:
            raise NotImplementedError("Method: {}".format(method))
        self.background = bkg
        
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
        if not isinstance(self.background, list):
            raise ValueError("Invalid background estimate: {}".format(self.background))
        for obs, bkg in zip(self.obs, self.background):
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

            offset = obs.pointing_radec.separation(self.target.on_region.center)
            arf = obs.aeff.to_effective_area_table(offset, energy=self.e_true)
            rmf = obs.edisp.to_energy_dispersion(offset,
                                                 e_reco=self.e_reco,
                                                 e_true=self.e_true)

            temp = SpectrumObservation(on_vec, off_vec, arf, rmf)
            spectrum_observations.append(temp)

        self._observations = SpectrumObservationList(spectrum_observations)

    def define_ethreshold(self, method_lo_threshold=None, func_lo_threshold=None , **kwargs):
        """Set the hi and lo Ethreshold for each observation based on implemented method in gammapy or on a function that
        you define on the IRFs on each observations and that take an observation object as parameters.

        Parameters
        ----------
        method_lo_threshold : {"AreaMax", "Myfunc"}
            method implemented to define a low energy threshold
        func_lo_threshold : function name
            Name of the function you define on the IRFs of the observation to define a low energy threshold
        kwargs : argument to the defined method or the function

        """
        for obs in self._observations:

            if method_lo_threshold == "AreaMax":
                self.on_vector.lo_threshold = obs.effective_area.area_max(**kwargs)
            elif method_lo_threshold == "Myfunc":
                if not func_lo_threshold:
                    log.info('You have to give a function do define the energy threshold')
                    break
                else:
                    self.on_vector.lo_threshold = func_lo_threshold(obs, **kwargs)
            elif not method_lo_threshold:
                log.info('You have to give a method name to define the energy threshold')
                break

    def write(self):
        """Write results to disk"""
        self.observations.write(self.OGIP_FOLDER)
        # TODO : add more debug plots etc. here
