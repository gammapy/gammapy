# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
import astropy.units as u
from regions import CircleSkyRegion
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..data import Target
from ..background import ReflectedRegionsBackgroundEstimator
from .core import PHACountsSpectrum
from .observation import SpectrumObservation, SpectrumObservationList

__all__ = [
    'SpectrumExtraction',
]

log = logging.getLogger(__name__)


class SpectrumExtraction(object):
    """Class for creating input data to 1D spectrum fitting

    This class is responsible for extracting a
    `~gammapy.spectrum.SpectrumObservation` from a
    `~gammapy.data.DataStoreObservation`, given a certain signal extraction
    region. A background estimate can be passed on initialization or created on
    the fly given a dict of parameters.  For point sources analyzed with 'full
    containement' IRFs, a correction for PSF leakage out of the circular ON
    region can be applied.  For more info see :ref:`spectral_fitting`.

    Parameters
    ----------
    target : `~gammapy.data.Target` or `~regions.SkyRegion`
        Signal extraction region
    obs : `~gammapy.data.ObservationList`
        Observations to process
    background : `~gammapy.background.BackgroundEstimate` or dict
        Background estimate or dict of parameters
    e_reco : `~astropy.units.Quantity`, optional
        Reconstructed energy binning
    e_true : `~astropy.units.Quantity`, optional
        True energy binning
    containment_correction : bool
        Apply containment correction for point sources and circular ON regions.

    Examples
    --------
    TODO

    """
    OGIP_FOLDER = 'ogip_data'
    """Folder that will contain the output ogip data"""
    DEFAULT_TRUE_ENERGY = np.logspace(-2, 2.5, 109) * u.TeV
    """True energy axis to be used if not specified otherwise"""
    DEFAULT_RECO_ENERGY = np.logspace(-2, 2, 73) * u.TeV
    """Reconstruced energy axis to be used if not specified otherwise"""

    # NOTE : The default true binning is not used in the test due to
    # extrapolation issues

    def __init__(self, target, obs, background, e_reco=None, e_true=None,
                 containment_correction=False):

        if isinstance(target, CircleSkyRegion):
            target = Target(target)
        self.obs = obs
        self.background = background
        self.target = target
        self.e_reco = e_reco or self.DEFAULT_RECO_ENERGY
        self.e_true = e_true or self.DEFAULT_TRUE_ENERGY
        self._observations = None
        self.containment_correction = containment_correction
        if self.containment_correction and not isinstance(target.on_region,
                                                          CircleSkyRegion):
            raise TypeError("Incorrect region type for containment correction."
                            " Should be CircleSkyRegion.")

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
        if not isinstance(self.background, list):
            log.info('Estimate background with config {}'.format(self.background))
            self.estimate_background(self.background)
        self.extract_spectrum()
        self.write()
        os.chdir(str(cwd))

    def estimate_background(self, config):
        """Create `~gammapy.background.BackgroundEstimate`

        In case no background estimate was passed on initialization, this
        method creates one given a dict of parameters. 

        TODO: Link to high-level docs page.

        Parameters
        ----------
        config : dict
            Background estimation method
        """
        method = self.background.pop('method')
        if method == 'reflected':
            kwargs = self.background.copy()
            kwargs.pop('n_min', None)
            refl = ReflectedRegionsBackgroundEstimator(
                on_region=self.target.on_region,
                obs_list=self.obs,
                **kwargs) 
            refl.run()
            self.background = refl.result
        else:
            raise NotImplementedError("Method: {}".format(method))

    def filter_observations(self):
        """Filter observations by number of reflected regions"""
        raise NotImplementedError("broken")
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
        log.info('Starting spectrum extraction')
        spectrum_observations = []
        if not isinstance(self.background, list):
            raise ValueError("Invalid background estimate: {}".format(self.background))
        for obs, bkg in zip(self.obs, self.background):
            log.info('Extracting spectrum for observation\n {}'.format(obs))
            offset = obs.pointing_radec.separation(self.target.on_region.center)
            log.info('Offset : {}\n'.format(offset))

            idx = self.target.on_region.contains(obs.events.radec)
            on_events = obs.events[idx]

            counts_kwargs = dict(energy=self.e_reco,
                                 livetime=obs.observation_live_time_duration,
                                 obs_id=obs.obs_id)

            # We now add a number of optional keywords for the DataStoreObservation
            # We first check that the entry exists in the table
            try:
                counts_kwargs.update(tstart=obs.tstart)
            except KeyError:
                pass
            try:
                counts_kwargs.update(tstop=obs.tstop)
            except KeyError:
                pass
            try:
                counts_kwargs.update(muoneff=obs.muoneff)
            except KeyError:
                pass
            try:
                counts_kwargs.update(zen_pnt=obs.pointing_zen)
            except KeyError:
                pass

            on_vec = PHACountsSpectrum(backscal=bkg.a_on, **counts_kwargs)
            off_vec = PHACountsSpectrum(backscal=bkg.a_off, is_bkg=True,
                                        **counts_kwargs)

            on_vec.fill(on_events)
            off_vec.fill(bkg.off_events)

            arf = obs.aeff.to_effective_area_table(offset, energy=self.e_true)
            rmf = obs.edisp.to_energy_dispersion(offset,
                                                 e_reco=self.e_reco,
                                                 e_true=self.e_true)

            # If required, correct arf for psf leakage
            # TODO: write correction factor as AREASCAL column in PHAFILE
            if self.containment_correction:
                # First need psf
                angles = np.linspace(0., 1.5, 150) * u.deg
                psf = obs.psf.to_table_psf(offset, angles)

                center_energies = arf.energy.nodes
                for index, energy in enumerate(center_energies):
                    try:
                        correction = psf.integral(energy,
                                                  0. * u.deg,
                                                  self.target.on_region.radius)
                    except:
                        # TODO: Why is this necessary?
                        correction = np.nan

                    arf.data[index] = arf.data[index] * correction

            temp = SpectrumObservation(on_vector=on_vec,
                                       aeff=arf,
                                       off_vector=off_vec,
                                       edisp=rmf)

            temp.hi_threshold=obs.aeff.high_threshold
            temp.lo_threshold=obs.aeff.low_threshold

            spectrum_observations.append(temp)

        self._observations = SpectrumObservationList(spectrum_observations)

    def define_energy_threshold(self, method_lo_threshold='area_max', **kwargs):
        """Set energy threshold

        Set the high and low energy threshold for each observation based on a
        chosen method.

        Available methods for setting the low energy threshold

        * area_max : Set energy threshold at x percent of the maximum effective
                     area (x given as kwargs['percent'])

        Available methods for setting the high energy threshold

        * TBD

        Parameters
        ----------
        method_lo_threshold : {'area_max'}
            method for defining the low energy threshold
        """
        # TODO: define method for the high energy threshold

        # It is important to update the low and high threshold for ON and OFF
        # vector, otherwise Sherpa will not understand the files
        for obs in self.observations:
            if method_lo_threshold == 'area_max':
                aeff_thres = kwargs['percent'] / 100 * obs.aeff.max_area
                thres = obs.aeff.find_energy(aeff_thres)
                obs.on_vector.lo_threshold = thres
                obs.off_vector.lo_threshold = thres
            else:
                raise ValueError('Undefine method for low threshold: {}'.format(
                    method_lo_threshold))

    def write(self):
        """Write results to disk"""
        self.observations.write(self.OGIP_FOLDER)
        # TODO : add more debug plots etc. here
