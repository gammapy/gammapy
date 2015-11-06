# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
import logging
import os
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from astropy.io import fits
from gammapy.extern.pathlib import Path
from gammapy.region import SkyCircleRegion
from ..background import ring_area_factor
from ..data import CountsSpectrum
from ..obs import DataStore
from ..spectrum import EnergyBounds, Energy
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = ['SpectrumAnalysis', 'SpectrumObservation']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(SpectrumAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error',
                                 'critical'],
                        help="Set the logging level")

    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    analysis = SpectrumAnalysis.from_yaml(args.config_file)
    analysis.run()


class SpectrumAnalysis(object):
    """Perform a 1D spectrum fit
    """

    def __init__(self, config):
        self.config = config
        log.info('Creating analysis ' + config['general']['outdir'])

        runs = config['general']['runlist']
        if isinstance(runs, six.string_types):
            runs = np.loadtxt(runs, dtype=np.int)

        storename = self.config['general']['datastore']
        store = DataStore.from_name(storename)

        nruns = self.config['general']['nruns'] - 1
        self.observations = []
        for i, obs in enumerate(runs):
            try:
                val = SpectrumObservation(obs, store, config)
            except IndexError:
                log.warn('Run {} not in store {}'.format(obs, store.name))
                nruns = nruns + 1
                continue
            self.observations.append(val)
            if i == nruns:
                break

    @classmethod
    def from_yaml(cls, filename):
        """Read config from YAML file.

        Parameters
        ----------
        filename : str
            YAML config file
        """
        import yaml
        log.info('Reading {}'.format(filename))
        with open(filename) as fh:
            config = yaml.safe_load(fh)
        return cls(config)

    def run(self):
        """Run analysis as specified in the config"""
        if self.config['general']['create_ogip']:
            self.prepare_ogip_data()
        if self.config['general']['run_fit']:
            model = self.config['model']['type']
            thres_low = Energy(self.config['model']['threshold_low'])
            thres_high = Energy(self.config['model']['threshold_high'])
            fit = self.run_hspec_fit(model, thres_low, thres_high)
            return fit

    def prepare_ogip_data(self):
        """Create OGIP files"""
        for obs in self.observations:
            obs.prepare_ogip_data()
            log.info('Creating OGIP data for run{}'.format(obs.obs))

    def run_hspec_fit(self, model, thres_low, thres_high):
        """Run the gammapy.hspec fit

        Parameters
        ----------
        model : str
            Sherpa model
        thres_high : `~gammapy.spectrum.Energy`
            Upper threshold of the spectral fit
        thres_low : `~gammapy.spectrum.Energy`
            Lower threshold of the spectral fit
        """

        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat
        from sherpa.models import PowLaw1D

        if (model == 'PL'):
            p1 = PowLaw1D('p1')
            p1.gamma = 2.2
            p1.ref = 1e9
            p1.ampl = 6e-19
        else:
            raise ValueError('Desired Model is not defined')

        thres = thres_low.to('keV').value
        emax = thres_high.to('keV').value

        sau.freeze(p1.ref)
        sau.set_conf_opt("max_rstat", 100)

        list_data = []
        for obs in self.observations:
            datid = obs.phafile.parts[-1][7:12]
            sau.load_data(datid, str(obs.phafile))
            sau.notice_id(datid, thres, emax)
            sau.set_source(datid, p1)
            list_data.append(datid)
        wstat.wfit(list_data)
        sau.covar()
        fit_val = sau.get_covar_results()
        fit_attrs = ('parnames', 'parvals', 'parmins', 'parmaxes')
        fit = dict((attr, getattr(fit_val, attr)) for attr in fit_attrs)
        fit = self.apply_containment(fit)
        sau.clean()
        return fit

    def apply_containment(self, fit):
        """Apply correction factor for PSF containment in ON region"""
        cont = self.get_containment()
        fit['containment'] = cont
        fit['parvals'] = list(fit['parvals'])
        fit['parvals'][1] = fit['parvals'][1] * cont
        return fit

    def get_containment(self):
        """Calculate PSF correction factor for containment in ON region"""
        return 1


class SpectrumObservation(object):
    """1D region based spectral analysis observation.

    This class handles the spectrum fit for one observation/run
    At the moment it can only be initialized with a yaml config file

    TODO: Link to example

    Parameters
    ----------
    obs : int
        Observation ID, runnumber
    store : `~gammapy.obs.DataStore
        Data Store
    config : dict
        Configuration, provided by YAML config file
    """

    def __init__(self, obs, store, config):

        # Raises Error if obs is not available
        store.filename(obs, 'events')
        self.config = config
        self.obs = obs
        self.store = store
        self._process_config()
        self.event_list = None
        self.pha = None
        self.bkg = None
        self.arf = None
        self.bkg = None
        self.phafile = None

    def make_on_vector(self):
        """Create ON vector

        Returns
        -------
        on_vec : `gammapy.data.CountsSpectrum`
            Counts spectrum inside the ON region
        """
        self._load_event_list()
        on_list = self.event_list.select_circular_region(self.on_region)
        on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
        self.pha = on_vec
        return on_vec

    def make_off_vector(self):
        """Create OFF vector

        Returns
        -------
        on_vec : `gammapy.data.CountsSpectrum`
            Counts spectrum inside the OFF region
        """
        self._load_event_list()
        if self.off_type == "ring":
            # TODO put in utils once there is a SkyRingRegion
            center = self.on_region.pos
            radius = self.on_region.radius
            inner = self.off_region['inner_radius']
            outer = self.off_region['outer_radius']
            off_list = self.event_list.select_sky_ring(center, inner, outer)
            off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
            alpha = ring_area_factor(radius.deg, inner.deg, outer.deg)

        elif self.off_type == "reflected":
            raise NotImplementedError
        else:
            raise ValueError(
                "Undefined background method: {}".format(self.off_type))

        off_vec.backscal = alpha
        self.bkg = off_vec
        return off_vec

    def make_arf(self):
        """Create effective area vector correct energy binning

        Returns
        -------
        arf : `~gammapy.irf.EffectiveAreaTable`
             effective area vector
        """
        self._load_event_list()
        aeff2D = self.store.load(self.obs, 'aeff')
        arf_vec = aeff2D.to_effective_area_table(self.offset)
        self.arf = arf_vec
        return arf_vec

    def make_rmf(self):
        """Create energy disperion matrix in correct energy binning

        Returns
        -------
        rmf : `~gammapy.irf.EnergyDispersion`
            energy dispersion matrix
        """
        edisp2D = self.store.load(self.obs, 'edisp')
        rmf_mat = edisp2D.to_energy_dispersion(self.offset,
                                               e_reco=self.ebounds)
        self.rmf = rmf_mat
        return rmf_mat

    def to_ogip(self, phafile=None, bkgfile=None, rmffile=None, arffile=None,
                outdir=None, clobber=True):
        """Write OGIP files

        Only those objects are written have been created with the appropriate
        functions before

        Parameters
        ----------
        phafile : str
            PHA filename
        bkgfile : str
            BKG filename
        arffile : str
            ARF filename
        rmffile : str
            RMF : filename
        outdir : None
            directory to write the files to
        clobber : bool
            Overwrite
        """

        if outdir is None:
            outdir = "ogip_data"

        basedir = Path(outdir)
        basedir.mkdir(exist_ok=True)

        if arffile is None:
            arffile = basedir / "arf_run{}.fits".format(self.obs)
        if rmffile is None:
            rmffile = basedir / "rmf_run{}.fits".format(self.obs)
        if phafile is None:
            phafile = basedir / "pha_run{}.pha".format(self.obs)
        if bkgfile is None:
            bkgfile = basedir / "bkg_run{}.pha".format(self.obs)

        self.phafile = phafile

        if self.pha is not None:
            self.pha.write(str(phafile), bkg=str(bkgfile), arf=str(arffile),
                           rmf=str(rmffile), clobber=clobber)
        if self.bkg is not None:
            self.bkg.write(str(bkgfile), clobber=clobber)
        if self.arf is not None:
            self.arf.write(str(arffile), energy_unit='keV', effarea_unit='cm2',
                           clobber=clobber)
        if self.rmf is not None:
            self.rmf.write(str(rmffile), energy_unit='keV', clobber=clobber)

    def prepare_ogip_data(self):
        """Perform all step to provide the OGIP data for a sherpa fit
        """
        self.make_on_vector()
        self.make_off_vector()
        self.make_arf()
        self.make_rmf()
        self.to_ogip()

    def _load_event_list(self):
        """Load event list associated with the observation

        If the event list is already loaded pass
        """
        if self.event_list is None:
            self.event_list = self.store.load(obs_id=self.obs, filetype='events')
            pointing = self.event_list.pointing_radec
            self.offset = pointing.separation(self.on_region.pos)
        else:
            pass

    def _process_config(self):
        """Convert config file info into objects
        """
        # Binning
        sec = self.config['binning']
        if sec['equal_log_spacing']:
            emin = Energy(sec['emin'])
            emax = Energy(sec['emax'])
            nbins = sec['nbins']
            self.ebounds = EnergyBounds.equal_log_spacing(
                emin, emax, nbins)
        else:
            if sec['binning'] is None:
                raise ValueError("No binning specified")
        log.debug('Binning: {}'.format(self.ebounds))

        # ON region
        radius = Angle(self.config['on_region']['radius'])
        x = self.config['on_region']['center_x']
        y = self.config['on_region']['center_y']
        frame = self.config['on_region']['system']
        center = SkyCoord(x, y, frame=frame)
        self.on_region = SkyCircleRegion(center, radius)

        # OFF region
        self.off_type = self.config['off_region']['type']
        if self.off_type == 'ring':
            irad = Angle(self.config['off_region']['inner_radius'])
            orad = Angle(self.config['off_region']['outer_radius'])
            self.off_region = dict(type='ring', inner_radius=irad,
                                   outer_radius=orad)
        elif self.off_type == 'reflected':
            pass

    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass


def _process_config(object):
    """Helper function to process the config file
    """

    # Data
    storename = object.config['general']['datastore']
    object.store = DataStore.from_name(storename)
    object.outdir = object.config['general']['outdir']
    basename = object.outdir + "/ogip_data"

    # TODO: use Path here (see Developer HOWTO entry why / how).
    if not os.path.isdir(object.outdir):
        os.mkdir(object.outdir)
        os.mkdir(basename)
    object.arffile = basename + "/arf_run" + str(object.obs) + ".fits"
    object.rmffile = basename + "/rmf_run" + str(object.obs) + ".fits"
    object.phafile = basename + "/pha_run" + str(object.obs) + ".pha"
    object.bkgfile = basename + "/bkg_run" + str(object.obs) + ".pha"

    # Target
    x = Angle(object.config['on_region']['center_x'])
    y = Angle(object.config['on_region']['center_y'])
    frame = object.config['on_region']['system']
    object.target = SkyCoord(x, y, frame=frame)

    # Pointing
    event_list = object.store.load(obs_id=object.obs, filetype='events')
    object.event_list = event_list
    object.pointing = object.event_list.pointing_radec
    object.offset = object.target.separation(object.pointing)

    # Excluded regions
    excl_file = object.config['excluded_regions']['file']
    object.exclusion = fits.open(excl_file)[0]

    # Binning
    sec = object.config['binning']
    if sec['equal_log_spacing']:
        emin = Energy(sec['emin'])
        emax = Energy(sec['emax'])
        nbins = sec['nbins']
        object.ebounds = EnergyBounds.equal_log_spacing(
            emin, emax, nbins)
    else:
        if sec[binning] is None:
            raise ValueError("No binning specified")
    log.debug('Binning: {}'.format(object.ebounds))

    # ON/OFF Region
    val = object.config['on_region']['radius']
    object.radius = Angle(val)

    object.off_type = object.config['off_region']['type']
    if object.off_type == 'ring':
        ival = object.config['off_region']['inner_radius']
        oval = object.config['off_region']['outer_radius']
        object.irad = Angle(ival)
        object.orad = Angle(oval)
        object.alpha = ring_area_factor(object.radius, object.irad,
                                        object.orad).value
    elif object.off_type == 'reflected':
        pass

    # Spectral fit
    object.model = object.config['model']['type']
    val = object.config['model']['threshold_low']
    val2 = object.config['model']['threshold_high']
    threshold = Energy(val)
    threshold2 = Energy(val2)
    object.thres = threshold.to('keV').value
    object.emax = threshold2.to('keV').value
