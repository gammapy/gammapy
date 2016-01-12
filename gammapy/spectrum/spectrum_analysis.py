# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import (print_function)
import logging
import os
import math
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
import astropy.io.fits as pf
from ..image import ExclusionMask
from ..region import SkyCircleRegion, find_reflected_regions
from . import CountsSpectrum
from astropy.table import Table
from gammapy.extern.pathlib import Path
from ..background import ring_area_factor
from ..data import DataStore
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import (
    get_parser, set_up_logging_from_args, read_yaml, make_path,
)
from . import CountsSpectrum

__all__ = [
    'SpectrumAnalysis',
    'SpectrumObservation',
    'SpectralFit',
    'run_spectral_fit_using_config',
]

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

    config = read_yaml(args.config_file)
    run_spectral_fit_using_config(config)


class SpectrumAnalysis(object):
    """Class for 1D spectrum fitting

    Parameters
    ----------
    datastore : `~gammapy.data.DataStore`
        Data for the analysis
    obs : list, str
        List of observations or file containing such a list
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts
    exclusion : `~gammapy.image.ExclusionMask`
        Exclusion regions
    bkg_method : dict, optional
        Background method including necessary parameters
    nobs : int
        number of observations to process, 0 means all observations
    ebounds : `~gammapy.utils.energy.EnergyBounds`, optional
        Reconstructed energy binning definition
    """

    def __init__(self, datastore, obs, on_region, exclusion, bkg_method=None,
                 nobs=-1, ebounds=None):

        self.on_region = on_region
        self.store = datastore
        self.exclusion = exclusion
        if ebounds is None:
            ebounds = EnergyBounds.equal_log_spacing(0.1, 10, 20, 'TeV')
        if bkg_method is None:
            bkg_method = dict(type='no method')

        if isinstance(obs, six.string_types):
            obs = np.loadtxt(obs, dtype=np.int)

        self._observations = []
        self._numberobservations = []
        for i, val in enumerate(obs):
            try:
                temp = SpectrumObservation(val, self.store, on_region,
                                           bkg_method, ebounds, exclusion)
            except IndexError:
                log.warning(
                    'Observation {} not in store {}'.format(val, datastore))
                nobs += 1
                continue
            self._observations.append(temp)
            self._numberobservations.append(val)
            if i == nobs - 1:
                break

        if len(self.observations) == 0:
            raise ValueError("No valid observations found")

    @property
    def observations(self):
        """List of all observations belonging to the analysis
        """
        return self._observations

    @property
    def offset(self):
        """List of offsets from the observation position for all observations
        """
        off = [obs.offset for obs in self.observations]
        return off

    @property
    def reflected_regions(self, **kwargs):
        """List of dicts containing information about the reflected regions
        for each observation
        """
        retval = list([])
        for obs in self.observations:
            reflected = obs.make_reflected_regions(**kwargs)
            val = dict(obs=obs.obs, pointing=obs.pointing, region=reflected)
            retval.append(val)
        return retval

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumAnalysis` from config dict

        Parameters
        ----------
        configfile : dict
            config dict
        """

        # Observations
        obs = config['general']['runlist']
        storename = config['general']['datastore']
        store = DataStore.from_all(storename)
        nobs = config['general']['nruns']

        # Binning
        sec = config['binning']
        if sec['equal_log_spacing']:
            emin = Energy(sec['emin'])
            emax = Energy(sec['emax'])
            nbins = sec['nbins']
            ebounds = EnergyBounds.equal_log_spacing(
                    emin, emax, nbins)
        else:
            if sec['binning'] is None:
                raise ValueError("No binning specified")

        # ON region
        radius = Angle(config['on_region']['radius'])
        x = config['on_region']['center_x']
        y = config['on_region']['center_y']
        frame = config['on_region']['system']
        center = SkyCoord(x, y, frame=frame)
        on_region = SkyCircleRegion(center, radius)

        # OFF region
        off_type = config['off_region']['type']
        if off_type == 'ring':
            irad = Angle(config['off_region']['inner_radius'])
            orad = Angle(config['off_region']['outer_radius'])
            bkg_method = dict(type='ring', inner_radius=irad,
                              outer_radius=orad)
        elif off_type == 'reflected':
            bkg_method = dict(type='reflected')
        else:
            raise ValueError("Invalid background method: {}".format(off_type))

        # Exclusion
        excl_file = config['excluded_regions']['file']
        exclusion = ExclusionMask.from_fits(excl_file)

        # Outdir
        outdir = config['general']['outdir']

        return cls(datastore=store, obs=obs, on_region=on_region,
                   bkg_method=bkg_method, exclusion=exclusion,
                   nobs=nobs, ebounds=ebounds)

    @classmethod
    def from_configfile(cls, configfile):
        """Create `~gammapy.spectrum.SpectrumAnalysis` from configfile

        Parameters
        ----------
        configfile : str
            YAML config file
        """
        import yaml
        with open(configfile) as fh:
            config = yaml.safe_load(fh)

        return cls.from_config(config)

    def info(self):
        """Print some information
        """
        ss = "\nSpectrum Analysis"
        ss += "Observations : {}\n".format(len(self.observations))
        ss += "ON region    : {}\n".format(self.on_region.pos)

        return ss

    def write_ogip_data(self, outdir):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            write directory
        """

        for obs in self.observations:
            obs.write_all_ogip_data(outdir)
            log.info('Creating OGIP data for run{}'.format(obs.obs))


class SpectrumObservation(object):
    """Helper class for 1D region based spectral analysis

    This class handles the spectrum fit for one observation/run

    TODO: Link to example

    Parameters
    ----------
    obs : int
        Observation ID, runnumber
    store : `~gammapy.data.DataStore`
        Data Store
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts
    bkg_method : dict
        Background method including necessary parameters
    ebounds : `~gammapy.utils.energy.EnergyBounds`
        Reconstructed energy binning definition
    exclusion : `~gammapy.image.ExclusionMask`
        Exclusion mask
    """

    def __init__(self, obs, store, on_region, bkg_method, ebounds, exclusion):
        # Raises Error if obs is not available
        store.filename(obs, 'events')
        self.obs = obs
        self.store = store
        self.on_region = on_region
        self.bkg_method = bkg_method
        self.ebounds = ebounds
        self.exclusion = exclusion
        self._event_list = None
        self.pha = None
        self.bkg = None
        self.arf = None
        self.rmf = None

    @property
    def event_list(self):
        """`~gammapy.data.EventList` corresponding to the observation
        """
        if self._event_list is None:
            self._event_list = self.store.load(obs_id=self.obs,
                                               filetype='events')
        return self._event_list

    @property
    def pointing(self):
        """`~astropy.coordinates.SkyCoord` corresponding to the obs position
        """
        return self.event_list.pointing_radec

    @property
    def offset(self):
        """`~astropy.coordinates.Angle` corresponding to the obs offset
        """
        return self.pointing.separation(self.on_region.pos)

    def make_on_vector(self):
        """Create ON vector

        Returns
        -------
        on_vec : `gammapy.spectrum.CountsSpectrum`
            Counts spectrum inside the ON region
        """
        on_list = self.event_list.select_circular_region(self.on_region)
        on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
        self.pha = on_vec
        return on_vec

    def make_reflected_regions(self, **kwargs):
        """Create reflected off regions

        Returns
        -------
        off_region : `~gammapy.region.SkyRegionList`
            Reflected regions

        kwargs are forwarded to gammapy.region.find_reflected_regions
        """
        off_region = find_reflected_regions(self.on_region, self.pointing,
                                            self.exclusion, **kwargs)
        return off_region

    def make_off_vector(self):
        """Create off vector

        Returns
        -------
        on_vec : `gammapy.spectrum.CountsSpectrum`
            Counts spectrum inside the OFF region
        """
        if self.bkg_method['type'] == "ring":
            # TODO put in utils once there is a SkyRingRegion
            center = self.on_region.pos
            radius = self.on_region.radius
            inner = self.bkg_method['inner_radius']
            outer = self.bkg_method['outer_radius']
            off_list = self.event_list.select_sky_ring(center, inner, outer)
            alpha = ring_area_factor(radius.deg, inner.deg, outer.deg)
        elif self.bkg_method['type'] == "reflected":
            kwargs = self.bkg_method.copy()
            kwargs.pop('type')
            off = self.make_reflected_regions(**kwargs)
            off_list = self.event_list.select_circular_region(off)
            alpha = len(off)
        else:
            raise ValueError("Undefined background method: {}".format(
                    self.bkg_method['type']))

        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
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
        aeff2d = self.store.load(obs_id=self.obs, filetype='aeff')
        arf_vec = aeff2d.to_effective_area_table(self.offset)
        self.arf = arf_vec
        return arf_vec

    def make_rmf(self):
        """Create energy disperion matrix in correct energy binning

        Returns
        -------
        rmf : `~gammapy.irf.EnergyDispersion`
            energy dispersion matrix
        """
        edisp2d = self.store.load(obs_id=self.obs, filetype='edisp')
        rmf_mat = edisp2d.to_energy_dispersion(self.offset,
                                               e_reco=self.ebounds)
        self.rmf = rmf_mat
        return rmf_mat

    def write_ogip(self, phafile=None, bkgfile=None, rmffile=None, arffile=None,
                   outdir=None, clobber=True):
        """Write OGIP files

        Only those objects are written have been created with the appropriate
        functions before

        Parameters
        ----------
        phafile : `~gammapy.extern.pathlib.Path`, str
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

        outdir = make_path('ogip_data') if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True)

        if phafile is None:
            phafile = outdir / "pha_run{}.pha".format(self.obs)
        if arffile is None:
            arffile = outdir / "arf_run{}.fits".format(self.obs)
        if rmffile is None:
            rmffile = outdir / "rmf_run{}.fits".format(self.obs)
        if bkgfile is None:
            bkgfile = outdir / "bkg_run{}.fits".format(self.obs)

        if self.pha is not None:
            self.pha.write(str(phafile), bkg=str(bkgfile), arf=str(arffile),
                           rmf=str(rmffile), offset=self.offset, clobber=clobber)
        if self.bkg is not None:
            self.bkg.write(str(bkgfile), clobber=clobber)
        if self.arf is not None:
            self.arf.write(str(arffile), energy_unit='keV', effarea_unit='cm2',
                           clobber=clobber)
        if self.rmf is not None:
            self.rmf.write(str(rmffile), energy_unit='keV', clobber=clobber)

    def write_all_ogip_data(self, outdir):
        """Perform all step to provide the OGIP data for a sherpa fit

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            Directory to write to
        """
        self.make_on_vector()
        self.make_off_vector()
        self.make_arf()
        self.make_rmf()
        self.write_ogip(outdir=outdir)

    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass


class SpectralFit(object):
    """
    Spectral Fit

    Parameters
    ----------
    pha : list of str, `~gammapy.extern.pathlib.Path`
        List of PHA files to fit
    """

    def __init__(self, pha, bkg=None, arf=None, rmf=None):

        self.pha = [make_path(f) for f in pha]
        self._model = None
        self._thres_lo = None
        self._thres_hi = None

    @classmethod
    def from_config(cls, config):

        outdir = make_path(config['general']['outdir'])
        pha_list = outdir.glob('pha_run*.pha')
        return cls(pha_list)

    @property
    def model(self):
        """
        Spectral model to be fit
        """
        return self._model

    @model.setter
    def model(self, model, name=None):
        """

        Parameters
        ----------
        model : `~sherpa.models.ArithmeticModel`
            Fit model
        name : str
            Name for Sherpa model instance, optional
        """
        import sherpa.models

        name = 'default' if name is None else name

        if isinstance(model, six.string_types):
            if model == 'PL' or model == 'PowerLaw':
                model = sherpa.models.PowLaw1D('powlaw1d.' + name)
                model.gamma = 2
                model.ref = 1e9
                model.ampl = 1e-20
            else:
                raise ValueError("Undefined model string: {}".format(model))

        if not isinstance(model, sherpa.models.ArithmeticModel):
            raise ValueError("Only sherpa models are supported at the moment")

        self._model = model

    @property
    def energy_threshold_low(self):
        """
        Low energy threshold of the spectral fit
        """
        return self._thres_lo

    @energy_threshold_low.setter
    def energy_threshold_low(self, energy):
        """
        Low energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            Low energy threshold
        """
        self._thres_lo = Energy(energy)

    @property
    def energy_threshold_high(self):
        """
       High energy threshold of the spectral fit
        """
        return self._thres_hi

    @energy_threshold_high.setter
    def energy_threshold_high(self, energy):
        """
        High energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            High energy threshold
        """
        self._thres_hi = Energy(energy)

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        ret = ''
        for p in self.pha:
            ret += str(p) + ","

        return ret

    def define_spectral_groups(self, pha_list, outdir, Offmin, Offmax, NOffbin, Effmin, Effmax, NEffbin, Zenmin, Zenmax, NZenbin):
        """
        Group the observations in different bands of zenith, efficiency and offset.

        Parameters
        ----------
        pha_list : list of str, `~gammapy.extern.pathlib.Path`
            List of PHA files to group in the different bands
        filename : str
            directory where to writte the bands files
        Offmin: float
            lower offset boundaries for the band
        Offmax: float
            higher offset boundaries for the band
        NOffbin: int
            Number of offset bin
        Effmin: float
            lower muon efficiency boundaries for the band
        Effmax: float
            higher muon efficiency boundaries for the band
        NEffbin: int
            Number of muon efficiency bin
        Zenmin: float
            lower zenith angle boundaries for the band
        Zenmax: float
            higher zenith angle boundaries for the band
        NZenbin: int
            Number of zenith angle bin

        Returns
        -------
        ListeBand : list
            name of the different  bands
        ListeGroupObs: list
            name of the pha files of the observations grouped for each previous band

        """

        CosZenmin = np.cos(Zenmax * math.pi / 180.)
        CosZenmax = np.cos(Zenmin * math.pi / 180.)
        NCosZenbin = NZenbin
        Offbin = (Offmax - Offmin) / NOffbin
        Effbin = (Effmax - Effmin) / NEffbin
        CosZenbin = (CosZenmax - CosZenmin) / NCosZenbin
        Offtab = np.arange(Offmin, Offmax, Offbin)
        Efftab = np.arange(Effmin, Effmax, Effbin)
        CosZentab = np.arange(CosZenmin, CosZenmax, CosZenbin)
        ListeBand = []
        ListeGroupObs = []

        # We loop over all the observations and determine to which band belong each observations. If for an observation, the band doesn't exist we create a new one otherwise we add the observation to the existing band.
        for filename in pha_list:
            try:
                shdu = pf.open(filename)
            except:
                print("Cannot open file: " + filename)
                print("skipping run")
            Offset = shdu[1].header['Offset']
            CosZen = np.cos(shdu[1].header['ZEN'] * math.pi / 180.)
            Efficiency = shdu[1].header['MUONEFF'] * 100
            for (ioff, off) in enumerate(Offtab):
                if ((Offset >= off) & (Offset < off + Offbin)):
                    break
            for (ieff, eff) in enumerate(Efftab):
                if ((Efficiency >= eff) & (Efficiency < eff + Effbin)):
                    break
            for (izen, zen) in enumerate(CosZentab):
                if ((CosZen >= zen) & (CosZen < zen + CosZenbin)):
                    break
            # Name of the band
            namebande = "Band_Eff_" + str(eff) + "_" + str(eff + Effbin) + "_CosZen_" + str(zen) + "_" + str(
                zen + CosZenbin) + "_Offset_" + str(off) + "_" + str(off + Offbin)

            # If the band doesn't existe (that is to say ListBand.count(namebande)==0), we add a new band to the list of Band ListeBand. We create a new list of observation to group in this new band
            if (ListeBand.count(namebande) == 0):
                ListeBand.append(namebande)
                ListeGroupObs.append([filename])
            # If the Band exist, we find the number of the band (ListeBand.index(namebande)) and we add this new observation to group in this band
            else:
                ListeGroupObs[ListeBand.index(namebande)].append(filename)

        for (iband, pha_list) in enumerate(ListeGroupObs):
            self.apply_groupping(pha_list, iband, outdir)

        return (ListeBand, ListeGroupObs)

    def apply_groupping(self, pha_list, band_number, outdir):
        #ATTENTION A LA GESTION DE L ENERGIE SOEUIL
        """
        Group ON, OFF, arf and rmf for a list of observations matching with a certain group in offset, efficiency and zenithal angle determined in the function band. Create the ogip file (pha, bkg, arf and rmf) for the band.

        Parameters
        ----------
        pha_list : list of str, `~gammapy.extern.pathlib.Path`
            List of PHA files of the band to group together
        band_number: int
            number of the band
        outdir: str
            directory where to store the ogip files of the band
        """
        Nontot = None
        Nofftot = None
        alphatot = None
        detchans = 0
        trueEbins = 0
        timetot = 0
        arftot = None
        ONheader = None
        OFFheader = None
        arfheader = None
        rmfheader = None
        enerheader = None
        elow = None
        ehigh = None
        emin = None
        emax = None
        for filename in pha_list:
            try:
                shdu = pf.open(filename)
                path = filename[:-(len(filename.split('/')[-1] + filename.split('/')[-2]) + 1)]
            except:
                print("Cannot open file: " + filename)
                print("skipping run")
            else:
                try:
                    bkgname = path + shdu[1].header['BACKFILE']
                    arfname = path + shdu[1].header['ANCRFILE']
                    rmfname = path + shdu[1].header['RESPFILE']
                    bhdu = pf.open(bkgname)
                    ahdu = pf.open(arfname)
                    rhdu = pf.open(rmfname)

                except:
                    print("Cannot open files associated to " + filename)
                    print("skipping run")
                else:
                    if (bhdu[1].header['BACKSCAL'] == 0):
                        print(filename)
                        print("Backscal value for OFF region equal 0")
                        continue

                    if Nontot is None:
                        ONheader = shdu[1].header
                        OFFheader = bhdu[1].header
                        arfheader = ahdu[1].header
                        rmfheader = rhdu[1].header
                        enerheader = rhdu[2].header

                        detchans = shdu[1].header['DETCHANS']
                        trueEbins = ahdu[1].header['NAXIS2']
                        elow = ahdu[1].data['ENERG_LO']
                        ehigh = ahdu[1].data['ENERG_HI']
                        emin = rhdu[2].data['E_MIN']
                        emax = rhdu[2].data['E_MAX']

                        Nontot = np.zeros(detchans)
                        Nofftot = np.zeros(detchans)
                        alphatot = np.zeros(detchans)
                        arftot = np.zeros(trueEbins)
                        rmftot = np.zeros((trueEbins, detchans))
                        rmf = np.zeros((trueEbins, detchans))

                    Non = shdu[1].data['counts']
                    Nontot += Non
                    Noff = bhdu[1].data['counts']
                    Nofftot += Noff
                    alpha = shdu[1].header['BACKSCAL'] / bhdu[1].header['BACKSCAL']
                    # print alpha
                    alphatot += alpha * Noff

                    time = shdu[1].header['EXPOSURE']
                    timetot += time
                    arf = ahdu[1].data['SPECRESP']
                    # an apparent bug in exported matrices
                    arf[np.where(arf < 0)[0]] *= 0.
                    arftot += arf * time

                    matrix = rhdu[1].data['MATRIX']
                    length_row = [row.shape[0] for row in matrix]

                    for i in np.where(length_row)[0]:
                        for j in range(rhdu[1].data['N_GRP'][i]):
                            fchan = rhdu[1].data['F_CHAN'][i][j]
                            nchan = rhdu[1].data['N_CHAN'][i][j]
                            rmftot[i, fchan:fchan + nchan] += matrix[i][:nchan] * arf[i] * time

        if Nontot is None:
            print("No runs to group in the band number " + str(
                band_number) + " due to the fact that all the observations in this band have a backscal value equal to zero")
        # sum runs together
        else:
            alphamean = np.zeros_like(alphatot)
            print(band_number)
            alphamean[Nofftot > 0] = alphatot[Nofftot > 0] / Nofftot[Nofftot > 0]
            backscale = np.ones_like(alphatot)
            backscale[Nofftot > 0] = Nofftot[Nofftot > 0] / alphatot[Nofftot > 0]

            arfmean = arftot / timetot
            rmfmean = np.zeros_like(rmftot)
            for i in np.where(arftot > 0.0)[0]:
                rmfmean[i] = rmftot[i] / arftot[i]

            channels = np.arange(detchans)
            grouping = np.ones(detchans)
            onscale = np.ones(detchans)

            # Now write results
            outputname = outdir + "/Band_number_" + str(band_number)
            onfile = outputname + '_pha.pha'
            offfile = outputname + '_bkg.pha'
            arffile = outputname + '_arf.fits'
            rmffile = outputname + '_rmf.fits'

            # total ON file
            ONcols = pf.ColDefs([pf.Column(name='CHANNEL', format=ONheader["TFORM1"], array=channels),
                                 pf.Column(name='COUNTS', format=ONheader["TFORM2"], array=Nontot, unit='count'),
                                 pf.Column(name='GROUPING', format='I', array=grouping),
                                 pf.Column(name='BACKSCAL', format='E', array=onscale)
                                 ])
            ONhdu = pf.BinTableHDU.from_columns(ONcols)

            pha_keywords = ['EXTNAME', 'TLMIN1', 'TLMAX1', 'POISSERR', 'DETCHANS', 'CORRSCAL', 'CHANTYPE', 'HDUCLASS',
                            'HDUCLAS1', 'HDUVERS', 'QUALITY', 'TELESCOP', 'INSTRUME', 'FILTER', 'AREASCAL', 'PCOUNT',
                            'GCOUNT']

            # Write keywords
            for entry in pha_keywords:
                ONhdu.header[entry] = ONheader[entry]

            ONhdu.header['BACKFILE'] = offfile
            ONhdu.header['ANCRFILE'] = arffile
            ONhdu.header['RESPFILE'] = rmffile
            ONhdu.header['EXPOSURE'] = timetot

            ONhdu.writeto(onfile, clobber=True)

            OFFcols = pf.ColDefs([pf.Column(name='CHANNEL', format=OFFheader["TFORM1"], array=channels),
                                  pf.Column(name='COUNTS', format=OFFheader["TFORM2"], array=Nofftot, unit='count'),
                                  pf.Column(name='GROUPING', format='I', array=grouping),
                                  pf.Column(name='BACKSCAL', format='E', array=backscale)
                                  ])
            OFFhdu = pf.BinTableHDU.from_columns(OFFcols)
            for entry in pha_keywords:
                OFFhdu.header[entry] = ONheader[entry]
            OFFhdu.header['BACKFILE'] = ''
            OFFhdu.header['ANCRFILE'] = ''
            OFFhdu.header['RESPFILE'] = ''
            OFFhdu.header['EXPOSURE'] = timetot

            OFFhdu.writeto(offfile, clobber=True)

            arfcols = pf.ColDefs([pf.Column(name='ENERG_LO', format=arfheader["TFORM1"], array=elow, unit='keV'),
                                  pf.Column(name='ENERG_HI', format=arfheader["TFORM2"], array=ehigh, unit='keV'),
                                  pf.Column(name='SPECRESP', format=arfheader["TFORM3"], array=arfmean, unit='cm2')
                                  ])
            arfhdu = pf.BinTableHDU.from_columns(arfcols)
            for entry in arfheader:
                arfhdu.header[entry] = arfheader[entry]
            arfhdu.writeto(arffile, clobber=True)

            totmatrix = np.zeros((trueEbins,), dtype=np.dtype('O'))
            f_chan = np.zeros((trueEbins,), dtype=np.dtype('O'))
            n_chan = np.zeros((trueEbins,), dtype=np.dtype('O'))
            n_grp = np.zeros((trueEbins,), dtype=np.dtype('int16'))

            for i in np.arange(trueEbins):
                idx = np.where(rmftot[i] > 0.)[0]
                n_grp[i] = 1
                if len(idx) > 0:
                    f_chan[i] = np.array([idx[0]], dtype=np.dtype('int16'))
                    n_chan[i] = np.array([idx[-1] - idx[0] + 1], dtype=np.dtype('int16'))
                    totmatrix[i] = rmfmean[i][idx[0]:idx[-1] + 1]
                else:
                    f_chan[i] = np.array([0], dtype=np.dtype('int16'))
                    n_chan[i] = np.array([0], dtype=np.dtype('int16'))
                    totmatrix[i] = np.array([], dtype=np.dtype('float32'))

            rmfcols = pf.ColDefs([pf.Column(name='ENERG_LO', format=rmfheader["TFORM1"], array=elow, unit='keV'),
                                  pf.Column(name='ENERG_HI', format=rmfheader["TFORM2"], array=ehigh, unit='keV'),
                                  pf.Column(name='N_GRP', format=rmfheader["TFORM3"], array=n_grp),
                                  pf.Column(name='F_CHAN', format=rmfheader["TFORM4"], array=f_chan),
                                  pf.Column(name='N_CHAN', format=rmfheader["TFORM5"], array=n_chan),
                                  pf.Column(name='MATRIX', format=rmfheader["TFORM6"], array=totmatrix)
                                  ])
            rmfhdu = pf.BinTableHDU.from_columns(rmfcols)

            enercols = pf.ColDefs([pf.Column(name='CHANNEL', format=enerheader["TFORM1"], array=channels),
                                   pf.Column(name='E_MIN', format=enerheader["TFORM2"], array=emin, unit='keV'),
                                   pf.Column(name='E_MAX', format=enerheader["TFORM3"], array=emax, unit='keV')
                                   ])
            enerhdu = pf.BinTableHDU.from_columns(enercols)
            for entry in rmfheader:
                rmfhdu.header[entry] = rmfheader[entry]
            for entry in enerheader:
                enerhdu.header[entry] = enerheader[entry]

            prihdu = pf.PrimaryHDU()
            thdulist = pf.HDUList([prihdu, rmfhdu, enerhdu])
            thdulist.writeto(rmffile, clobber=True)

    def info(self):
        """Print some basic info"""
        ss = 'Model\n'
        ss += str(self.model)
        ss += '\nEnergy Range\n'
        ss += str(self.energy_threshold_low) + ' - ' + str(
                self.energy_threshold_high)
        return ss

    def run(self, method='hspec'):
        if method == 'hspec':
            self._run_hspec_fit()
        elif method == 'sherpa':
            self._run_sherpa_fit()
        else:
            raise ValueError('Undefined fitting method')

    def _run_hspec_fit(self):
        """Run the gammapy.hspec fit
        """

        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat

        sau.set_conf_opt("max_rstat", 100)

        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        sau.freeze(self.model.ref)

        list_data = []
        for pha in self.pha:
            datid = pha.parts[-1][7:12]
            sau.load_data(datid, str(pha))
            sau.notice_id(datid, thres_lo, thres_hi)
            sau.set_source(datid, self.model)
            list_data.append(datid)

        wstat.wfit(list_data)

    def _run_sherpa_fit(self):
        """Plain sherpa fit not using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(self.info())
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)
        ds.set_source(self.model)
        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        ds.notice(thres_lo, thres_hi)
        ds.subtract()
        ds.fit()
        ds.plot_fit()
        ds.clear_stack()
        ds.clear_models()

    def apply_containment(self, fit):
        """Apply correction factor for PSF containment in ON region"""
        cont = self.get_containment()
        fit['containment'] = cont
        fit['parvals'] = list(fit['parvals'])
        fit['parvals'][1] = fit['parvals'][1] * cont
        return fit

    def get_containment(self):
        """Calculate PSF correction factor for containment in ON region"""
        # TODO: do something useful here
        return 1


def run_spectral_fit_using_config(config):
    """
    Run a 1D spectral analysis using a config dict

    Parameters
    ----------
    config : dict
        Config dict

    Returns
    -------
    fit : `~gammapy.spectrum.SpectralFit`
        Fit instance
    """

    if config['general']['create_ogip']:
        analysis = SpectrumAnalysis.from_config(config)
        outdir = config['general']['outdir']
        analysis.write_ogip_data(outdir)

    method = config['general']['run_fit']
    if method is not 'False':
        fit = SpectralFit.from_config(config)
        fit.model = config['model']['type']
        fit.energy_threshold_low = Energy(config['model']['threshold_low'])
        fit.energy_threshold_high = Energy(config['model']['threshold_high'])
        fit.run(method=method)

    return fit
