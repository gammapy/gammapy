# Licensed under a 3-clause BSD style license - see LICENSE.rst
# from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import math
from astropy.table import Column
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from astropy.wcs.utils import skycoord_to_pixel
from astropy.table import Table
from . import CountsSpectrum
from ..irf.effective_area_table import EffectiveAreaTable
from ..irf.energy_dispersion import EnergyDispersion
from ..background import ring_area_factor, Cube
from ..data import DataStore
from ..image import ExclusionMask
from ..region import SkyCircleRegion, find_reflected_regions
from ..utils.energy import EnergyBounds, Energy
from ..data import ObservationGroupAxis, ObservationGroups
from ..utils.scripts import (
    get_parser, set_up_logging_from_args, read_yaml, make_path,
)

__all__ = [
    'SpectrumAnalysis',
    'SpectrumObservation',
    'SpectrumFit',
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

    This class creates several `~gammapy.spectrum.SpectrumObservation` instances
    for a given list of observations. All functionality is implemented in the
    `~gammapy.spectrum.SpectrumObservation` class. This class is only
    responsible for managing several observations.
    For more info see :ref:`spectral_fitting`.

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
    bkg_method : dict
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
        self.ebounds = ebounds
        self.bkg_method = 'no method' if bkg_method is None else bkg_method

        if isinstance(obs, six.string_types):
            obs = make_path(obs)
            obs = np.loadtxt(str(obs), dtype=np.int)

        observations = []
        for i, val in enumerate(obs):
            try:
                temp = SpectrumObservation(val, self.store, on_region,
                                           bkg_method, ebounds, exclusion)
            except IndexError:
                log.warning(
                    'Observation {} not in store {}'.format(val, datastore))
                nobs += 1
                continue
            observations.append(temp)
            if i == nobs - 1:
                break

        self._observations = np.array(observations)
        if len(self.observations) == 0:
            raise ValueError("No valid observations found")
        if bkg_method['type'] == 'reflected':
            mask, ibkg_obs = self.filter_by_reflected_regions(bkg_method['n_min'])
            self._bkg_observations = self.observations[ibkg_obs]
            self._observations = self.observations[mask]

    def copy(self, bkg_method=None):
        """Return copy of `~gammapy.spectrum.SpectrumAnalysis`

        Parameters
        ----------
        bkg_method : dict, optional
            New background estimation method
        """

        bkg_method = self.bkg_method if bkg_method is None else bkg_method

        ana = SpectrumAnalysis(datastore=self.store, obs=self.obs_ids,
                               on_region=self.on_region, bkg_method=bkg_method,
                               exclusion=self.exclusion, nobs=0,
                               ebounds=self.ebounds)
        return ana

    @property
    def observations(self):
        """`np.array` of all observations belonging to the analysis
        """
        return self._observations

    @property
    def obs_ids(self):
        """List of all observation ids"""
        return [o.obs for o in self.observations]

    def get_obs_by_id(self, id):
        """Return an certain observation belonging to the analysis

        Parameters
        ----------
        id : int
            Observation Id (runnumber)

        Returns
        -------
        observation : `~gammapy.spectrum.SpectrumObservation`
            Spectrum observation
        """
        try:
            i = self.obs_ids.index(id)
        except ValueError:
            raise ValueError("Observation {} not found".format(id))
        return self.observations[i]

    @property
    def offset(self):
        """List of offsets from the observation position for all observations
        """
        return [obs.offset for obs in self.observations]

    @property
    def on_vector(self):
        """List of on `~gammapy.spectrum.CountsSpectrum`
        """
        return [obs.on_vector for obs in self.observations]

    @property
    def off_vector(self):
        """`np.array` of off `~gammapy.spectrum.CountsSpectrum`

        For available methods see :ref:`spectrum_background_method`

        Parameters
        ----------
        method : dict, optional
            Background estimation method
        """
        return np.array([obs.off_vector for obs in self.observations])

    @property
    def alpha(self):
        """`np.array` of all exposure ratios between ON and OFF regions
        """
        return np.array([obs.alpha for obs in self.observations])

    @property
    def total_alpha(self):
        """Averaged exposure ratio between ON and OFF regions

        :math:`\\alpha_{\\mathrm{tot}}` for all observations is calculated as

        .. math:: \\alpha_{\\mathrm{tot}} = \\frac{\\sum_{i}\\alpha_i \\cdot N_i}{\\sum_{i} N_i}

        where :math:`N_i` is the number of OFF counts for observation :math:`i`
        """
        val = [o.alpha * o.off_vector.total_counts for o in self.observations]
        num = np.sum(val)
        den = np.sum([o.off_vector.total_counts for o in self.observations])
        return num / den

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
        bkg_method = config['off_region']

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
        ss = "\nSpectrum Analysis\n"
        ss += "Observations : {}\n".format(len(self.observations))
        ss += "ON region    : {}\n".format(self.on_region.pos)

        return ss

    def write_ogip_data(self, outdir, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            write directory
        """

        for obs in self.observations:
            obs.write_ogip(outdir=outdir, **kwargs)
            log.info('Creating OGIP data for run{}'.format(obs.obs))

    def filter_by_reflected_regions(self, n_min):
        """Filter runs according to number of reflected regions

        Condition: number of reflected regions >= nmin

        Parameters
        ----------
        n_min : int
            Minimum number of reflected regions

        Returns
        -------
        idx : `np.array`
            Indices of element fulfilling the condition
        """
        val = self.bkg_method['type']
        if val != 'reflected':
            raise ValueError("Wrong background method: {}".format(val))

        condition = np.array([o.backscal for o in self.off_vector]) >= n_min
        idx = np.nonzero(condition)
        idx1 = np.where(condition == False)
        return idx[0], idx1[0]

    def define_spectral_groups(self, OffsetRange=[0, 2.5], NOffbin=25, EffRange=[0, 100], NEffbin=40,
                               ZenRange=[0., 70.], NZenbin=30):
        # Tab contiendrait les bandes et les observations a grouper pour chaque bande
        [Offmin, Offmax] = OffsetRange
        [Effmin, Effmax] = EffRange
        [Zenmin, Zenmax] = ZenRange
        CosZenmin = np.cos(Zenmax * math.pi / 180.)
        CosZenmax = np.cos(Zenmin * math.pi / 180.)
        Offtab = Angle(np.linspace(Offmin, Offmax, NOffbin + 1), "deg")
        Efftab = Quantity(np.linspace(Effmin, Effmax, NEffbin + 1), "")
        CosZentab = Quantity(np.linspace(CosZenmin, CosZenmax, NZenbin + 1), "")
        list_obs_group_axis = [ObservationGroupAxis('MUONEFF', Efftab / 100., 'bin_edges'),
                               ObservationGroupAxis('CosZEN', CosZentab, 'bin_edges'),
                               ObservationGroupAxis('Offset', Offtab, 'bin_edges')]
        obs_groups = ObservationGroups(list_obs_group_axis)
        Observation_Table = self.store.obs_table
        list_index_bkg_obs = []
        for obs in self._bkg_observations:
            i = np.where(obs.obs == Observation_Table["OBS_ID"])
            list_index_bkg_obs.append(i[0][0])

        Observation_Table.remove_rows(list_index_bkg_obs)
        offset = [i.value for i in self.offset]
        Offcol = Column(offset, name='Offset', unit="deg")
        Observation_Table.add_column(Offcol)
        obs_table_grouped = obs_groups.group_observation_table(Observation_Table)
        Nband = obs_groups.n_groups
        observation_band_list = []
        for nband in range(Nband):
            tablegroup = obs_groups.get_group_of_observations(obs_table_grouped, nband)
            if len(tablegroup) == 0:
                continue
            else:
                spectrum_observation_band_list = []

                obsvervations_id = tablegroup["OBS_ID"]
                for obs in obsvervations_id:
                    ind = np.where(self.obs_ids == obs)
                    spectrum_observation_band_list.append(self._observations[ind][0])
                ObsBand = SpectrumObservation(nband, self.store, self.on_region,
                                              self.bkg_method, self.ebounds, self.exclusion, True)
                ObsBand.apply_grouping(spectrum_observation_band_list, self.ebounds)
                observation_band_list.append(ObsBand)

        self._observations = np.array(observation_band_list)
        # import IPython; IPython.embed()


class SpectrumObservation(object):
    """Helper class for 1D region based spectral analysis

    This class corresponds to one spectral observation

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

    def __init__(self, obs, store, on_region, bkg_method, ebounds, exclusion, band=False):
        # Raises Error if obs is not available
        if band is False:
            store.filename(obs, 'events')
        self.obs = obs
        self.store = store
        self.on_region = on_region
        self.bkg_method = bkg_method
        self.ebounds = ebounds
        self.exclusion = exclusion
        self._event_list = None
        self._on = None
        self._off = None
        self._aeff = None
        self._edisp = None
        self.reflected_regions = None

    @property
    def event_list(self):
        """`~gammapy.data.EventList` corresponding to the observation
        """
        if self._event_list is None:
            self._event_list = self.store.load(obs_id=self.obs,
                                               filetype='events')
        return self._event_list

    @property
    def effective_area(self):
        """`~gammapy.irf.EffectiveAreaTable` corresponding to the observation
        """
        if self._aeff is None:
            self._make_aeff()
        return self._aeff

    @property
    def energy_dispersion(self):
        """`~gammapy.irf.EnergyDispersion` corresponding to the observation
        """
        if self._edisp is None:
            self._make_edisp()
        return self._edisp

    @property
    def on_vector(self):
        """ON `gammapy.spectrum.CountsSpectrum` corresponding to the observation
        """
        if self._on is None:
            self._make_on()
        return self._on

    @property
    def off_vector(self):
        """OFF `gammapy.spectrum.CountsSpectrum` corresponding to the observation
        """
        if self._off is None:
            self.make_off_vector()
        return self._off

    @property
    def alpha(self):
        """Exposure ratio between ON and OFF region"""
        return self.on_vector.backscal / self.off_vector.backscal

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

    @property
    def livetime(self):
        """Livetime of the observation"""
        return self.event_list.observation_live_time_duration

    def _make_on(self):
        """Create ON vector
        """
        on_list = self.event_list.select_circular_region(self.on_region)
        on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
        self._on = on_vec

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

        self.reflected_regions = off_region
        return off_region

    def make_off_vector(self):
        """Create off vector

        For available methods see :ref:`spectrum_background_method`

        Returns
        -------
        off_vec : `gammapy.spectrum.CountsSpectrum`
            Counts spectrum inside the OFF region
        """

        method = self.bkg_method
        if method['type'] == "ring":
            off_vec = self._make_off_vector_ring(method)
        elif method['type'] == "reflected":
            off_vec = self._make_off_vector_reflected(method)
        elif method['type'] == "bgmodel":
            off_vec = self._make_off_vector_bgmodel(method)
        else:
            raise ValueError("Undefined background method: {}".format(
                method['type']))

        self._off = off_vec
        return off_vec

    def _make_off_vector_reflected(self, method):
        """Helper function to create OFF vector from reflected regions"""
        kwargs = method.copy()
        kwargs.pop('type')
        kwargs.pop('n_min')
        off = self.make_reflected_regions(**kwargs)
        off_list = self.event_list.select_circular_region(off)
        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
        off_vec.backscal = len(off)
        return off_vec

    def _make_off_vector_ring(self, method):
        """Helper function to create OFF vector from ring"""
        center = self.on_region.pos
        radius = self.on_region.radius
        inner = Angle(method['inner_radius'])
        outer = Angle(method['outer_radius'])
        off_list = self.event_list.select_sky_ring(center, inner, outer)
        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
        alpha = ring_area_factor(radius.deg, inner.deg, outer.deg)
        off_vec.backscal = alpha
        return off_vec

    def _make_off_vector_bgmodel(self, method):
        """Helper function to create OFF vector from BgModel"""
        filename = self.store.filename(obs_id=self.obs, filetype='background')
        cube = Cube.read(filename, scheme='bg_cube')
        # TODO: Properly transform to SkyCoords
        coords = Angle([self.offset, '0 deg'])
        spec = cube.make_spectrum(coords, self.ebounds)
        cnts = spec * self.ebounds.bands * self.livetime * self.on_region.area
        off_vec = CountsSpectrum(cnts.decompose(), self.ebounds, backscal=1)
        return off_vec

    def _make_aeff(self):
        """Create effective area vector correct energy binning
        """
        aeff2d = self.store.load(obs_id=self.obs, filetype='aeff')
        arf_vec = aeff2d.to_effective_area_table(self.offset)
        self._aeff = arf_vec

    def _make_edisp(self):
        """Create energy dispersion matrix in correct energy binning
        """
        edisp2d = self.store.load(obs_id=self.obs, filetype='edisp')
        rmf_mat = edisp2d.to_energy_dispersion(self.offset,
                                               e_reco=self.ebounds)
        self._edisp = rmf_mat

    def write_ogip(self, phafile=None, bkgfile=None, rmffile=None, arffile=None,
                   outdir=None, clobber=True):
        """Write OGIP files

        The arf, rmf and bkg files are set in the :ref:`gadf:ogip-pha` FITS
        header. If no filenames are given, default names will be chosen.

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
            phafile = "pha_run{}.pha".format(self.obs)
        if arffile is None:
            arffile = "arf_run{}.fits".format(self.obs)
        if rmffile is None:
            rmffile = "rmf_run{}.fits".format(self.obs)
        if bkgfile is None:
            bkgfile = "bkg_run{}.fits".format(self.obs)

        self.on_vector.write(str(outdir / phafile), bkg=str(bkgfile), arf=str(arffile),
                             rmf=str(rmffile), clobber=clobber)
        self.off_vector.write(str(outdir / bkgfile), clobber=clobber)
        self.effective_area.write(str(outdir / arffile), energy_unit='keV',
                                  effarea_unit='cm2', clobber=clobber)
        self.energy_dispersion.write(str(outdir / rmffile), energy_unit='keV',
                                     clobber=clobber)

    def plot_exclusion_mask(self, size=None, **kwargs):
        """Plot exclusion mask for this observation

        The plot will be centered at the pointing position

        Parameters
        ----------
        size : `~astropy.coordinates.Angle`
            Edge length of the plot
        """
        size = Angle('5 deg') if size is None else size
        ax = self.exclusion.plot(**kwargs)
        self._set_ax_limits(ax, size)
        point = skycoord_to_pixel(self.pointing, ax.wcs)
        ax.scatter(point[0], point[1], s=250, marker="+", color='black')
        return ax

    def plot_on_region(self, ax=None, **kwargs):
        """Plot target regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        self.on_region.plot(ax, **kwargs)

    def plot_reflected_regions(self, ax=None, **kwargs):
        """Plot reflected regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        if self.reflected_regions is None:
            self.make_reflected_regions()
        self.reflected_regions.plot(ax, **kwargs)

    def _check_binning(self, **kwargs):
        """Check that ARF and RMF binnings are compatible
        """
        pass

    def _set_ax_limits(self, ax, extent):

        if 'GLAT' in ax.wcs.to_header()['CTYPE2']:
            center = self.pointing.galactic
            xlim = (center.l + extent / 2).value, (center.l - extent / 2).value
            ylim = (center.b + extent / 2).value, (center.b - extent / 2).value
        else:
            center = self.pointing.icrs
            xlim = (center.ra + extent / 2).value, (center.ra - extent / 2).value
            ylim = (center.dec + extent / 2).value, (center.dec - extent / 2).value

        limits = ax.wcs.wcs_world2pix(xlim, ylim, 1)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    def apply_grouping(self, spectrum_observation_list, ebounds):

        """Method that stack the ON, OFF, arf and RMF for an observation group
        
        Parameters
        ----------
        spectrum_observation_list : list of `~gammapy.spectrum.SpectrumObservation`
                                 contain the list of the observations to group together
        
        """
        # Loop over the List of SpectrumObservation object to stack the ON, OFF rmf et arf
        ONband = None
        OFFband = None
        OFFtotband = None
        backscalband = None
        livetimeband = None
        arfband = None
        rmfband = None
        for (n, obs) in enumerate(spectrum_observation_list):
            obs._make_on()
            obs.make_off_vector()
            obs._make_aeff()
            obs._make_edisp()
            # import IPython; IPython.embed()
            on_vector = obs._on.counts
            off_vector = obs._off.counts
            OFF = np.sum(off_vector)
            # For the moment alpha for one band independent of the energy, weighted by the total OFF events
            backscal = obs._off.backscal
            livetime = obs._off.livetime
            arf_vector = obs._aeff.effective_area
            rmf_matrix = obs._edisp.pdf_matrix
            # Find a better way to do this since I initialize for the first SpectrumObservation of the band (n==0) and I sum for the other observations... I think there are a way to combine the initialisation and sum
            if (n == 0):
                # Pour la creation de l objet effective_area_table et de l objet energy_dispersion pour ecrire en forma ogip
                energy_hi = obs._aeff.energy_hi
                energy_lo = obs._aeff.energy_lo
                # Ca c est tres sale car normalement les membres avec un _ on doit pas y acceder direct comme ca voir comment determiner etrue autrement
                e_true = obs._edisp._e_true
                ONband = on_vector
                OFFband = off_vector
                OFFtotband = OFF
                backscalband = backscal * OFF
                # For a dependent energy backscale
                # backscalband=backscal*off_vector
                livetimeband = livetime
                arfband = arf_vector * livetime
                # For the first observation to group: the rmftab dimension is initialized to dim(Etrue,Ereco)
                dim_Etrue = np.shape(rmf_matrix)[0]
                dim_Ereco = np.shape(rmf_matrix)[1]
                rmfband = np.zeros((dim_Etrue, dim_Ereco))
                rmfmean = np.zeros((dim_Etrue, dim_Ereco))
                for ind_Etrue in range(dim_Etrue):
                    rmfband[ind_Etrue, :] = rmf_matrix[ind_Etrue, :] * arf_vector[ind_Etrue] * livetime
            else:
                ONband += on_vector
                OFFband += off_vector
                OFFtotband += OFF
                backscalband += backscal * OFF
                # For a dependent energy backscale
                # backscalband=backscal*off_vector
                livetimeband += livetime
                arfband += arf_vector * livetime
                # rmf et dimEtrue already defined in the if(n==0) for the first observation
                for ind_Etrue in range(dim_Etrue):
                    rmfband[ind_Etrue, :] += rmf_matrix[ind_Etrue, :] * arf_vector[ind_Etrue] * livetime

        # Mean backscale of the band
        backscalmean = backscalband / OFFtotband
        # backscalmean = backscalband / OFFband
        arfmean = arfband / livetimeband
        for ind_Etrue in range(dim_Etrue):
            rmfmean[ind_Etrue, :] = rmfband[ind_Etrue, :] / arfband[ind_Etrue]
        rmfmean[np.isnan(rmfmean)] = 0
        if (self.obs == 375):
            print rmfmean
            import IPython;
            IPython.embed()
        self._on = CountsSpectrum(ONband, ebounds, livetimeband)
        self._off = CountsSpectrum(OFFband, ebounds, livetimeband, backscalmean)
        self._aeff = EffectiveAreaTable(energy_lo, energy_hi, arfmean)
        self._edisp = EnergyDispersion(rmfmean, e_true, ebounds)
        # import IPython; IPython.embed()


class SpectrumFit(object):
    """
    Spectral Fit

    Parameters
    ----------
    pha : list of str, `~gammapy.extern.pathlib.Path`
        List of PHA files to fit
    """

    DEFAULT_STAT = 'cash'

    def __init__(self, pha, bkg=None, arf=None, rmf=None, stat=DEFAULT_STAT):

        self.pha = [make_path(f) for f in pha]
        self._model = None
        self._thres_lo = None
        self._thres_hi = None
        self._stat = stat

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumFit` from config file"""
        outdir = make_path(config['general']['outdir'])
        return cls.from_dir(outdir)

    @classmethod
    def from_dir(cls, dir):
        """Create `~gammapy.spectrum.SpectrumFit` using directory

        All PHA files in the directory will be used
        """
        dir = make_path(dir)
        pha_list = dir.glob('pha_run*.pha')
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
            raise ValueError("Only sherpa models are supported")

        self._model = model

    @property
    def statistic(self):
        """Statistic to be used in the fit"""
        return self._stat

    @statistic.setter
    def statistic(self, stat):
        """Set Statistic to be used in the fit

        Parameters
        ----------
        stat : `~sherpa.stats.Stat`, str
            Statistic
        """
        import sherpa.stats as s

        if isinstance(stat, six.string_types):
            if stat == 'cash':
                stat = s.Cash()
            else:
                raise ValueError("Undefined stat string: {}".format(stat))

        if not isinstance(stat, s.Stat):
            raise ValueError("Only sherpa statistics are supported")

        self._stat = stat

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
        datastack.set_stat(self.statistic)
        ds.fit()
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

    This function is called by the ``gammapy-spectrum`` command line tool.
    See :ref:`spectrum_command_line_tool`.

    Parameters
    ----------
    config : dict
        Config dict

    Returns
    -------
    fit : `~gammapy.spectrum.SpectrumFit`
        Fit instance
    """

    if config['general']['create_ogip']:
        analysis = SpectrumAnalysis.from_config(config)
        outdir = config['general']['outdir']
        analysis.write_ogip_data(outdir)

    method = config['general']['run_fit']
    if method is not 'False':
        fit = SpectrumFit.from_config(config)
        fit.model = config['model']['type']
        fit.energy_threshold_low = Energy(config['model']['threshold_low'])
        fit.energy_threshold_high = Energy(config['model']['threshold_high'])
        fit.info()
        fit.run(method=method)

    return fit
