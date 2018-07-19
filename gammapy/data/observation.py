# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from collections import OrderedDict
from astropy.utils import lazyproperty
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from ..utils.fits import earth_location_from_dict
from ..utils.time import time_ref_from_dict
from ..utils.table import table_row_to_dict

__all__ = [
    'Observation',
    'ObservationMeta',
    'ObservationIACT',
    'DataStoreObservation',
    'ObservationIACTMaker',
    'ObservationChecker',
]

log = logging.getLogger(__name__)


class Observation(object):
    """Container class for a generic observations

    Parameters
    ----------
    obs_id : `int`
        Observation ID
    events : `~gammapy.data.EventList`
        Event list, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html
    gti : `~gammapy.data.GTI`
        Good Time Intervals, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/events/gti.html
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/edisp/index.html
    psf : `~gammapy.irf.PSF3D` or `~gammapy.irf.EnergyDependentMultiGaussPSF` or `~gammapy.irf.PSFKing`
        Tabled Point Spread Function, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/index.html

    Other Parameters
    ----------------
    **kwargs :
        All other keyword arguments are passed on to the `~gammapy.data.ObservationMeta` constructor and can be
        accessed via the `metadata` class attribute:

        >>> from gammapy.data import Observation
        >>> myObs = Observation(obs_id=1, events=my_event_list, psf=my_psf, myMetadata='Best observation ever!')
        >>> myObs.metadata.myMetadata

    """
    def __init__(self, obs_id=None, events=None, gti=None, aeff=None, edisp=None, psf=None, **kwargs):
        self.obs_id = obs_id
        self.events = events
        self.gti = gti
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.metadata = ObservationMeta(**kwargs)

    def __str__(self):
        """Generate summary info string."""
        ss = 'Info for OBS_ID = {}\n'.format(self.obs_id)
        ss += '- Number of events: {}\n'.format(len(self.events.table) if self.events else 'None')
        ss += '- PSF type: {}\n'.format(type(self.psf))
        return ss

    def check_observation(self):
        """Convenient method to perform some basic sanity checks on this observation with the ObservationChecker."""
        obs_checker = ObservationChecker(self)
        return obs_checker.check_all()


class ObservationMeta(object):
    """Container class for observation metadata

    TODO: Maybe come up with some basic metadata that every observation holds

    Parameter
    ---------
    **kwargs :
        Arbitrary keyword arguments that will be stored as class attributes:

        >>> from gammapy.data import ObservationMeta
        >>> myObsMeta = ObservationMeta(myMeta='hands off!', yourMeta='whatever', ourMeta='fine...')
        >>> myObsMeta.myMeta
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ObservationIACT(Observation):
    """Container class for an IACT observation

    Parameters follow loosely the "Required columns" here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html#required-columns

    Parameters
    ----------
    pointing_radec : `~astropy.coordinates.SkyCoord`
        Pointing RA / DEC sky coordinates
    pointing_altaz : `~astropy.coordinates.SkyCoord`
        Pointing ALT / AZ sky coordinates
    pointing_zen : `~astropy.coordinates.SkyCoord`
        Pointing zenith angle sky
    observation_time_duration : `~astropy.units.Quantity`
        Observation time duration in seconds

        The wall time, including dead-time.
    observation_live_time_duration : `~astropy.units.Quantity`
        Live-time duration in seconds

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
    observation_dead_time_fraction : `float`
        Dead-time fraction

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
    tstart : `~astropy.units.Quantity`
        Observation start time
    tstop : `~astropy.units.Quantity`
        Observation stop time
    telescope_ids : list of `int`
        Telescope IDs of participating telescopes

    Other Parameters
    ----------------
    **kwargs :
        All other keyword arguments are passed on to the `~gammapy.data.Observation` constructor.

    """
    def __init__(self, pointing_radec=None, pointing_altaz=None, pointing_zen=None, observation_time_duration=None,
                 observation_live_time_duration=None, observation_dead_time_fraction=None, tstart=None, tstop=None,
                 telescope_ids=None, **kwargs):
        super(ObservationIACT, self).__init__(**kwargs)
        self.pointing_radec = pointing_radec
        self.pointing_altaz = pointing_altaz
        self.pointing_zen = pointing_zen
        self.observation_time_duration = observation_time_duration
        self.observation_live_time_duration = observation_live_time_duration
        self.observation_dead_time_fraction = observation_dead_time_fraction
        self.tstart = tstart
        self.tstop = tstop
        self.telescope_ids = telescope_ids

    def __str__(self):
        """Generate summary info string."""
        ss = 'Info for OBS_ID = {}\n'.format(self.obs_id)
        ss += '- Number of events: {}\n'.format(len(self.events.table) if self.events else 'None')
        ss += '- PSF type: {}\n'.format(type(self.psf))
        ss += '- Start time: {:.2f}\n'.format(self.tstart.mjd if self.tstart else 'None')
        ss += '- Pointing pos: RA {:.2f} / Dec {:.2f}\n'.format(self.pointing_radec.ra, self.pointing_radec.dec if
                                                                self.pointing_radec else 'None')
        ss += '- Observation duration: {}\n'.format(self.observation_time_duration)
        ss += '- Dead-time fraction: {:5.3f} %\n'.format(100 * self.observation_dead_time_fraction)

        return ss


class DataStoreObservation(ObservationIACT):
    """An IACT observation linked to a DataStore object.

    In this way the event list, for example, will not be stored in memory but always be read from disk.

    See :ref:`data_store`

    Parameters
    ----------
    data_store : `~gammapy.data.DataStore`
        Data store
    obs_id : int
        Observation ID

    Other Parameters
    ----------------
    **kwargs :
        All other keyword arguments are passed on to the `~gammapy.data.ObservationIACT` constructor.
    """

    def __init__(self, data_store, obs_id, **kwargs):
        super(DataStoreObservation, self).__init__(**kwargs)
        # Assert that `obs_id` is available
        if obs_id not in data_store.obs_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in obs index table.'.format(obs_id))
        if obs_id not in data_store.hdu_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in HDU index table.'.format(obs_id))

        self.obs_id = obs_id
        self.data_store = data_store

    @property
    def events(self):
        """Load `gammapy.data.EventList` object."""
        return self.data_store.hdu_table.hdu_location(obs_id=self.obs_id, hdu_type='events').load()

    @events.setter
    def events(self, value):
        pass


class ObservationIACTMaker(object):
    @staticmethod
    def from_data_store(data_store, obs_ids=None, link_data_store=False):
        """Make a list of IACT observations from a DataStore object.

        Parameters
        ----------
        data_store : `~gammapy.data.DataStore`
            DataStore object
        obs_ids : `int` or list of `int`, default: `None`
            Observation IDs. If `None`, all observations in the DataStore object will be selected.
        link_data_store : `bool`, default: `False`
            If true, the observations will be linked to the DataStore object. See `~gammapy.data.DataStoreObservation`.

        Returns
        --------
        obs_list : list of `~gammapy.data.Observation` or `~gammapy.data.DataStoreObservation`
            If `link_data_store` is True, a list of `~gammapy.data.DataStoreObservation` will be returned.

        """
        obs_ids = np.atleast_1d(obs_ids)

        # If no obs_ids are given, take all obs_ids available in the DataStore object
        if not obs_ids:
            obs_ids = data_store.obs_table['OBS_ID']

        # List of Observation objects that will be returned
        obs_list = []

        for obs_id in obs_ids:
            obs = DataStoreObservation(data_store, obs_id) if link_data_store else ObservationIACT(obs_id=obs_id)
            obs_filler = _ObservationIACTFillerFromDataStore(obs, data_store, obs_id)
            obs_filler.fill()

            obs_list.append(obs)

        return obs_list


class _ObservationIACTFillerFromDataStore(object):
    """Helper class for the ObservationIACTMaker to fill an Observation with data from a DataStore object

    Additionally it stores following parameters in the metadata of the observation:

    bkg_2d : `~gammapy.irf.Background2D`
        2D Background rate, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html#bkg-2d
    bkg_3d : `~gammapy.irf.Background3D`
        3D Background rate, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html#bkg-3d
    target_radec : `~astropy.coordinates.SkyCoord`
        Target RA / DEC sky coordinates
    observatory_earth_location : `~astropy.coordinates.EarthLocation`
        Observatory location
    muoneff : `float`?
        Observation muon efficiency

    """
    def __init__(self, obs, data_store, obs_id):
        self.obs = obs
        self.data_store = data_store
        self.obs_id = obs_id

    @lazyproperty
    def obs_info(self):
        """Observation information (`~collections.OrderedDict`)."""
        row = self.data_store.obs_table.select_obs_id(obs_id=self.obs_id)[0]
        return table_row_to_dict(row)

    def fill_from_hdu_index_table(self):
        """Locates and loads the HDU of interest via the HDU_TYPE and HDU_CLASS keywords.

        For details see: http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html
        """
        obs_from_hdu_dict = {
            'events': ['events', None], 'gti': ['gti', None], 'aeff': ['aeff', None], 'edisp': ['edisp', None],
            'psf': ['psf', None]
        }
        for obs_attr, hdu_value in obs_from_hdu_dict.items():
            try:
                setattr(self.obs, obs_attr, self.data_store.hdu_table.hdu_location(obs_id=self.obs_id,
                                                                                   hdu_type=hdu_value[0],
                                                                                   hdu_class=hdu_value[1]).load())
            except KeyError:
                log.warning("Could not fill {}".format(obs_attr))

    def fill_tstart(self):
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        self.obs.tstart = met_ref + Quantity(self.obs_info['TSTART'].astype('float64'), 'second')

    def fill_tstop(self):
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        self.obs.tstop = met_ref + Quantity(self.obs_info['TSTOP'].astype('float64'), 'second')

    def fill_observation_time_duration(self):
        self.obs.observation_time_duration = Quantity(self.obs_info['ONTIME'], 'second')

    def fill_observation_live_time_duration(self):
        self.obs.observation_live_time_duration = Quantity(self.obs_info['LIVETIME'], 'second')

    def fill_observation_dead_time_fraction(self):
        self.obs.observation_dead_time_fraction = 1 - self.obs_info['DEADC']

    def fill_pointing_radec(self):
        lon, lat = self.obs_info['RA_PNT'], self.obs_info['DEC_PNT']
        self.obs.pointing_radec = SkyCoord(lon, lat, unit='deg', frame='icrs')

    def fill_pointing_altaz(self):
        alt, az = self.obs_info['ALT_PNT'], self.obs_info['AZ_PNT']
        self.obs.pointing_altaz = SkyCoord(az, alt, unit='deg', frame='altaz')

    def fill_poiinting_zen(self):
        self.obs.pointing_zen = Quantity(self.obs_info['ZEN_PNT'], unit='deg')

    def fill_target_radec(self):
        lon, lat = self.obs_info['RA_OBJ'], self.obs_info['DEC_OBJ']
        self.obs.metadata.target_radec = SkyCoord(lon, lat, unit='deg', frame='icrs')

    def fill_observatory_earth_location(self):
        self.obs.metadata.observatory_earth_location = earth_location_from_dict(self.obs_info)

    def fill_muoneff(self):
        self.obs.metadata.muoneff = self.obs_info['MUONEFF']

    def fill(self):
        fill_methods = [method for method in dir(self) if method.startswith('fill_')]
        for fill_method in fill_methods:
            try:
                getattr(self, fill_method)()
            except KeyError:
                log.warning("{} failed".format(fill_method))


class ObservationChecker(object):
    """Class to perform sanity checks on an Observation object

    Parameter
    ---------
    observation : `~gammapy.data.Observation`

    """
    def __init__(self, observation=None):
        self.obs = observation

    def check_all(self):
        """Perform some basic sanity checks on an observation.

        Returns
        -------
        results : `~collections.OrderedDict`
            Dictionary with failure messages for the individual checks that failed.
            If `results['status'] == 'ok'`, every check passed.
        """
        results = OrderedDict()
        self.results['event list'] = self.check_event_list()
        self.results['effective area'] = self.check_effective_area()
        self.results['energy dispersion'] = self.check_energy_dispersion()
        self.results['psf'] = self.check_psf()

        status = 'ok'
        for key in results:
            if results[key]['status'] == 'failed':
                status = 'failed'
        results['status'] = status

        return results

    def check_event_list(self):
        """Check event list"""
        check_result = OrderedDict()
        if len(self.obs.events.table) == 0:
            check_result['nr of events'] = 'No events found in the event list'

        self._add_status(check_result)
        return check_result

    def check_effective_area(self):
        """Check effective area"""
        check_result = OrderedDict()
        if self.obs.aeff.meta['LO_THRES'] >= self.obs.aeff.meta['HI_THRES']:
            check_result['energy thresholds'] = "LO_THRES >= HI_THRES in effective area meta data"
        if np.max(self.obs.aeff.data.data) <= 0:
            check_result['values'] = "maximum entry of effective area table <= 0"

        self._add_status(check_result)
        return check_result

    def check_energy_dispersion(self):
        """Check energy dispersion"""
        check_result = OrderedDict()
        if np.max(self.obs.edisp.data.data) <= 0:
            check_result['value'] = "maximum entry of energy dispersion table <= 0"

        self._add_status(check_result)
        return check_result

    def check_psf(self):
        """Check PSF"""
        check_result = OrderedDict()
        if self.obs.psf.energy_thresh_lo >= self.obs.psf.energy_thresh_hi:
            check_result['energy thresholds'] = "LO_THRES >= HI_THRES in psf meta data"

        self._add_status(check_result)
        return check_result

    @staticmethod
    def _add_status(check_result):
        if check_result:
            check_result['status'] = "failed"
        else:
            check_result['status'] = "ok"
