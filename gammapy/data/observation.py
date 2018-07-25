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
from ..irf import EnergyDependentTablePSF, PSF3D

__all__ = [
    'Observation',
    'ObservationMeta',
    'ObservationIACT',
    'ObservationIACTLinked',
    'ObservationIACTMaker',
    'Checker',
    'ObservationChecker',
]

log = logging.getLogger(__name__)


class Observation(object):
    """Container class for a generic observation

    Parameters
    ----------
    obs_id : int
        Observation ID
    gti : `~gammapy.data.GTI`
        Good Time Intervals, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/events/gti.html
    events : `~gammapy.data.EventList`
        Event list, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/edisp/index.html
    psf : `~gammapy.irf.PSF3D` or `~gammapy.irf.EnergyDependentMultiGaussPSF` or `~gammapy.irf.PSFKing`
        Tabled Point Spread Function, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/index.html
    bkg: `~gammapy.irf.Background2D` or `~gammapy.irf.Background3D`
        Background rate, see: http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html

    Other Parameters
    ----------------
    **kwargs :
        All other keyword arguments are passed on to the `~gammapy.data.ObservationMeta` constructor and can be
        accessed via the `meta` class attribute:

        >>> from gammapy.data import Observation
        >>> myObs = Observation(obs_id=1, events=my_event_list, myMetadata='Best observation ever!')
        >>> myObs.meta.myMetadata

    """
    def __init__(self, obs_id=None, gti=None, events=None, aeff=None, edisp=None, psf=None, bkg=None, **kwargs):
        self.obs_id = obs_id
        self.gti = gti
        self.events = events
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self.meta = ObservationMeta(**kwargs)

    def __str__(self):
        """Generate summary info string."""
        ss = 'Info for OBS_ID = {}\n'.format(self.obs_id)
        ss += '- Number of events: {}\n'.format(len(self.events.table) if self.events else 'None')
        ss += '- Number of good time intervals: {}\n'.format(len(self.gti.table) if self.gti else 'None')
        ss += '- Type of eff. area: {}\n'.format(type(self.aeff))
        ss += '- Type of energy disp.: {}\n'.format(type(self.edisp))
        ss += '- Type of PSF: {}\n'.format(type(self.psf))
        return ss

    def check_observation(self, checks='all'):
        """Convenient method to perform some basic sanity checks on
        this observation with the ObservationChecker.

        See docstring of :func:`gammapy.data.ObservationChecker.run`
        """
        obs_checker = ObservationChecker(self)
        return obs_checker.run(checks)


class ObservationMeta(object):
    """Container class for observation metadata

    TODO: Maybe come up with some basic metadata that every observation holds
          or maybe just add the metadata to the obs.__dict__ ??

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
        where the detector did not record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
    tstart : `~astropy.units.Quantity`
        Observation start time
    tstop : `~astropy.units.Quantity`
        Observation stop time
    telescope_ids : list of int
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
        ss = super(ObservationIACT, self).__str__()
        ss += '- Start time: {:.2f}\n'.format(self.tstart.mjd if self.tstart else 'None')
        ss += '- Pointing pos: RA {:.2f} / Dec {:.2f}\n'.format(
            self.pointing_radec.ra if self.pointing_radec else 'None',
            self.pointing_radec.dec if self.pointing_radec else 'None')
        ss += '- Observation duration: {}\n'.format(self.observation_time_duration)
        ss += '- Dead-time fraction: {:5.3f} %\n'.format(100 * self.observation_dead_time_fraction)

        return ss

    @classmethod
    def from_data_store(cls, data_store, obs_id):
        """Convenient method to create this observation from a DataStore object.

        See :func:`gammapy.data.ObservationIACTMaker.from_data_store`
        """
        return ObservationIACTMaker.from_data_store(data_store, obs_id)

    # TODO: This method should be moved outside of this class
    def make_psf(self, position, energy=None, rad=None):
        """Make energy-dependent PSF for a given source position.

        Parameters
        ----------
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
        offset = position.separation(self.pointing_radec)
        energy = energy or self.psf.to_energy_dependent_table_psf(theta=offset).energy
        rad = rad or self.psf.to_energy_dependent_table_psf(theta=offset).rad

        if isinstance(self.psf, PSF3D):
            # PSF3D is a table PSF, so we use the native RAD binning by default
            # TODO: should handle this via a uniform caller API
            psf_value = self.psf.to_energy_dependent_table_psf(theta=offset).evaluate(energy)
        else:
            psf_value = self.psf.to_energy_dependent_table_psf(theta=offset, rad=rad).evaluate(energy)

        arf = self.aeff.data.evaluate(offset=offset, energy=energy)
        exposure = arf * self.observation_live_time_duration

        psf = EnergyDependentTablePSF(energy=energy, rad=rad,
                                      exposure=exposure, psf_value=psf_value)
        return psf


class ObservationIACTLinked(ObservationIACT):
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
        super(ObservationIACTLinked, self).__init__(**kwargs)
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

    @classmethod
    def from_data_store(cls, data_store, obs_id):
        """Convenient method to create this observation from a DataStore object.

        See :func:`gammapy.data.ObservationIACTMaker.from_data_store`
        """
        return ObservationIACTMaker.from_data_store(data_store, obs_id, link_data_store=True)


class ObservationIACTMaker(object):
    """Make IACT observations from different input types"""
    @staticmethod
    def from_data_store(data_store, obs_id, link_data_store=False):
        """Make an IACT observations from a DataStore object.

        Parameters
        ----------
        data_store : `~gammapy.data.DataStore`
            DataStore object
        obs_id : int
            Observation ID
        link_data_store : `bool`, default: `False`
            If true, the observations will be linked to the DataStore object. See `~gammapy.data.ObservationIACTLinked`.

        Returns
        --------
        obs : `~gammapy.data.ObservationIACT` or `~gammapy.data.ObservationIACTLinked`
            If `link_data_store` is True, a `~gammapy.data.ObservationIACTLinked` instance will be returned.

        """
        obs = ObservationIACTLinked(data_store, obs_id) if link_data_store else ObservationIACT(obs_id=obs_id)
        obs_filler = _ObservationIACTFillerFromDataStore(obs, data_store, obs_id)
        obs_filler.run()

        return obs


class _ObservationIACTFillerFromDataStore(object):
    """Helper class for the ObservationIACTMaker to fill an Observation with data from a DataStore object.

    Additionally it stores following parameters in the metadata of the observation:

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

    def fill_from_hdu_types(self):
        """Locates and loads objects from HDU tables via the HDU_TYPE and HDU_CLASS keywords.

        For details see: http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html
        TODO: data_store should get a method like `load`, like the one from the old DataStoreObservation class
        """
        # hdu types and the observation attributes happen to coincide
        hdu_type_to_obs_attr = OrderedDict(events='events', gti='gti', aeff='aeff', edisp='edisp', psf='psf', bkg='bkg')
        for hdu_type, obs_attr in hdu_type_to_obs_attr.items():
            try:
                setattr(self.obs, obs_attr,
                        self.data_store.hdu_table.hdu_location(obs_id=self.obs_id, hdu_type=hdu_type).load())
            except IndexError:
                log.warning('fill_{} failed'.format(obs_attr))

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

    def fill_pointing_zen(self):
        self.obs.pointing_zen = Quantity(self.obs_info['ZEN_PNT'], unit='deg')

    def fill_target_radec(self):
        lon, lat = self.obs_info['RA_OBJ'], self.obs_info['DEC_OBJ']
        self.obs.meta.target_radec = SkyCoord(lon, lat, unit='deg', frame='icrs')

    def fill_observatory_earth_location(self):
        self.obs.meta.observatory_earth_location = earth_location_from_dict(self.obs_info)

    def fill_muoneff(self):
        self.obs.meta.muoneff = self.obs_info['MUONEFF']

    def run(self):
        fill_methods = [method for method in dir(self) if method.startswith('fill_')]
        for fill_method in fill_methods:
            try:
                getattr(self, fill_method)()
            except KeyError or IndexError:
                log.warning("{} failed".format(fill_method))


class Checker(object):
    """Base class to perform some sanity checks on a certain container class.

    TODO: Move this class to another file/module
    """
    def run(self, checks='all'):
        """Run checks.

        Parameters
        ----------
        checks : str or list of str or 'all' (default)
            Which checks to run, a list of available checks is found in the property `available_checks`.

        Returns
        -------
        results : `~collections.OrderedDict`
            Dictionary with failure messages for the individual checks that failed.
            If `results['status'] == 'ok'`, every available check passed.
        """
        if checks == 'all':
            checks = self.available_checks
        else:
            unknown_checks = set(checks).difference(self.available_checks)
            if unknown_checks:
                raise ValueError('Unknown checks: {}'.format(unknown_checks))

        results = OrderedDict()
        for check in np.atleast_1d(checks):
            try:
                check_method = getattr(self, 'check_{}'.format(check))
                results[check] = check_method()
            except AttributeError:
                results[check] = OrderedDict(status='Not available')

        self._check_all_status(results)

        return results

    @property
    def available_checks(self):
        raise NotImplementedError

    @staticmethod
    def _add_status(check_result):
        if check_result:
            check_result['status'] = "failed"
        else:
            check_result['status'] = "ok"

    @staticmethod
    def _check_all_status(results):
        status = 'ok'
        for key in results:
            if results[key]['status'] == 'failed':
                status = 'failed'
        results['status'] = status


class ObservationChecker(Checker):
    """Class to perform sanity checks on an Observation object

    Parameter
    ---------
    observation : `~gammapy.data.Observation`

    """
    def __init__(self, observation=None):
        self.obs = observation

    @property
    def available_checks(self):
        return ['event_list', 'gti', 'effective_area', 'energy_dispersion', 'psf']

    def check_event_list(self):
        """Check event list
        TODO: Implement an EventListChecker and just call it from here, same goes for the rest of the checks. Like:
            >>> def check_event_list(self):
            >>>     return EventListChecker(obs.events).run()
        """
        check_result = OrderedDict()
        if len(self.obs.events.table) == 0:
            check_result['nr of events'] = 'No events found in the event list'

        self._add_status(check_result)

        return check_result

    def check_gti(self):
        """Check GTI"""
        check_result = OrderedDict()
        if len(self.obs.gti.table) == 0:
            check_result['nr of gtis'] = 'No good time intervals found in the GTI table'

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
