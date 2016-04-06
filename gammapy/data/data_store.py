# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import numpy as np
from collections import OrderedDict

import subprocess
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from ..utils.scripts import make_path
from .observation import ObservationTable
from .hdu_index_table import HDUIndexTable
from .utils import _earth_location_from_dict

__all__ = [
    'DataStore',
    'DataStoreObservation',
]

log = logging.getLogger(__name__)


class DataStore(object):
    """IACT data store.

    The data selection and access happens an observation
    and an HDU index file as described at :ref:`gadf:iact-storage`.

    See :ref:`data_store`.

    Parameters
    ----------
    hdu_table : `~astropy.data.HDUIndexTable`
        HDU index table
    obs_table : `~gammapy.data.ObservationTable`
        Observation index table
    name : str
        Data store name
    """
    DEFAULT_HDU_TABLE = 'hdu-index.fits.gz'
    """Default HDU table filename."""

    DEFAULT_OBS_TABLE = 'obs-index.fits.gz'
    """Default observation table filename."""

    DEFAULT_NAME = 'noname'
    """Default data store name."""

    def __init__(self, hdu_table=None, obs_table=None, name=None):
        self.hdu_table = hdu_table
        self.obs_table = obs_table

        if name:
            self.name = name
        else:
            self.name = self.DEFAULT_NAME

    @classmethod
    def from_files(cls, base_dir, hdu_table_filename=None, obs_table_filename=None, name=None):
        """Construct `DataStore` from HDU and observation index table files."""
        if hdu_table_filename:
            log.debug('Reading {}'.format(hdu_table_filename))
            hdu_table = HDUIndexTable.read(str(hdu_table_filename), format='fits')

            hdu_table.meta['BASE_DIR'] = base_dir
        else:
            hdu_table = None

        if obs_table_filename:
            log.debug('Reading {}'.format(str(obs_table_filename)))
            obs_table = ObservationTable.read(str(obs_table_filename), format='fits')
        else:
            obs_table = None

        return cls(
            hdu_table=hdu_table,
            obs_table=obs_table,
            name=name,
        )

    @classmethod
    def from_dir(cls, base_dir, name=None):
        """Create a `DataStore` from a directory.

        This assumes that the HDU and observations index tables
        have the default filename.
        """
        base_dir = make_path(base_dir)
        return cls.from_files(
            base_dir=base_dir,
            hdu_table_filename=base_dir / cls.DEFAULT_HDU_TABLE,
            obs_table_filename=base_dir / cls.DEFAULT_OBS_TABLE,
            name=name,
        )

    @classmethod
    def from_config(cls, config):
        """Create a `DataStore` from a config dict."""
        base_dir = config['base_dir']
        name = config.get('name', cls.DEFAULT_NAME)
        hdu_table_filename = config.get('hduindx', cls.DEFAULT_HDU_TABLE)
        obs_table_filename = config.get('obsindx', cls.DEFAULT_OBS_TABLE)

        hdu_table_filename = cls._find_file(hdu_table_filename, base_dir)
        obs_table_filename = cls._find_file(obs_table_filename, base_dir)

        return cls.from_files(
            base_dir=base_dir,
            hdu_table_filename=hdu_table_filename,
            obs_table_filename=obs_table_filename,
            name=name,
        )

    @staticmethod
    def _find_file(filename, dir):
        """Find a file at an absolute or relative location.

        - First tries ``Path(filename)``
        - Second tries ``Path(dir) / filename``
        - Raises ``OSError`` if both don't exist.
        """
        path1 = make_path(filename)
        path2 = make_path(dir) / filename

        if path1.is_file():
            filename = path1
        elif path2.is_file():
            filename = path2
        else:
            raise OSError('File not found at {} or {}'.format(path1, path2))

        return filename

    @classmethod
    def from_name(cls, name):
        """Convenience method to look up DataStore from DataManager."""
        # This import needs to be delayed to avoid a circular import
        # It can't be moved to the top of the file
        from .data_manager import DataManager
        dm = DataManager()
        return dm[name]

    # TODO: seems too magical. needed? remove?
    @classmethod
    def from_all(cls, val):
        """Try different DataStore constructors.
        
        Currently tried (in this order)
        - :func:`~gammapy.data.DataStore.from_dir`
        - :func:`~gammapy.data.DataStore.from_name`

        Parameters
        ----------
        val : str
            Key to construct DataStore from
        """
        try:
            store = cls.from_dir(val)
        except OSError as e1:
            try:
                store = cls.from_name(val)
            except KeyError as e2:
                raise ValueError('Not able to contruct DataStore using key:'
                                 ' {0}.\nErrors\nfrom_dir: {1}\nfrom_name: {2}'
                                 .format(val, e1, e2))

        return store

    def info(self, file=None):
        """Print some info"""
        if not file:
            stream = sys.stdout

        print(file=stream)
        print('Data store summary info:', file=file)
        print('name: {}'.format(self.name), file=file)
        print('', file=file)
        self.hdu_table.summary(file=file)
        print('', file=file)
        self.obs_table.summary(file=file)

    def obs(self, obs_id):
        """Access a given `~gammapy.data.DataStoreObservation`.

        Parameters
        ----------
        obs_id : int
            Observation ID.

        Returns
        -------
        obs : `~gammapy.data.DataStoreObservation`
            Observation container
        """
        return DataStoreObservation(
            obs_id=obs_id,
            data_store=self,
        )

    def load_all(self, hdu_type=None, hdu_class=None):
        """Load a given file type for all observations

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        list : python list of object
            Object depends on type, e.g. for `events` it is a list of `~gammapy.data.EventList`.
        """
        obs_ids = self.obs_table['OBS_ID']
        return self.load_many(obs_ids=obs_ids, hdu_type=hdu_type, hdu_class=hdu_class)

    def load_many(self, obs_ids, hdu_type=None, hdu_class=None):
        """Load a given file type for certain observations in an obs_table

        Parameters
        ----------
        obs_ids : list
            List of observation IDs
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        list : list of object
            Object depends on type, e.g. for `events` it is a list of `~gammapy.data.EventList`.
        """
        things = []
        for obs_id in obs_ids:
            obs = self.obs(obs_id=obs_id)
            thing = obs.load(hdu_type=hdu_type, hdu_class=hdu_class)
            things.append(thing)

        return things

    def check_integrity(self, logger=None):
        """Check integrity, i.e. whether index and observation table match.
        """
        # Todo: This is broken - remove or fix?
        sane = True
        if logger is None:
            logger = logging.getLogger('default')

        logger.info('Checking event list files')
        available = self.check_available_event_lists(logger)
        if np.any(~available):
            logger.warning('Number of missing event list files: {}'.format(np.invert(available).sum()))

        # TODO: implement better, more complete integrity checks.
        return sane

    def make_table_of_files(self, observation_table=None, filetypes=['events']):
        """Make list of files in the datastore directory.

        Parameters
        ----------
        observation_table : `~gammapy.data.ObservationTable` or None
            Observation table (``None`` means select all observations).
        filetypes : list of str
            File types (TODO: document in a central location and reference from here).

        Returns
        -------
        table : `~astropy.table.Table`
            Table summarising info about files.
        """
        if observation_table is None:
            observation_table = ObservationTable(self.obs_table)

        data = []
        for observation in observation_table:
            for filetype in filetypes:
                row = dict()
                row['OBS_ID'] = observation['OBS_ID']
                row['filetype'] = filetype
                filename = self.filename(observation['OBS_ID'], filetype=filetype, abspath=True)
                row['filename'] = filename
                data.append(row)

        return Table(data=data, names=['OBS_ID', 'filetype', 'filename'])

    def check_available_event_lists(self, logger=None):
        """Check if all event lists are available.

        TODO: extend this function, or combine e.g. with ``make_table_of_files``.

        Returns
        -------
        file_available : `~numpy.ndarray`
            Boolean mask which files are available.
        """
        # Todo: This is broken. Remove (covered by HDUlocation class)?

        observation_table = self.obs_table
        file_available = np.ones(len(observation_table), dtype='bool')
        for ii in range(len(observation_table)):
            obs_id = observation_table['OBS_ID'][ii]
            filename = self.filename(obs_id)
            if not make_path(filename).is_file():
                file_available[ii] = False
                if logger:
                    logger.warning('For OBS_ID = {:06d} the event list file is missing: {}'
                                   ''.format(obs_id, filename))

        return file_available

    def copy_obs(self, obs_table, outdir):
        """Create a new `~gammapy.data.DataStore` containing a subset of observations

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Table of observation to create the subset
        outdir : str, Path
            Directory for the new store
        """
        outdir = make_path(outdir)
        obs_ids = obs_table['OBS_ID'].data

        hdutable = self.hdu_table
        hdutable.add_index('OBS_ID')
        subhdutable = hdutable.loc[obs_ids]
        subobstable = self.obs_table.select_obs_id(obs_ids)

        for ii in range(len(subhdutable)):
            # Changes to the file structure could be made here
            loc = subhdutable._location_info(ii)
            targetdir = outdir / loc.file_dir
            targetdir.mkdir(exist_ok=True, parents=True)
            cmd = ['cp', str(loc.path()), str(targetdir)]
            subprocess.call(cmd)

        subhdutable.write(str(outdir/self.DEFAULT_HDU_TABLE), format='fits')
        subobstable.write(str(outdir/self.DEFAULT_OBS_TABLE), format='fits')


class DataStoreObservation(object):
    """IACT data store observation.

    See :ref:`data_store`
    """

    def __init__(self, obs_id, data_store):
        # Assert that `obs_id` is available
        if obs_id not in data_store.obs_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in obs index table.'.format(obs_id))
        if obs_id not in data_store.hdu_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in HDU index table.'.format(obs_id))

        self.obs_id = obs_id
        self.data_store = data_store

    def location(self, hdu_type=None, hdu_class=None):
        """HDU location object.

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        location : `~gammapy.data.HDULocation`
            HDU location
        """
        location = self.data_store.hdu_table.hdu_location(
            obs_id=self.obs_id,
            hdu_type=hdu_type,
            hdu_class=hdu_class,
        )
        return location

    def load(self, hdu_type=None, hdu_class=None):
        """Load data file as appropriate object.

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        object : object
            Object depends on type, e.g. for `events` it's a `~gammapy.data.EventList`.
        """
        location = self.location(hdu_type=hdu_type, hdu_class=hdu_class)
        return location.load()

    @lazyproperty
    def events(self):
        """Load `gammapy.data.EventList` object (lazy property)."""
        return self.load(hdu_type='events')

    @lazyproperty
    def gti(self):
        """Load `gammapy.data.GTI` object (lazy property)."""
        return self.load(hdu_type='gti')

    @lazyproperty
    def aeff(self):
        """Load effective area object (lazy property)."""
        return self.load(hdu_type='aeff')

    @lazyproperty
    def edisp(self):
        """Load energy dispersion object (lazy property)."""
        return self.load(hdu_type='edisp')

    @lazyproperty
    def psf(self):
        """Load point spread function object (lazy property)."""
        return self.load(hdu_type='psf')

    @lazyproperty
    def bkg(self):
        """Load background object (lazy property)."""
        return self.load(hdu_type='bkg')

    # TODO: maybe the obs table row info should be put in a separate object?
    @lazyproperty
    def _obs_info(self):
        """Observation information"""
        row = self.data_store.obs_table.select_obs_id(obs_id=self.obs_id)[0]
        data = OrderedDict(zip(row.colnames, row.as_void()))
        return data

    @lazyproperty
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        info = self._obs_info
        lon, lat = info['RA_PNT'], info['DEC_PNT']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @lazyproperty
    def tstart(self):
        """Observation start time (`~astropy.time.Time`)."""
        info = self._obs_info
        return Quantity(info['TSTART'], 'second')

    @lazyproperty
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        info = self._obs_info
        lon, lat = info['RA_OBJ'], info['DEC_OBJ']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @lazyproperty
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        info = self._obs_info
        return _earth_location_from_dict(info)

    @lazyproperty
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`)

        The wall time, including dead-time.
        """
        info = self._obs_info
        return Quantity(info['ONTIME'], 'second')

    @lazyproperty
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`)

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        info = self._obs_info
        return Quantity(info['LIVETIME'], 'second')

    @lazyproperty
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float)

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        http://en.wikipedia.org/wiki/Dead_time
        http://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        info = self._obs_info
        return 1 - info['DEADC']

    def __str__(self):
        """Generate summary info string."""
        ss = 'Info for OBS_ID = {}\n'.format(self.obs_id)
        ss += '- Start time: {:.2f}\n'.format(self.tstart)
        ss += '- Pointing pos: RA {:.2f} / Dec {:.2f}\n'.format(self.pointing_radec.ra, self.pointing_radec.dec)
        ss += '- Observation duration: {}\n'.format(self.observation_time_duration)
        ss += '- Dead-time fraction: {:5.3f} %\n'.format(100 * self.observation_dead_time_fraction)

        # TODO: Which target was observed?
        # TODO: print info about available HDUs for this observation ...
        return ss

    def peek(self):
        """Quick-look plots in a few panels."""
        raise NotImplementedError

    def make_exposure_image(self, fov, energy_range):
        """Make exposure image.

        TODO: Do we want such methods here or as standalone functions that work with obs objects?
        """
        raise NotImplementedError
