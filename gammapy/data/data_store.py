# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import numpy as np
from collections import OrderedDict
import subprocess
from astropy.table import Table
from ..utils.scripts import make_path
from .obs_table import ObservationTable
from .hdu_index_table import HDUIndexTable
from .observations import DataStoreObservation, ObservationList

__all__ = [
    'DataStore',
]

log = logging.getLogger(__name__)


class DataStore(object):
    """IACT data store.

    The data selection and access happens using an observation
    and an HDU index file as described at :ref:`gadf:iact-storage`.

    See :gp-extra-notebook:`cta_1dc_introduction` for usage examples.

    Parameters
    ----------
    hdu_table : `~gammapy.data.HDUIndexTable`
        HDU index table
    obs_table : `~gammapy.data.ObservationTable`
        Observation index table
    name : str
        Data store name

    Examples
    --------
    Here's an example how to create a `DataStore` to access H.E.S.S. data:

    >>> from gammapy.data import DataStore
    >>> dir = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
    >>> data_store = DataStore.from_dir(dir)
    >>> data_store.info()
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
        """Construct from HDU and observation index table files."""
        if hdu_table_filename:
            log.debug('Reading {}'.format(hdu_table_filename))
            hdu_table = HDUIndexTable.read(str(hdu_table_filename), format='fits')

            hdu_table.meta['BASE_DIR'] = str(base_dir)
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
        """Create from a directory.

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
        """Create from a config dict."""
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

    def info(self, show=True):
        """Print some info."""
        s = 'Data store summary info:\n'
        s += 'name: {!r}\n\n'.format(self.name)
        s += self.hdu_table.summary()
        s += '\n\n'
        s += self.obs_table.summary()

        if show:
            print(s)
        else:
            return s

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
            obs_id=int(obs_id),
            data_store=self,
        )

    def obs_list(self, obs_id, skip_missing=False):
        """Generate a `~gammapy.data.ObservationList`.

        Parameters
        ----------
        obs_id : list
            Observation IDs.
        skip_missing : bool, optional
            Skip missing observations, default: False

        Returns
        -------
        obs : `~gammapy.data.ObservationList`
            List of `~gammapy.data.DataStoreObservation`
        """
        obslist = ObservationList()
        for _ in obs_id:
            try:
                obs = self.obs(_)
            except ValueError as err:
                if skip_missing:
                    log.warn('Obs {} not in store, skip.'.format(_))
                    continue
                else:
                    raise err
            else:
                obslist.append(obs)
        return obslist

    def load_all(self, hdu_type=None, hdu_class=None):
        """Load a given file type for all observations.

        Parameters
        ----------
        hdu_type : str
            HDU type (see `~gammapy.data.HDUIndexTable.VALID_HDU_TYPE`)
        hdu_class : str
            HDU class (see `~gammapy.data.HDUIndexTable.VALID_HDU_CLASS`)

        Returns
        -------
        list : python list of object
            Object depends on type, e.g. for 'events' it is a list of `~gammapy.data.EventList`.
        """
        obs_ids = self.obs_table['OBS_ID']
        return self.load_many(obs_ids=obs_ids, hdu_type=hdu_type, hdu_class=hdu_class)

    def load_many(self, obs_ids, hdu_type=None, hdu_class=None):
        """Load a given file type for certain observations in an observation table.

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
            Object depends on type, e.g. for 'events' it is a list of `~gammapy.data.EventList`.
        """
        things = []
        for obs_id in obs_ids:
            obs = self.obs(obs_id=obs_id)
            thing = obs.load(hdu_type=hdu_type, hdu_class=hdu_class)
            things.append(thing)

        return things

    def check_observations(self):
        """Perform some sanity checks for all observations.

        Returns
        -------
        results : OrderedDict
            dictionary containing failure messages for all runs that fail a check.
        """

        results = OrderedDict()

        # Loop over all obs_ids in obs_table
        for obs_id in self.obs_table['OBS_ID']:
            messages = self.obs(obs_id).check_observation()
            if len(messages) > 0:
                results[obs_id] = messages

        return results

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
        # TODO : remove or fix
        raise NotImplementedError

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
        # TODO: This is broken. Remove (covered by HDUlocation class)?
        raise NotImplementedError

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

    def copy_obs(self, obs_id, outdir, hdu_class=None, verbose=False, overwrite=False):
        """Create a new `~gammapy.data.DataStore` containing a subset of observations.

        Parameters
        ----------
        obs_id : array-like, `~gammapy.data.ObservationTable`
            List of observations to copy
        outdir : str, Path
            Directory for the new store
        hdu_class : list of str
            see :attr:`gammapy.data.HDUIndexTable.VALID_HDU_CLASS`
        verbose : bool
            Print copied files
        overwrite : bool
            Overwrite
        """
        # TODO : Does rsync give any benefits here?

        outdir = make_path(outdir)
        if isinstance(obs_id, ObservationTable):
            obs_id = obs_id['OBS_ID'].data

        hdutable = self.hdu_table
        hdutable.add_index('OBS_ID')
        with hdutable.index_mode('discard_on_copy'):
            subhdutable = hdutable.loc[obs_id]
        if hdu_class is not None:
            subhdutable.add_index('HDU_CLASS')
            with subhdutable.index_mode('discard_on_copy'):
                subhdutable = subhdutable.loc[hdu_class]
        subobstable = self.obs_table.select_obs_id(obs_id)

        for idx in range(len(subhdutable)):
            # Changes to the file structure could be made here
            loc = subhdutable.location_info(idx)
            targetdir = outdir / loc.file_dir
            targetdir.mkdir(exist_ok=True, parents=True)
            cmd = ['cp', '-v'] if verbose else ['cp']
            if not overwrite:
                cmd += ['-n']
            cmd += [str(loc.path()), str(targetdir)]
            subprocess.call(cmd)

        subhdutable.write(str(outdir / self.DEFAULT_HDU_TABLE), format='fits', overwrite=overwrite)
        subobstable.write(str(outdir / self.DEFAULT_OBS_TABLE), format='fits', overwrite=overwrite)

    def data_summary(self, obs_id=None, summed=False):
        """Create a summary `~astropy.table.Table` with HDU size information.

        Parameters
        ----------
        obs_id : array-like
            Observation IDs to include in the summary
        summed : bool
            Sum up file size?
        """
        if obs_id is None:
            obs_id = self.obs_table['OBS_ID'].data

        hdut = self.hdu_table
        hdut.add_index('OBS_ID')
        subhdut = hdut.loc[obs_id]

        subhdut_grpd = subhdut.group_by('OBS_ID')
        colnames = subhdut_grpd.groups[0]['HDU_CLASS']
        temp = np.zeros(len(colnames), dtype=int)

        rows = []
        for key, group in zip(subhdut_grpd.groups.keys, subhdut_grpd.groups):
            # This is needed to get the column order right
            group.add_index('HDU_CLASS')
            vals = group.loc[colnames]['SIZE']
            if summed:
                temp = temp + vals
            else:
                rows.append(np.append(key['OBS_ID'], vals))

        if summed:
            rows.append(temp)
        else:
            colnames = np.append(['OBS_ID'], colnames)

        return Table(rows=rows, names=colnames)

    def check(self, checks='all'):
        """Check index tables and data files.

        This is a generator that yields a list of dicts.
        """
        checker = DataStoreChecker(self)
        return checker.run(checks=checks)


class DataStoreChecker(object):
    CHECKS = OrderedDict(
        obs_table='check_obs_table',
        hdu_table='check_hdu_table',
        observations='check_observations',
        consistency='check_consistency',
    )

    def __init__(self, data_store):
        self.data_store = data_store

    def run(self, checks='all'):
        if checks == 'all':
            checks = self.CHECKS.keys()

        unknown_checks = set(checks).difference(self.CHECKS.keys())
        if unknown_checks:
            raise ValueError('Unknown checks: {}'.format(unknown_checks))

        for check in checks:
            for record in getattr(self, self.CHECKS[check])():
                yield record

    def check_obs_table(self):
        """Checks for the observation index table."""
        t = self.data_store.obs_table
        m = t.meta
        if m.get('HDUCLAS1', '') != 'INDEX':
            yield {'type': 'error', 'hdu': 'obs-index', 'msg': 'Invalid header key. Must have HDUCLAS1=INDEX'}
        if m.get('HDUCLAS2', '') != 'OBS':
            yield {'type': 'error', 'hdu': 'obs-index', 'msg': 'Invalid header key. Must have HDUCLAS2=OBS'}

    def check_hdu_table(self):
        """Checks for the HDU index table."""
        t = self.data_store.hdu_table
        m = t.meta
        if m.get('HDUCLAS1', '') != 'INDEX':
            yield {'type': 'error', 'hdu': 'hdu-index', 'msg': 'Invalid header key. Must have HDUCLAS1=INDEX'}
        if m.get('HDUCLAS2', '') != 'HDU':
            yield {'type': 'error', 'hdu': 'hdu-index', 'msg': 'Invalid header key. Must have HDUCLAS2=HDU'}

        # Check that all HDU in the data files exist
        for idx in range(len(t)):
            location_info = t.location_info(idx)
            try:
                location_info.get_hdu()
            except KeyError:
                yield {'type': 'error', 'msg': 'HDU not found: {!r}'.format(location_info.__dict__)}

        # TODO: all HDU in the index table should be present

    def check_consistency(self):
        """Consistency checks between multiple HDUs"""
        # obs and HDU index should have the same OBS_ID
        obs_table_obs_id = set(self.data_store.obs_table['OBS_ID'])
        hdu_table_obs_id = set(self.data_store.hdu_table['OBS_ID'])
        if not obs_table_obs_id == hdu_table_obs_id:
            yield {'type': 'error', 'msg': 'Inconsistent OBS_ID in obs and HDU index tables'}

        # TODO: obs table and events header should have the same times

    def check_observations(self):
        """Perform some sanity checks for all observations."""
        for obs_id in self.data_store.obs_table['OBS_ID']:
            obs = self.data_store.obs(obs_id)

            for records in self.check_observation(obs):
                yield records

    @staticmethod
    def check_observation(obs):
        """Perform some basic sanity checks on this observation."""

        # Check that events table is not empty
        events = obs.events
        if len(events.table) == 0:
            yield {'type': 'error', 'obs_id': obs.obs_id, 'msg': 'Events table has zero rows'}

        # # Check that thresholds are meaningful for aeff
        # if self.aeff.meta['LO_THRES'] >= self.aeff.meta['HI_THRES']:
        #     messages.append('LO_THRES >= HI_THRES in effective area meta data')
        # # Check that maximum value of aeff is greater than zero
        # if np.max(self.aeff.data.data) <= 0:
        #     messages.append('maximum entry of effective area table <= 0')
        # # Check that maximum value of edisp matrix is greater than zero
        # if np.max(self.edisp.data.data) <= 0:
        #     messages.append('maximum entry of energy dispersion table <= 0')
        # # Check that thresholds are meaningful for psf
        # if self.psf.energy_thresh_lo >= self.psf.energy_thresh_hi:
        #     messages.append('LO_THRES >= HI_THRES in psf meta data')
