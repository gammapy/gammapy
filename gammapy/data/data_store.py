# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.io import fits
from .observation import ObservationTable
from ..utils.scripts import make_path
from ..extern.pathlib import Path

__all__ = [
    'DataStore',
]

log = logging.getLogger(__name__)


class DataStore(object):
    """Data store - convenient way to access and select data.

    Parameters
    ----------
    base_dir : str
        Base directory
    hdu_table : `~astropy.table.Table`
        File table
    obs_table : `~gammapy.data.ObservationTable`
        Observation table
    name : str
        Data store name
    """
    DEFAULT_HDU_TABLE = 'hdu-index.fits.gz'
    DEFAULT_OBS_TABLE = 'obs-index.fits.gz'
    DEFAULT_NAME = 'noname'

    def __init__(self, base_dir, hdu_table=None, obs_table=None, name=None):
        self.base_dir = make_path(base_dir)
        self.hdu_table = hdu_table
        self.obs_table = obs_table
        self.name = name

    @classmethod
    def from_files(cls, base_dir, hdu_table_filename=None, obs_table_filename=None, name=None):
        """Construct `DataStore` from file and obs table files."""
        if hdu_table_filename:
            log.debug('Reading {}'.format(hdu_table_filename))
            hdu_table = Table.read(str(hdu_table_filename), format='fits')
        else:
            hdu_table = None

        if obs_table_filename:
            log.debug('Reading {}'.format(str(obs_table_filename)))
            obs_table = ObservationTable.read(str(obs_table_filename), format='fits')
        else:
            obs_table = None

        return cls(
            base_dir=base_dir,
            hdu_table=hdu_table,
            obs_table=obs_table,
            name=name,
        )

    @classmethod
    def from_dir(cls, base_dir):
        """Create a `DataStore` from a directory.

        This assumes that the files and observations index tables
        are in the default locations.
        """
        config = dict(base_dir=base_dir)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        """Create a `DataStore` from a config dict."""
        base_dir = config['base_dir']
        name = config.get('name', cls.DEFAULT_NAME)
        hdu_table_filename = config.get('files', cls.DEFAULT_HDU_TABLE)
        obs_table_filename = config.get('observations', cls.DEFAULT_OBS_TABLE)

        hdu_table_filename = _find_file(hdu_table_filename, base_dir)
        obs_table_filename = _find_file(obs_table_filename, base_dir)

        return cls.from_files(
            base_dir=base_dir,
            hdu_table_filename=hdu_table_filename,
            obs_table_filename=obs_table_filename,
            name=name,
        )

    @classmethod
    def from_name(cls, name):
        """Convenience method to look up DataStore from DataManager."""
        # This import needs to be delayed to avoid a circular import
        # It can't be moved to the top of the file
        from .data_manager import DataManager
        dm = DataManager()
        return dm[name]

    @classmethod
    def from_all(cls, val):
        """Try different DataStore construtors.
        
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

    def info(self, stream=None):
        """Print some info"""
        if not stream:
            stream = sys.stdout

        print(file=stream)
        print('Data store summary info:', file=stream)
        print('name:       {}'.format(self.name), file=stream)
        print('base_dir:   {}'.format(self.base_dir), file=stream)
        print('obs table:  {}'.format(len(self.obs_table)), file=stream)
        print('file table: {}'.format(len(self.hdu_table)), file=stream)

    def filename(self, obs_id, filetype, abspath=True):
        """File name (relative to datastore `dir`).

        Parameters
        ----------
        obs_id : int
            Observation ID.
        filetype : {'events', 'aeff', 'edisp', 'psf', 'bkg'}
            Type of file.
        abspath : bool
            Absolute path (including data store base_dir)?

        Returns
        -------
        filename : str
            Filename (including the full absolute path or relative to base_dir)

        Examples
        --------
        TODO
        """
        _validate_filetype(filetype)

        val = None
        for row in self.hdu_table:
            id = row['OBS_ID']
            type = row['HDU_TYPE'].strip()
            if id == obs_id and type == filetype:
                val = row
                break

        if val is None:
            msg = 'File not in table: OBS_ID = {}, TYPE = {}'.format(obs_id, filetype)
            raise IndexError(msg)
        else:
            filedir = Path(val['FILE_DIR'].strip())
            filename = filedir / val['FILE_NAME'].strip()

        if abspath:
            filename = self.base_dir / filename

        return str(filename)

    def load(self, obs_id, filetype):
        """Load data file as appropriate object.

        Parameters
        ----------
        obs_id : int
            Observation ID.
        filetype : {'events', 'aeff', 'edisp', 'psf', 'bkg'}
            Type of file.

        Returns
        -------
        object : object
            Object depends on type, e.g. for `events` it's a `~gammapy.data.EventList`.
        """
        # Delayed imports to avoid circular import issues
        # Do not move to the top of the file!
        from ..data import EventList
        from ..background import Cube
        from .. import irf

        filename = self.filename(obs_id=obs_id, filetype=filetype)
        if filetype == 'events':
            return EventList.read(filename)
        elif filetype == 'aeff':
            return irf.EffectiveAreaTable2D.read(filename)
        elif filetype == 'edisp':
            return irf.EnergyDispersion2D.read(filename)
        elif filetype == 'psf':
            return irf.EnergyDependentMultiGaussPSF.read(filename)
        elif filetype == 'bkg':
            return Cube.read(filename)
        else:
            raise ValueError('Invalid filetype.')

    def load_all(self, filetype):
        """Load a given file type for all observations

        Parameters
        ----------
        filetype : {'events', 'aeff', 'edisp', 'psf', 'bkg'}
            Type of file.

        Returns
        -------
        list : python list of object
            Object depends on type, e.g. for `events` it is a list of `~gammapy.data.EventList`.
        """
        data_lists = []
        for obs_id in self.obs_table['OBS_ID']:
            data_list = self.load(obs_id, filetype)
            data_lists.append(data_list)
        return data_lists

    def check_integrity(self, logger):
        """Check integrity, i.e. whether index table and files match.
        """
        logger.info('Checking event list files')
        available = self.check_available_event_lists(logger)
        if np.any(~available):
            logger.warning('Number of missing event list files: {}'.format(np.invert(available).sum()))

            # TODO: implement better, more complete integrity checks.

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
            observation_table = ObservationTable(self.index_table)

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
        observation_table = self.index_table
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


def _find_file(filename, dir):
    """Find a file at an absolute or relative location.

    - First tries Path(filename)
    - Second tris Path(dir) / filename
    - Raises OSError if both don't exist.
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


def _validate_filetype(filetype):
    VALID_FILETYPES = ['events', 'aeff', 'edisp', 'psf', 'bkg']
    if not filetype in VALID_FILETYPES:
        msg = "Invalid filetype: '{}'. ".format(filetype)
        msg += 'Valid filetypes are: {}'.format(VALID_FILETYPES)
        raise ValueError(msg)


def _get_min_energy_threshold(observation_table, data_dir):
    """Get minimum energy threshold from a list of observations.

    TODO: make this a method from ObservationTable or DataStore?

    Parameters
    ----------
    observation_table : `~gammapy.data.ObservationTable`
        Observation list.
    data_dir : str
        Path to the data files.

    Parameters
    ----------
    min_energy_threshold : `~astropy.units.Quantity`
        Minimum energy threshold.
    """
    observatory_name = observation_table.meta['OBSERVATORY_NAME']
    if observatory_name == 'HESS':
        scheme = 'HESS'
    else:
        s_error = "Warning! Storage scheme for {}".format(observatory_name)
        s_error += "not implemented. Only H.E.S.S. scheme is available."
        raise ValueError(s_error)

    data_store = DataStore(dir=data_dir, scheme=scheme)
    aeff_table_files = data_store.make_table_of_files(observation_table,
                                                      filetypes=['effective area'])
    min_energy_threshold = Quantity(999., 'TeV')

    # loop over effective area files to get necessary infos from header
    for i_aeff_file in aeff_table_files['filename']:
        aeff_hdu = fits.open(i_aeff_file)['EFFECTIVE AREA']
        # TODO: Gammapy needs a class that interprets IRF files!!!
        if aeff_hdu.header.comments['LO_THRES'] == '[TeV]':
            energy_threshold_unit = 'TeV'
        energy_threshold = Quantity(aeff_hdu.header['LO_THRES'],
                                    energy_threshold_unit)
        # TODO: Aeff FITS files contain some header keywords,
        # where the units are stored in comments -> hard to parse!!!
        min_energy_threshold = min(min_energy_threshold, energy_threshold)

    return min_energy_threshold
