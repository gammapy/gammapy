# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import numpy as np
from astropy.table import Table
from ..extern.pathlib import Path
from ..obs import ObservationTable

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
    file_table : `~astropy.table.Table`
        File table
    obs_table : `~gammapy.obs.ObservationTable`
        Observation table
    name : str
        Data store name
    """
    DEFAULT_FILE_TABLE = 'files.fits.gz'
    DEFAULT_OBS_TABLE = 'observations.fits.gz'
    DEFAULT_NAME = 'noname'

    def __init__(self, base_dir, file_table=None, obs_table=None, name=None):
        self.base_dir = Path(base_dir)
        self.file_table = file_table
        self.obs_table = obs_table
        self.name = name

    @classmethod
    def from_files(cls, base_dir, file_table_filename=None, obs_table_filename=None, name=None):
        """Construct `DataStore` from file and obs table files."""
        if file_table_filename:
            log.debug('Reading {}'.format(file_table_filename))
            file_table = Table.read(str(file_table_filename), format='fits')
        else:
            file_table = None

        if obs_table_filename:
            log.debug('Reading {}'.format(str(obs_table_filename)))
            obs_table = ObservationTable.read(str(obs_table_filename), format='fits')
        else:
            obs_table = None

        return cls(
            base_dir=base_dir,
            file_table=file_table,
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
        file_table_filename = config.get('files', cls.DEFAULT_FILE_TABLE)
        obs_table_filename = config.get('observations', cls.DEFAULT_OBS_TABLE)

        file_table_filename = _find_file(file_table_filename, base_dir)
        obs_table_filename = _find_file(obs_table_filename, base_dir)

        return cls.from_files(
            base_dir=base_dir,
            file_table_filename=file_table_filename,
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

    def info(self, stream=None):
        """Print some info"""
        if not stream:
            stream = sys.stdout

        print(file=stream)
        print('Data store summary info:', file=stream)
        print('name: {}'.format(self.name), file=stream)
        print('base_dir: {}'.format(self.base_dir), file=stream)
        print('observations: {}'.format(len(self.obs_table)), file=stream)
        print('files: {}'.format(len(self.file_table)), file=stream)

    def filename(self, obs_id, filetype, abspath=True):
        """File name (relative to datastore `dir`).

        Parameters
        ----------
        obs_id : int
            Observation ID.
        filetype : {'events', 'aeff', 'edisp', 'psf', 'background'}
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

        t = self.file_table
        mask = (t['OBS_ID'] == obs_id) & (t['TYPE'] == filetype)
        try:
            idx = np.where(mask)[0][0]
        except IndexError:
            msg = 'File not in table: OBS_ID = {}, TYPE = {}'.format(obs_id, filetype)
            raise IndexError(msg)

        filename = t['NAME'][idx]

        if abspath:
            filename = self.base_dir / filename

        return str(filename)

    def load(self, obs_id, filetype):
        """Load data file as appropriate object.

        Parameters
        ----------
        obs_id : int
            Observation ID.
        filetype : {'events', 'aeff', 'edisp', 'psf', 'background'}
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
        elif filetype == 'background':
            return Cube.read(filename)
        else:
            raise ValueError('Invalid filetype.')

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
        observation_table : `~gammapy.obs.ObservationTable` or None
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
            if not Path(filename).is_file():
                file_available[ii] = False
                if logger:
                    logger.warning('For OBS_ID = {:06d} the event list file is missing: {}'
                                   ''.format(obs_id, filename))

        return file_available


def _find_file(filename, dir):
    """Find a file at an absolute or relative location.

    - First tries Path(filename)
    - Second tris Path(dir) / filename
    - Raises FileNotFoundError if both don't exist.
    """
    path1 = Path(filename)
    path2 = Path(dir) / filename
    if path1.is_file():
        filename = path1
    elif path2.is_file():
        filename = path2
    else:
        raise FileNotFoundError('File not found at {} or {}'.format(path1, path2))
    return filename


def _validate_filetype(filetype):
    VALID_FILETYPES = ['events', 'aeff', 'edisp', 'psf', 'background']
    if not filetype in VALID_FILETYPES:
        msg = "Invalid filetype: '{}'. ".format(filetype)
        msg += 'Valid filetypes are: {}'.format(VALID_FILETYPES)
        raise ValueError(msg)


def _get_min_energy_threshold(observation_table, data_dir):
    """Get minimum energy threshold from a list of observations.

    TODO: make this a method from ObservationTable or DataStore?

    Parameters
    ----------
    observation_table : `~gammapy.obs.ObservationTable`
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

