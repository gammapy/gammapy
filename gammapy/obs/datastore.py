# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from ..catalog import skycoord_from_table
from ..obs import ObservationTable

__all__ = ['DataStore',
           'DataStoreIndexTable',
           'convert_obs_list_format_to_gammapy',
           ]


def _make_filename_hess_scheme(obs_id, filetype='events'):
    """Make filename string for the HESS storage scheme.

    Parameters
    ----------
    obs_id : int
        Observation ID.
    filetype : {'events', 'effective area', 'energy dispersion', 'psf', 'background'}
        Type of file.

    Examples
    --------
    >>> _make_filename_hess_scheme(obs_id=89565, filetype='effective area')
    'run089400-089599/run089565/hess_aeff_2d_089565.fits.gz'
    """
    obs_id_min = obs_id - (obs_id % 200)
    obs_id_max = obs_id_min + 199
    group_folder = 'run{:06d}-{:06d}'.format(obs_id_min, obs_id_max)
    obs_folder = 'run{:06d}'.format(obs_id)

    if filetype == 'events':
        label = 'events'
    elif filetype == 'psf':
        label = 'psf_king'
    elif filetype == 'effective area':
        label = 'aeff_2d'
    elif filetype == 'background':
        label = 'bkg_offruns'
    elif filetype == 'energy dispersion':
        label = 'edisp_2d'

    else:
        raise ValueError('Unknown filetype: {}'.format(filetype))

    filename = 'hess_{}_{:06d}.fits.gz'.format(label, obs_id)

    return os.path.join(group_folder, obs_folder, filename)


class DataStoreIndexTable(ObservationTable):
    """Data store index table.

    The index table is a FITS file that stores which observations
    are available and what their most important parameters are.

    This makes it possible to select observations of interest and find out
    what data is available without opening up thousands of FITS files
    that contain the event list and IRFs and have similar information in the
    FITS header.

    TODO: how is this different from an `gammapy.obs.ObservationTable`?
    Can they be combined or code be shared?
    (I think we want both, but the `DataStoreIndexTable` contains more info
    like event list and IRF file names and basic info
    ... maybe it should be a sub-class of `gammapy.obs.ObservationTable`?)

    """
    # For now I've decided to not do the cleanup in `__init__`,
    # but instead in `read`.
    # See https://groups.google.com/d/msg/astropy-dev/0EaOw9peWSk/MSjH7q_7htoJ
    # def __init__(self, *args, **kwargs):
    #     super(DataStoreIndexTable, self).__init__(*args, **kwargs)
    #     self._fixes()

    @classmethod
    def read(cls, scheme, *args, **kwargs):
        """Read from FITS file. See `~astropy.table.Table.read`."""
        table = Table.read(*args, **kwargs)
        # convert to the gammapy format
        table = convert_obs_list_format_to_gammapy(table, scheme)
        table = cls(table)
        table._init_cleanup(scheme)
        return table

    def _init_cleanup(self, scheme):
        # Add useful extra columns
        if not set(['GLON', 'GLAT']).issubset(self.colnames):
            skycoord = skycoord_from_table(self).galactic
            self['GLON'] = skycoord.l.to('degree')
            self['GLAT'] = skycoord.b.to('degree')

    def summary(self):
        ss = 'Data store index table summary:\n'
        ss += 'Number of observations: {}\n'.format(len(self))
        obs_id = self['OBS_ID']
        ss += 'Observation IDs: {} to {}\n'.format(obs_id.min(), obs_id.max())
        ss += 'Dates: {} to {}\n'.format('TODO', 'TODO')
        return ss

    @property
    def radec(self):
        """ICRS sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        return SkyCoord(self['RA'], self['DEC'], unit='deg', frame='icrs')

    @property
    def galactic(self):
        """Galactic sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        return SkyCoord(self['GLON'], self['GLAT'], unit='deg', frame='galactic')


class DataStore(object):
    """Data store - convenient way to access and select data.

    This is an ad-hoc prototype implementation for HESS of what will be the "archive"
    and "archive interface" for CTA.

    TODO: add methods to sync with remote datastore...

    Parameters
    ----------
    dir : str
        Data store directory on user machine.
    scheme : {'HESS'}
        Scheme for file naming and organisation.
    """

    def __init__(self, dir, scheme='HESS'):
        self.dir = dir
        self.index_table_filename = 'runinfo.fits'
        filename = os.path.join(dir, self.index_table_filename)
        #print('Reading {}'.format(filename))
        self.index_table = DataStoreIndexTable.read(scheme, filename)
        self.scheme = scheme

    def info(self):
        """Summary info string."""
        ss = 'Data store summary info:\n'
        ss += 'Directory: {}\n'.format(self.dir)
        ss += 'Index table: {}\n'.format(self.index_table_filename)
        ss += 'Scheme: {}\n'.format(self.scheme)
        #Does not exist
        #ss += self.index_table.info()
        return ss

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

        observation_table = convert_obs_list_format_to_gammapy(observation_table,
                                                               self.scheme)

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

    def make_summary_plots(self):
        """Make some plots summarising the available observations.

        E.g. histograms of time, run duration, muon efficiency, zenith angle, ...
        Aitoff plot showing run locations.
        """
        raise NotImplementedError

    def filename(self, obs_id, filetype='events', abspath=True):
        """File name (relative to datastore `dir`).

        Parameters
        ----------
        obs_id : int
            Observation ID.
        filetype : {'events', 'effective area', 'energy dispersion', 'psf', 'background'}
            Type of file.
        abspath : bool
            Absolute path (including data store dir)?

        Returns
        -------
        filename : str
            Filename (including the directory path).
        """
        scheme = self.scheme

        if scheme == 'HESS':
            filename = _make_filename_hess_scheme(obs_id, filetype)
        else:
            raise ValueError('Invalid scheme: {}'.format(scheme))

        if abspath:
            return os.path.join(self.dir, filename)
        else:
            return filename

    def make_observation_table(self, selection=None):
        """Make an observation table, applying some selection.

        Wrapper function for `~gammapy.obs.ObservationTable.select_observations`.
        For details, please refer to the doc on the mentioned function.

        Parameters
        ----------
        selection : dict
            Dictionary with a few keywords for applying selection cuts.

        Returns
        -------
        table : `~gammapy.obs.ObservationTable`
            Observation table after selection.

        Examples
        --------
        >>> selection = dict(type='sky_box', frame='icrs',
        ...                  lon=Angle([150, 300], 'degree'),
        ...                  lat=Angle([-50, 0], 'degree'),
        ...                  border=Angle(2, 'degree'))
        >>> run_list = data_store.make_observation_table(selection)
        """
        table = self.index_table

        if selection:
            table = table.select_observations(selection)

        return table

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
            if not os.path.isfile(filename):
                file_available[ii] = False
                if logger:
                    logger.warning('For OBS_ID = {:06d} the event list file is missing: {}'
                                   ''.format(obs_id, filename))

        return file_available


def convert_obs_list_format_to_gammapy(obs_list, scheme):
    """Convert oservation list from supported formats to Gammapy format.

    This script calls the corresponding format converter, depending
    on the value of the **scheme** parameter, in order to make
    observation lists from different experiments comply to the format
    described in :ref:`dataformats_observation_lists`.

    Curretly only the H.E.S.S. scheme is supported.

    Parameters
    ----------
    obs_list : `~astropy.table.Table`
        Observation list to convert.
    scheme : {'HESS'}
        Scheme for file naming and organisation.
        Format of the input observation list to convert.

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
        Converted observation list.
    """
    if scheme == 'HESS':
        return _convert_obs_list_hess_to_gammapy(obs_list)
    else:
        raise ValueError('Invalid scheme: {}'.format(scheme))


def _convert_obs_list_hess_to_gammapy(obs_list):
    """Convert oservation list from H.E.S.S. format to Gammapy format.

    The H.E.S.S. observation lists are produced following a different
    format as Gammapy. This function should convert the
    format to the one described in :ref:`dataformats_observation_lists`.

    This script renames the columns and edits the header keywords of
    the observation lists. Columns and header keywords not defined in
    :ref:`dataformats_observation_lists` are left unchanged.

    This function has no tests implemented, since the H.E.S.S. data
    is private.

    Parameters
    ----------
    obs_list : `~astropy.table.Table`
        Observation list to convert.

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
        Converted observation list.
    """
    try:
        observatory = obs_list.meta['OBSERVATORY_NAME']
        if observatory == 'HESS':
            # already converted
            return obs_list
        else:
            s_error = "Expected OBSERVATORY_NAME = HESS, "
            s_error += "but got {}".format(obs_list.meta['OBSERVATORY_NAME'])
            raise ValueError(s_error)
    except KeyError:
        # needs conversion
        pass

    obs_table = ObservationTable(obs_list)

    # rename column names
    renames = [('RA_PNT', 'RA'),
               ('DEC_PNT', 'DEC'),
               ('ALT_PNT', 'ALT'),
               ('AZ_PNT', 'AZ'),
               ('MUONEFF', 'MUON_EFFICIENCY'),
               ('ONTIME', 'TIME_OBSERVATION'),
               ('LIVETIME', 'TIME_LIVE'),
               ('TSTART', 'TIME_START '),
               ('TSTOP', 'TIME_STOP'),
               ('TRGRATE', 'TRIGGER_RATE'),
               ('MEANTEMP', 'MEAN_TEMPERATURE'),
               ('TELLIST', 'TEL_LIST')
               ]
    for name, new_name in renames:
        obs_table.rename_column(name, new_name)

    # add missing header entries
    obs_table.meta['OBSERVATORY_NAME'] = 'HESS'
    obs_table.meta['TIME_FORMAT'] = 'relative'

    return obs_table
