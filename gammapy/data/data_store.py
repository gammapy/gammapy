# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import logging
import numpy as np
from collections import OrderedDict
import subprocess
from ..extern.six.moves import UserList
from astropy.table import Table
from astropy.utils import lazyproperty
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from ..utils.scripts import make_path
from ..utils.energy import Energy
from ..utils.time import time_ref_from_dict
from ..utils.table import table_row_to_dict
from .obs_table import ObservationTable
from .hdu_index_table import HDUIndexTable
from .utils import _earth_location_from_dict
from ..irf import EnergyDependentTablePSF, IRFStacker, PSF3D

__all__ = [
    'DataStore',
    'DataStoreObservation',
    'ObservationList',
]

log = logging.getLogger(__name__)


class DataStore(object):
    """IACT data store.

    The data selection and access happens using an observation
    and an HDU index file as described at :ref:`gadf:iact-storage`.

    See :ref:`data_store` or :gp-extra-notebook:`data_iact` for usage examples.

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
                                 ' {}.\nErrors\nfrom_dir: {}\nfrom_name: {}'
                                 .format(val, e1, e2))

        return store

    def info(self, file=None):
        """Print some info."""
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
            Object depends on type, e.g. for `events` it is a list of `~gammapy.data.EventList`.
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
            Object depends on type, e.g. for `events` it is a list of `~gammapy.data.EventList`.
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


class DataStoreObservation(object):
    """IACT data store observation.

    See :ref:`data_store`

    Parameters
    ----------
    obs_id : int
        Observation ID
    data_store : `~gammapy.data.DataStore`
        Data store
    """

    def __init__(self, obs_id, data_store):
        # Assert that `obs_id` is available
        if obs_id not in data_store.obs_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in obs index table.'.format(obs_id))
        if obs_id not in data_store.hdu_table['OBS_ID']:
            raise ValueError('OBS_ID = {} not in HDU index table.'.format(obs_id))

        self.obs_id = obs_id
        self.data_store = data_store

    def __str__(self):
        """Generate summary info string."""
        ss = 'Info for OBS_ID = {}\n'.format(self.obs_id)
        ss += '- Start time: {:.2f}\n'.format(self.tstart.mjd)
        ss += '- Pointing pos: RA {:.2f} / Dec {:.2f}\n'.format(self.pointing_radec.ra, self.pointing_radec.dec)
        ss += '- Observation duration: {}\n'.format(self.observation_time_duration)
        ss += '- Dead-time fraction: {:5.3f} %\n'.format(100 * self.observation_dead_time_fraction)

        # TODO: Which target was observed?
        # TODO: print info about available HDUs for this observation ...
        return ss

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
        return self.data_store.hdu_table.hdu_location(
            obs_id=self.obs_id,
            hdu_type=hdu_type,
            hdu_class=hdu_class,
        )

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

    @property
    def events(self):
        """Load `gammapy.data.EventList` object (lazy property)."""
        return self.load(hdu_type='events')

    @property
    def gti(self):
        """Load `gammapy.data.GTI` object (lazy property)."""
        return self.load(hdu_type='gti')

    @property
    def aeff(self):
        """Load effective area object (lazy property)."""
        return self.load(hdu_type='aeff')

    @property
    def edisp(self):
        """Load energy dispersion object (lazy property)."""
        return self.load(hdu_type='edisp')

    @property
    def psf(self):
        """Load point spread function object (lazy property)."""
        return self.load(hdu_type='psf')

    @property
    def bkg(self):
        """Load background object (lazy property)."""
        return self.load(hdu_type='bkg')

    @lazyproperty
    def obs_info(self):
        """Observation information (`~collections.OrderedDict`)."""
        row = self.data_store.obs_table.select_obs_id(obs_id=self.obs_id)[0]
        return table_row_to_dict(row)

    @lazyproperty
    def tstart(self):
        """Observation start time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info['TSTART'].astype('float64'), 'second')
        time = met_ref + met
        return time

    @lazyproperty
    def tstop(self):
        """Observation stop time (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.data_store.obs_table.meta)
        met = Quantity(self.obs_info['TSTOP'].astype('float64'), 'second')
        time = met_ref + met
        return time

    @lazyproperty
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`).

        The wall time, including dead-time.
        """
        return Quantity(self.obs_info['ONTIME'], 'second')

    @lazyproperty
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`).

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return Quantity(self.obs_info['LIVETIME'], 'second')

    @lazyproperty
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float).

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        https://en.wikipedia.org/wiki/Dead_time
        https://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        return 1 - self.obs_info['DEADC']

    @lazyproperty
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.obs_info['RA_PNT'], self.obs_info['DEC_PNT']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @lazyproperty
    def pointing_altaz(self):
        """Pointing ALT / AZ sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        alt, az = self.obs_info['ALT_PNT'], self.obs_info['AZ_PNT']
        return SkyCoord(az, alt, unit='deg', frame='altaz')

    @lazyproperty
    def pointing_zen(self):
        """Pointing zenith angle sky (`~astropy.units.Quantity`)."""
        return Quantity(self.obs_info['ZEN_PNT'], unit='deg')

    @lazyproperty
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self.obs_info['RA_OBJ'], self.obs_info['DEC_OBJ']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @lazyproperty
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return _earth_location_from_dict(self.obs_info)

    @lazyproperty
    def muoneff(self):
        """Observation muon efficiency."""
        return self.obs_info['MUONEFF']

    def peek(self):
        """Quick-look plots in a few panels."""
        raise NotImplementedError

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

    def check_observation(self):
        """Perform some basic sanity checks on this observation.

        Returns
        -------
        results : list
            List with failure messages for the checks that failed
        """
        messages = []
        # Check that events table is not empty
        if len(self.events.table) == 0:
            messages.append('events table empty')
        # Check that thresholds are meaningful for aeff
        if self.aeff.meta['LO_THRES'] >= self.aeff.meta['HI_THRES']:
            messages.append('LO_THRES >= HI_THRES in effective area meta data')
        # Check that maximum value of aeff is greater than zero
        if np.max(self.aeff.data.data) <= 0:
            messages.append('maximum entry of effective area table <= 0')
        # Check that maximum value of edisp matrix is greater than zero
        if np.max(self.edisp.data.data) <= 0:
            messages.append('maximum entry of energy dispersion table <= 0')
        # Check that thresholds are meaningful for psf
        if self.psf.energy_thresh_lo >= self.psf.energy_thresh_hi:
            messages.append('LO_THRES >= HI_THRES in psf meta data')

        return messages


class ObservationList(UserList):
    """List of `~gammapy.data.DataStoreObservation`.

    Could be extended to hold a more generic class of observations.
    """

    def __str__(self):
        s = self.__class__.__name__ + '\n'
        s += 'Number of observations: {}\n'.format(len(self))
        for obs in self:
            s += str(obs)
        return s

    def make_mean_psf(self, position, energy=None, rad=None):
        """Compute mean energy-dependent PSF.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position at which to compute the PSF
        energy : `~astropy.units.Quantity`
            1-dim energy array for the output PSF.
            If none is given, the energy array of the PSF from the first
            observation is used.
        rad : `~astropy.coordinates.Angle`
            1-dim offset wrt source position array for the output PSF.
            If none is given, the energy array of the PSF from the first
            observation is used.

        Returns
        -------
        psf : `~gammapy.irf.EnergyDependentTablePSF`
            Mean PSF
        """
        psf = self[0].make_psf(position, energy, rad)

        rad = rad or psf.rad
        energy = energy or psf.energy
        exposure = psf.exposure
        psf_value = psf.psf_value.T * psf.exposure

        for obs in self[1:]:
            psf = obs.make_psf(position, energy, rad)
            exposure += psf.exposure
            psf_value += psf.psf_value.T * psf.exposure

        psf_value /= exposure
        psf_tot = EnergyDependentTablePSF(energy=energy, rad=rad,
                                          exposure=exposure,
                                          psf_value=psf_value.T)
        return psf_tot

    def make_mean_edisp(self, position, e_true, e_reco,
                        low_reco_threshold=Energy(0.002, "TeV"),
                        high_reco_threshold=Energy(150, "TeV")):
        """Compute mean energy dispersion.

        Compute the mean edisp of a set of observations j at a given position

        The stacking is implemented in :func:`~gammapy.irf.IRFStacker.stack_edisp`

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position at which to compute the mean EDISP
        e_true : `~gammapy.utils.energy.EnergyBounds`
            True energy axis
        e_reco : `~gammapy.utils.energy.EnergyBounds`
            Reconstructed energy axis
        low_reco_threshold : `~gammapy.utils.energy.Energy`
            low energy threshold in reco energy, default 0.002 TeV
        high_reco_threshold : `~gammapy.utils.energy.Energy`
            high energy threshold in reco energy , default 150 TeV

        Returns
        -------
        stacked_edisp : `~gammapy.irf.EnergyDispersion`
            Stacked EDISP for a set of observation
        """
        list_aeff = []
        list_edisp = []
        list_livetime = []
        list_low_threshold = [low_reco_threshold] * len(self)
        list_high_threshold = [high_reco_threshold] * len(self)

        for obs in self:
            offset = position.separation(obs.pointing_radec)
            list_aeff.append(obs.aeff.to_effective_area_table(offset,
                                                              energy=e_true))
            list_edisp.append(obs.edisp.to_energy_dispersion(offset,
                                                             e_reco=e_reco,
                                                             e_true=e_true))
            list_livetime.append(obs.observation_live_time_duration)

        irf_stack = IRFStacker(list_aeff=list_aeff, list_edisp=list_edisp,
                               list_livetime=list_livetime,
                               list_low_threshold=list_low_threshold,
                               list_high_threshold=list_high_threshold)
        irf_stack.stack_edisp()

        return irf_stack.stacked_edisp
