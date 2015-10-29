# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, AltAz
from astropy.table import Table
from ..extern.pathlib import Path
from ..image import wcs_histogram2d
from ..data import GoodTimeIntervals, TelescopeArray
from ..data import InvalidDataError
from ..time import time_ref_from_dict
from ..background.reflected import ReflectedRegionMaker
from .utils import _earth_location_from_dict

__all__ = [
    'EventList',
    'EventListDataset',
    'EventListDatasetChecker',
    'event_lists_to_counts_image',
]

log = logging.getLogger(__name__)


class EventList(Table):
    """Event list `~astropy.table.Table`.

    The most important reconstructed event parameters
    are available as the following columns:

    - ``TIME`` - Mission elapsed time (sec)
    - ``RA``, ``DEC`` - FK5 J2000 (or ICRS?) position (deg)
    - ``ENERGY`` - Energy (usually MeV for Fermi and TeV for IACTs)

    Other optional (columns) that are sometimes useful for high-level analysis:

    - ``GLON``, ``GLAT`` - Galactic coordinates (deg)
    - ``DETX``, ``DETY`` - Field of view coordinates (radian?)

    Note that when reading data for analysis you shouldn't use those
    values directly, but access them via properties which create objects
    of the appropriate class:

    - `time` for ``TIME``
    - `radec` for ``RA``, ``DEC``
    - `energy` for ``ENERGY``
    - `galactic` for ``GLON``, ``GLAT``
    """

    @property
    def summary(self):
        """Summary info string."""
        s = '---> Event list info:\n'
        # TODO: Which telescope?

        # When and how long was the observation?
        s += '- Observation duration: {}\n'.format(self.observation_time_duration)
        s += '- Dead-time fraction: {:5.3f} %\n'.format(100 * self.observation_dead_time_fraction)

        # TODO: Which target was observed?

        s += '-- Event info:\n'
        s += '- Number of events: {}\n'.format(len(self))
        # TODO: add time, RA, DEC and if present GLON, GLAT info ...
        s += '- Median energy: {}\n'.format(np.median(self.energy))
        # TODO: azimuth should be circular median
        s += '- Median azimuth: {}\n'.format(np.median(self['AZ']))
        s += '- Median altitude: {}\n'.format(np.median(self['ALT']))

        return s

    @property
    def time(self):
        """Event times (`~astropy.time.Time`)

        Notes
        -----
        Times are automatically converted to 64-bit floats.
        With 32-bit floats times will be incorrect by a few seconds
        when e.g. adding them to the reference time.
        """
        met_ref = time_ref_from_dict(self.meta)
        met = Quantity(self['TIME'].astype('float64'), 'second')
        time = met_ref + met
        return time

    @property
    def radec(self):
        """Event RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)

        TODO: the `radec` and `galactic` properties should be cached as table columns
        """
        lon, lat = self['RA'], self['DEC']
        return SkyCoord(lon, lat, unit='deg', frame='fk5')

    @property
    def galactic(self):
        """Event Galactic sky coordinates (`~astropy.coordinates.SkyCoord`)

        Note: uses the ``GLON`` and ``GLAT`` columns.
        If only ``RA`` and ``DEC`` are present use the explicit
        ``event_list.radec.to('galactic')`` instead.
        """
        self.add_galactic_columns()
        lon, lat = self['GLON'], self['GLAT']
        return SkyCoord(lon, lat, unit='deg', frame='galactic')

    def add_galactic_columns(self):
        """Add Galactic coordinate columns to the table.

        Adds the following columns to the table if not already present:
        - "GLON" - Galactic longitude (deg)
        - "GLAT" - Galactic latitude (deg)
        """
        if set(['GLON', 'GLAT']).issubset(self.colnames):
            return

        galactic = self.radec.galactic
        self['GLON'] = galactic.l.degree
        self['GLAT'] = galactic.b.degree

    @property
    def altaz(self):
        """Event horizontal sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        time = self.time
        location = self.observatory_earth_location
        altaz_frame = AltAz(obstime=time, location=location)

        lon, lat = self['AZ'], self['ALT']
        return SkyCoord(lon, lat, unit='deg', frame=altaz_frame)

    @property
    def energy(self):
        """Event energies (`~astropy.units.Quantity`)"""
        energy = self['ENERGY']
        return Quantity(energy, self.meta['EUNIT'])

    @property
    def target_radec(self):
        """Target RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        lon, lat = self.meta['RA_OBJ'], self.meta['DEC_OBJ']
        return SkyCoord(lon, lat, unit='deg', frame='fk5')

    @property
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)"""
        lon, lat = self.meta['RA_PNT'], self.meta['DEC_PNT']
        return SkyCoord(lon, lat, unit='deg', frame='fk5')

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        return _earth_location_from_dict(self.meta)

    # TODO: I'm not sure how to best exposure header data
    # as quantities ... maybe expose them on `meta` or
    # a completely separate namespace?
    # For now I'm taking very verbose names ...

    @property
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`)

        The wall time, including dead-time.
        """
        return Quantity(self.meta['ONTIME'], 'second')

    @property
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`)

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return Quantity(self.meta['LIVETIME'], 'second')

    @property
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
        return 1 - self.meta['DEADC']

    def select_energy(self, energy_band):
        """Select events in energy band.

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
            Energy band ``[energy_min, energy_max)``

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.

        Examples
        --------
        >>> from astropy.units import Quantity
        >>> from gammapy.data import EventList
        >>> event_list = EventList.read('events.fits')
        >>> energy_band = Quantity([1, 20], 'TeV')
        >>> event_list = event_list.select_energy()
        """
        energy = self.energy
        mask = (energy_band[0] <= energy)
        mask &= (energy < energy_band[1])
        return self[mask]

    def select_time(self, time_interval):
        """Select events in interval.
        """
        time = self.time
        mask = (time_interval[0] <= time)
        mask &= (time < time_interval[1])
        return self[mask]

    def select_sky_cone(self, center, radius):
        """Select events in sky circle.

        Parameters
        ----------
        center : `~astropy.coordinates.SkyCoord`
            Sky circle center
        radius : `~astropy.coordinates.Angle`
            Sky circle radius

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        position = self.radec
        separation = center.separation(position)
        mask = separation < radius
        return self[mask]

    def select_sky_ring(self, center, inner_radius, outer_radius):
        """Select events in sky circle.

        Parameters
        ----------
        center : `~astropy.coordinates.SkyCoord`
            Sky ring center
        inner_radius : `~astropy.coordinates.Angle`
            Sky ring inner radius
        outer_radius : `~astropy.coordinates.Angle`
            Sky ring outer radius

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """

        position = self.radec
        separation = center.separation(position)
        mask1 = inner_radius < separation
        mask2 = separation < outer_radius
        mask = mask1 * mask2

        return self[mask]

    def select_sky_box(self, lon_lim, lat_lim, frame='icrs'):
        """Select events in sky box.

        TODO: move `gammapy.catalog.select_sky_box` to gammapy.utils.
        """
        from ..catalog import select_sky_box
        return select_sky_box(self, lon_lim, lat_lim, frame)

    def select_reflected_regions(self, on_center, on_radius, exclusion,
                                 angle_increment=0.1):
        """Select events from reflected regions.

        More info on the reflected regions background estimation methond
        can be found in [Berge2007]_

        Parameters
        ----------
        on_center : `~astropy.coordinates.SkyCoord`
            ON region center
        on_radius : `~astropy.coordinates.Angle`
            ON region radius
        exclusion : ImageHDU
            Excluded regions mask
        angle_increment : float (optional)
            Angle between two reflected regions

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        
        point = self.pointing_radec
        fov = dict(x=point.ra.value, y=point.dec.value)
        rr_maker = ReflectedRegionMaker(exclusion, fov, angle_increment)
        x_on = on_center.rad.value
        y_on = on_center.dec.value
        r_on = on_radius.value
        rr_maker.compute(x_on, y_on, r_on)
        from IPython import embed; embed()
        

    def fill_counts_image(self, image):
        """Fill events in counts image.

        TODO: what's a good API here to support ImageHDU and Header as input?

        Parameters
        ----------
        image : `~astropy.io.fits.ImageHDU`
            Image HDU

        Returns
        -------
        image : `~astropy.io.fits.ImageHDU`
            Input image with changed data (event count added)

        See also
        --------
        EventList.fill_counts_header
        """
        header = image.header
        lon, lat = self._get_lon_lat(header)
        counts_image = wcs_histogram2d(header, lon, lat)
        image.data += counts_image.data
        return image

    def fill_counts_header(self, header):
        """Fill events in counts image specified by a FITS header.

        TODO: document. Is this a good API?

        See also
        --------
        EventList.fill_counts_image
        """
        lon, lat = self._get_lon_lat(header)
        counts_image = wcs_histogram2d(header, lon, lat)
        return counts_image

    def _get_lon_lat(self, header):
        # TODO: this frame detection should be moved to a utility function
        CTYPE1 = header['CTYPE1']
        if 'RA' in CTYPE1:
            pos = self.radec
            lon = pos.ra.degree
            lat = pos.dec.degree
        elif 'GLON' in CTYPE1:
            pos = self.galactic
            lon = pos.l.degree
            lat = pos.b.degree
        else:
            raise ValueError('CTYPE1 = {} is not supported.'.format(CTYPE1))

        return lon, lat

    def peek(self):
        """Summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))
        self.plot_image(ax=axes[0])
        self.plot_energy_dependence(ax=axes[1])
        self.plot_offset_dependence(ax=axes[2])
        plt.tight_layout()
        plt.show()



class EventListDataset(object):
    """Event list dataset (event list plus some extra info).

    TODO: I'm not sure if IRFs should be included in this
    class or if an extra container class should be added.

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list table
    telescope_array : `~gammapy.data.TelescopeArray`
        Telescope array info
    good_time_intervals : `~gammapy.data.GoodTimeIntervals`
        Observation time interval info
    """

    def __init__(self, event_list,
                 telescope_array=None,
                 good_time_intervals=None):
        self.event_list = event_list
        self.telescope_array = telescope_array
        self.good_time_intervals = good_time_intervals

    @classmethod
    def from_hdu_list(cls, hdu_list):
        """Create `EventList` from a `~astropy.io.fits.HDUList`.
        """
        # TODO: This doesn't work because FITS / Table is not integrated.
        # Maybe the easiest solution for now it to write the hdu_list
        # to an in-memory buffer with StringIO and then read it
        # back using Table.read()?
        raise NotImplementedError
        event_list = EventList.from_hdu(hdu_list['EVENTS'])
        telescope_array = TelescopeArray.from_hdu(hdu_list['TELARRAY'])
        good_time_intervals = GoodTimeIntervals.from_hdu(hdu_list['GTI'])

        return cls(event_list, telescope_array, good_time_intervals)

    @classmethod
    def read(cls, filename):
        """Read event list from FITS file.
        """
        # return EventList.from_hdu_list(fits.open(filename))
        event_list = EventList.read(filename, hdu='EVENTS')
        try:
            telescope_array = TelescopeArray.read(filename, hdu='TELARRAY')
        except KeyError:
            telescope_array = None
            # self.logger.debug('No TELARRAY extension')

        try:
            good_time_intervals = GoodTimeIntervals.read(filename, hdu='GTI')
        except KeyError:
            good_time_intervals = None

        return cls(event_list, telescope_array, good_time_intervals)

    @classmethod
    def vstack_from_files(cls, filenames, logger=None):
        """Stack event lists vertically (combine events and GTIs).

        This function stacks (a.k.a. concatenates) event lists.
        E.g. if you have one event list with 100 events (i.e. 100 rows)
        and another with 42 events, the output event list will have 142 events.

        It also stacks the GTIs so that exposure computations are still
        possible using the stacked event list.


        At the moment this can require a lot of memory.
        All event lists are loaded into memory at the same time.

        TODO: implement and benchmark different a more efficient method:
        Get number of rows from headers, pre-allocate a large table,
        open files one by one and fill correct rows.

        TODO: handle header keywords "correctly".
        At the moment the output event list header keywords are copies of
        the values from the first observation, i.e. meaningless.
        Here's a (probably incomplete) list of values we should handle
        (usually by computing the min, max or mean or removing it):
        - OBS_ID
        - DATE_OBS, DATE_END
        - TIME_OBS, TIME_END
        - TSTART, TSTOP
        - LIVETIME, DEADC
        - RA_PNT, DEC_PNT
        - ALT_PNT, AZ_PNT


        Parameters
        ----------
        filenames : list of str
            List of event list filenames

        Returns
        -------
        event_list_dataset : `~gammapy.data.EventListDataset`

        """
        total_filesize = 0
        for filename in filenames:
            total_filesize += Path(filename).stat().st_size

        if logger:
            logger.info('Number of files to stack: {}'.format(len(filenames)))
            logger.info('Total filesize: {:.2f} MB'.format(total_filesize / 1024. ** 2))
            logger.info('Reading event list files ...')

        event_lists = []
        gtis = []
        from astropy.utils.console import ProgressBar
        for filename in ProgressBar(filenames):
            # logger.info('Reading {}'.format(filename))
            event_list = Table.read(filename, hdu='EVENTS')

            # TODO: Remove and modify header keywords for stacked event list
            meta_del = ['OBS_ID', 'OBJECT']
            meta_mod = ['DATE_OBS', 'DATE_END', 'TIME_OBS', 'TIME_END']

            gti = Table.read(filename, hdu='GTI')
            event_lists.append(event_list)
            gtis.append(gti)

        from astropy.table import vstack as vstack_tables
        total_event_list = vstack_tables(event_lists, metadata_conflicts='silent')
        total_gti = vstack_tables(gtis, metadata_conflicts='silent')

        total_event_list.meta['EVTSTACK'] = 'yes'
        total_gti.meta['EVTSTACK'] = 'yes'

        return cls(event_list=total_event_list, good_time_intervals=total_gti)

    def write(self, *args, **kwargs):
        """Write to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

    def to_fits(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with EVENTS and GTI extension.
        """
        # TODO: simplify when Table / FITS integration improves:
        # https://github.com/astropy/astropy/issues/2632#issuecomment-70281392
        # TODO: I think this makes an in-memory copy, i.e. is inefficient.
        # Can we avoid this?
        hdu_list = fits.HDUList()


        # TODO:
        del self.event_list['TELMASK']

        data = self.event_list.as_array()
        header = fits.Header()
        header.update(self.event_list.meta)
        hdu_list.append(fits.BinTableHDU(data=data, header=header, name='EVENTS'))

        data = self.good_time_intervals.as_array()
        header = fits.Header()
        header.update(self.good_time_intervals.meta)
        hdu_list.append(fits.BinTableHDU(data, header=header, name='GTI'))

        return hdu_list

    @property
    def info(self):
        """Summary info string."""
        s = '===> Event list dataset information:\n'
        s += self.event_list.summary
        s += self.telescope_array.summary
        s += self.good_time_intervals.summary
        s += '- telescopes: {}\n'.format(len(self.telescope_array))
        s += '- good time intervals: {}\n'.format(len(self.good_time_intervals))
        return s

    def check(self, checks='all'):
        """Check if format and content is ok.

        This is a convenience method that instantiates
        and runs a `~gammapy.data.EventListDatasetChecker` ...
        if you want more options use this way to use it:

        >>> from gammapy.data import EventListDatasetChecker
        >>> checker = EventListDatasetChecker(event_list, ...)
        >>> checker.run(which, ...)  #

        Parameters
        ----------
        checks : list of str or 'all'
            Which checks to run (see list in
            `~gammapy.data.EventListDatasetChecker.run` docstring).

        Returns
        -------
        ok : bool
            Everything ok?
        """
        checker = EventListDatasetChecker(self)
        return checker.run(checks)


class EventListDatasetChecker(object):
    """Event list dataset checker.

    TODO: link to defining standard documents,
     especially the CTA event list spec.

    Having such a checker is useful at the moment because
    the CTA data formats are quickly evolving and there's
    various sources of event list data, e.g. exporters are
    being written for the existing IACTs and simulators
    are being written for CTA.

    Parameters
    ----------
    event_list_dataset : `~gammapy.data.EventListDataset`
        Event list dataset
    logger : `logging.Logger` or None
        Logger to use (use module-level Gammapy logger by default)
    """
    _AVAILABLE_CHECKS = OrderedDict(
        misc='check_misc',
        times='check_times',
        coordinates='check_coordinates',
    )

    accuracy = OrderedDict(
        angle=Angle('1 arcsec'),
        time=Quantity(1, 'microsecond'),

    )

    def __init__(self, event_list_dataset, logger=None):
        self.dset = event_list_dataset
        if logger:
            self.logger = logger
        else:
            self.logger = log

    def run(self, checks='all'):
        """Run checks.

        Available checks: {...}

        Parameters
        ----------
        checks : list of str or "all"
            Which checks to run

        Returns
        -------
        ok : bool
            Everything ok?
        """
        if checks == 'all':
            checks = self._AVAILABLE_CHECKS.keys()

        unknown_checks = set(checks).difference(self._AVAILABLE_CHECKS.keys())
        if unknown_checks:
            raise ValueError('Unknown checks: {}'.format(unknown_checks))

        ok = True
        for check in checks:
            check_method = getattr(self, self._AVAILABLE_CHECKS[check])
            ok &= check_method()

        return ok

    def check_misc(self):
        """Check misc basic stuff."""
        ok = True

        required_meta = ['TELESCOP', 'OBS_ID']
        missing_meta = set(required_meta) - set(self.dset.event_list.meta)
        if missing_meta:
            ok = False
            self.logger.error('Missing meta info: {}'.format(missing_meta))

        # TODO: implement more basic checks that all required info is present.

        return ok

    def _check_times_gtis(self):
        """Check GTI info"""
        # TODO:
        # Check that required info is there
        for colname in ['START', 'STOP']:
            if colname not in self.colnames:
                raise InvalidDataError('GTI missing column: {}'.format(colname))

        for key in ['TSTART', 'TSTOP', 'MJDREFI', 'MJDREFF']:
            if key not in self.meta:
                raise InvalidDataError('GTI missing header keyword: {}'.format(key))

        # TODO: Check that header keywords agree with table entries
        # TSTART, TSTOP, MJDREFI, MJDREFF

        # Check that START and STOP times are consecutive
        times = np.ravel(self['START'], self['STOP'])
        # TODO: not sure this is correct ... add test with a multi-gti table from Fermi.
        if not np.all(np.diff(times) >= 0):
            raise InvalidDataError('GTIs are not consecutive or sorted.')

    def check_times(self):
        """Check if various times are consistent.

        The headers and tables of the FITS EVENTS and GTI extension
        contain various observation and event time information.
        """
        ok = True

        # http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html
        telescope_met_refs = OrderedDict(
            FERMI=Time('2001-01-01T00:00:00'),
            HESS=Time('2000-01-01T12:00:00.000'),
            # TODO: Once CTA has specified their MET reference add check here
        )

        telescope = self.dset.event_list.meta['TELESCOP']
        met_ref = time_ref_from_dict(self.dset.event_list.meta)

        if telescope in telescope_met_refs.keys():
            dt = (met_ref - telescope_met_refs[telescope])
            if dt > self.accuracy['time']:
                ok = False
                self.logger.error('MET reference is incorrect.')
        else:
            self.logger.debug('Skipping MET reference check ... not known for this telescope.')

        # TODO: check latest CTA spec to see which info is required / optional
        # EVENTS header keywords:
        # 'DATE_OBS': '2004-10-14'
        # 'TIME_OBS': '00:08:27'
        # 'DATE_END': '2004-10-14'
        # 'TIME_END': '00:34:44'
        # 'TSTART': 150984507.0
        # 'TSTOP': 150986084.0
        # 'MJDREFI': 51544
        # 'MJDREFF': 0.5
        # 'TIMEUNIT': 's'
        # 'TIMESYS': 'TT'
        # 'TIMEREF': 'local'
        # 'TASSIGN': 'Namibia'
        # 'TELAPSE': 0
        # 'ONTIME': 1577.0
        # 'LIVETIME': 1510.95910644531
        # 'DEADC': 0.964236799627542

        return ok

    def check_coordinates(self):
        """Check if various event list coordinates are consistent.

        Parameters
        ----------
        event_list_dataset : `~gammapy.data.EventListDataset`
            Event list dataset
        accuracy : `~astropy.coordinates.Angle`
            Required accuracy.

        Returns
        -------
        status : bool
            All coordinates consistent?
        """
        ok = True
        ok &= self._check_coordinates_header()
        ok &= self._check_coordinates_galactic()
        ok &= self._check_coordinates_altaz()
        ok &= self._check_coordinates_field_of_view()
        return ok

    def _check_coordinates_header(self):
        """Check TODO"""
        # TODO: implement
        return True

    def _check_coordinates_galactic(self):
        """Check if RA / DEC matches GLON / GLAT."""
        event_list = self.dset.event_list

        for colname in ['RA', 'DEC', 'GLON', 'GLAT']:
            if colname not in event_list.colnames:
                # GLON / GLAT columns are optional ...
                # so it's OK if they are not present ... just move on ...
                self.logger.info('Skipping Galactic coordinate check. '
                                 'Missing column: "{}".'.format(colname))
                return True

        radec = event_list.radec
        galactic = event_list.galactic
        separation = radec.separation(galactic).to('arcsec')
        return self._check_separation(separation, 'GLON / GLAT', 'RA / DEC')

    def _check_coordinates_altaz(self):
        """Check if ALT / AZ matches RA / DEC."""
        event_list = self.dset.event_list
        telescope_array = self.dset.telescope_array

        for colname in ['RA', 'DEC', 'AZ', 'ALT']:
            if colname not in event_list.colnames:
                # AZ / ALT columns are optional ...
                # so it's OK if they are not present ... just move on ...
                self.logger.info('Skipping AltAz coordinate check. '
                                 'Missing column: "{}".'.format(colname))
                return True

        radec = event_list.radec
        altaz_expected = event_list.altaz
        altaz_actual = radec.transform_to(altaz_expected)
        separation = altaz_actual.separation(altaz_expected).to('arcsec')
        return self._check_separation(separation, 'ALT / AZ', 'RA / DEC')

    def _check_coordinates_field_of_view(self):
        """Check if DETX / DETY matches ALT / AZ"""
        # TODO: implement
        return True

    def _check_separation(self, separation, tag1, tag2):
        max_separation = separation.max()

        if max_separation > self.accuracy['angle']:
            # TODO: probably we need to print run number and / or other
            # things for this to be useful in a pipeline ...
            fmt = '{0} not consistent with {1}. Max separation: {2}'
            args = [tag1, tag2, max_separation]
            self.logger.warning(fmt.format(*args))
            return False
        else:
            return True


def event_lists_to_counts_image(header, table_of_files, logger=None):
    """Make count image from event lists (like gtbin).

    TODO: what's a good API and location for this?

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS header
    table_of_files : `~astropy.table.Table`
        Table of event list filenames
    logger : `logging.Logger` or None
        Logger to use

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        Count image
    """
    shape = (header['NAXIS2'], header['NAXIS1'])
    data = np.zeros(shape, dtype='int')

    for row in table_of_files:
        if row['filetype'] != 'events':
            continue
        ds = EventListDataset.read(row['filename'])
        if logger:
            logger.info('Processing OBS_ID = {:06d} with {:6d} events.'
                        ''.format(row['OBS_ID'], len(ds.event_list)))
            # TODO: fill events in image.

    return fits.ImageHDU(data=data, header=header)
