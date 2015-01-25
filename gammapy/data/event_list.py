# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Event list: table of LON, LAT, ENERGY, TIME
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from collections import OrderedDict
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Angle, AltAz
from astropy.table import Table
from ..data import GoodTimeIntervals, TelescopeArray
from . import utils

__all__ = ['EventList',
           'EventListDataset',
           'EventListDatasetChecker',
           ]


class EventList(Table):
    """Event list `~astropy.table.Table`.

    The most important reconstructed event parameters
    are available as the following columns:

    - ``TIM'`` - Mission elapsed time (sec)
    - ``RA``, ``DEC`` - FK5 J2000 (or ICRS?) position (deg)
    - ``ENERGY`` - Energy (usually MeV for Fermi and TeV for IACTs)

    Other optional (columns) that are sometimes useful for high-level analysis:

    - ``GLON``, ``GLAT`` - Galactic coordinates (deg)
    - ``DETX``, ``DETY`` - Field of view coordinates (radian?)

    Note that when reading data for analysis you shouldn't use those
    values directly, but access them via properties which create objects
    of the appropriate class and convert to 64 bit:

    - `obstime` for '`TIME`'
    """
    @property
    def info(self):
        """Summary info string."""
        s = '---> Event list info:\n'
        s += '- events: {}\n'.format(len(self))
        return s

    @property
    def radec(self):
        """RA / DEC sky coordinate (`~astropy.coordinates.SkyCoord`)"""
        lon = self['RA'].astype('f64')
        lat = self['DEC'].astype('f64')
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @property
    def galactic(self):
        """Galactic sky coordinate (`~astropy.coordinates.SkyCoord`)"""
        lon = self['GLON'].astype('f64')
        lat = self['GLAT'].astype('f64')
        return SkyCoord(lon, lat, unit='deg', frame='galactic')

    @property
    def altaz(self):
        """Horizontal sky coordinate (`~astropy.coordinates.SkyCoord`)"""
        lon = self['AZ'].astype('f64')
        lat = self['ALT'].astype('f64')
        obstime = self.obstime
        location = self.observatory_earth_location
        altaz_frame = AltAz(obstime=obstime, location=location)
        return SkyCoord(lon, lat, unit='deg', frame=altaz_frame)

    @property
    def obstime(self):
        """Event times as `~astropy.time.Time` objects."""
        met_ref = utils._time_ref_from_dict(self.meta)
        met = TimeDelta(self['TIME'].astype('f64'), format='sec')
        return met_ref + met

    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        return utils._earth_location_from_dict(self.meta)


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

    @staticmethod
    def from_hdu_list(hdu_list):
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

        return EventListDataset(event_list, telescope_array, good_time_intervals)

    @staticmethod
    def read(filename):
        """Read event list from FITS file.
        """
        # return EventList.from_hdu_list(fits.open(filename))
        event_list = EventList.read(filename, hdu='EVENTS')
        telescope_array = TelescopeArray.read(filename, hdu='TELARRAY')
        good_time_intervals = GoodTimeIntervals.read(filename, hdu='GTI')

        return EventListDataset(event_list, telescope_array, good_time_intervals)

    @property
    def info(self):
        """Summary info string."""
        s = '===> Event list dataset information:\n'
        s += self.event_list.info
        s += self.telescope_array.info
        s += self.good_time_intervals.info
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
        Logger to use
    """
    _AVAILABLE_CHECKS = OrderedDict(
        misc='check_misc',
        times='check_times',
        coordinates='check_coordinates',
    )

    accuracy = OrderedDict(
        angle=Angle('1 arcsec'),
        time=TimeDelta(1e-6, format='sec'),

    )

    def __init__(self, event_list_dataset, logger=None):
        self.dset = event_list_dataset
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger('EventListDatasetChecker')

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
            logging.error('Missing meta info: {}'.format(missing_meta))

        # TODO: implement more basic checks that all required info is present.

        return ok

    def check_times(self):
        """Check if various times are consistent.

        The headers and tables of the FITS EVENTS and GTI extension
        contain various observation and event time information.
        """
        ok = True

        # http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html
        telescope_met_refs = OrderedDict(
            FERMI=Time('2001-01-01 00:00:00', scale='utc'),
            HESS=Time('2000-01-01 12:00:00.000', scale='utc'),
            # TODO: Once CTA has specified their MET reference add check here
        )

        telescope = self.dset.event_list.meta['TELESCOP']
        met_ref = utils._time_ref_from_dict(self.dset.event_list.meta)

        if telescope in telescope_met_refs.keys():
            dt = (met_ref - telescope_met_refs[telescope])
            if dt > self.accuracy['time']:
                ok = False
                logging.error('MET reference is incorrect.')
        else:
            logging.debug('Skipping MET reference check ... not known for this telescope.')

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

        # TODO: I think we don't need this here, because `event_list.altaz`
        # can be used as the coordinate frame to transform to...
        # location = telescope_array.get_earth_location()
        # obstime = event_list.obstime
        # altaz_frame = AltAz(location=location, obstime=obstime)

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
