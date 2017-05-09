# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import OrderedDict
import numpy as np
from astropy.utils.console import ProgressBar
from astropy.io import fits
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, AltAz
from astropy.table import Table
from astropy.table import vstack as vstack_tables
from ..utils.energy import EnergyBounds
from ..utils.scripts import make_path
from ..extern.pathlib import Path
from ..utils.time import time_ref_from_dict
from .utils import _earth_location_from_dict
from .gti import GTI
from . import InvalidDataError

__all__ = [
    'EventList',
    'EventListDataset',
    'EventListDatasetChecker',
]

log = logging.getLogger(__name__)


class EventList(object):
    """Event list.

    Data format specification: ref:`gadf:iact-events`

    Event list data is stored in ``table`` (`~astropy.table.Table`) data member.

    TODO: merge this class with EventListDataset, which also holds a GTI extension.

    The most important reconstructed event parameters
    are available as the following columns:

    - ``TIME`` - Mission elapsed time (sec)
    - ``RA``, ``DEC`` - ICRS system position (deg)
    - ``ENERGY`` - Energy (usually MeV for Fermi and TeV for IACTs)

    Other optional (columns) that are sometimes useful for high-level analysis:

    - ``GLON``, ``GLAT`` - Galactic coordinates (deg)
    - ``DETX``, ``DETY`` - Field of view coordinates (deg)

    Note that when reading data for analysis you shouldn't use those
    values directly, but access them via properties which create objects
    of the appropriate class:

    - `time` for ``TIME``
    - `radec` for ``RA``, ``DEC``
    - `energy` for ``ENERGY``
    - `galactic` for ``GLON``, ``GLAT``

    Parameters
    ----------
    table : `~astropy.table.Table`
        Event list table

    Examples
    --------
    To load an example H.E.S.S. event list:

    >>> from gammapy.data import EventList
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
    >>> events = EventList.read(filename)

    To load an example Fermi-LAT event list (the one corresponding to the 2FHL catalog dataset):

    >>> filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz'
    >>> events = EventList.read(filename)
    """

    def __init__(self, table):

        # TODO: remove this temp fix once we change to a new test dataset
        # This is a temp fix because this test dataset is used for many Gammapy tests
        # but it doesn't have the units set properly
        # '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2'
        if 'ENERGY' in table.colnames:
            if not table['ENERGY'].unit:
                table['ENERGY'].unit = 'TeV'

        self.table = table

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from FITS file.

        Format specification: :ref:`gadf:iact-events`

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        if 'hdu' not in kwargs:
            kwargs.update(hdu='EVENTS')

        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    @classmethod
    def stack(cls, event_lists, **kwargs):
        """Stack (concatenate) list of event lists.

        Calls `~astropy.table.vstack`.

        Parameters
        ----------
        event_lists : list
            list of `~gammapy.data.EventList` to stack
        """
        tables = [_.table for _ in event_lists]
        stacked_table = vstack_tables(tables, **kwargs)
        return cls(stacked_table)

    def __str__(self):
        ss = 'EventList info:\n'
        ss += '- Number of events: {}\n'.format(len(self.table))
        # TODO: add time, RA, DEC and if present GLON, GLAT info ...

        ss += '- Median energy: {}\n'.format(np.median(self.energy))

        if 'AZ' in self.table.colnames:
            # TODO: azimuth should be circular median
            ss += '- Median azimuth: {}\n'.format(np.median(self.table['AZ']))

        if 'ALT' in self.table.colnames:
            ss += '- Median altitude: {}\n'.format(np.median(self.table['ALT']))

        return ss

    @property
    def time(self):
        """Event times (`~astropy.time.Time`).

        Notes
        -----
        Times are automatically converted to 64-bit floats.
        With 32-bit floats times will be incorrect by a few seconds
        when e.g. adding them to the reference time.
        """
        met_ref = time_ref_from_dict(self.table.meta)
        met = Quantity(self.table['TIME'].astype('float64'), 'second')
        time = met_ref + met
        return time

    @property
    def radec(self):
        """Event RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`).

        TODO: the `radec` and `galactic` properties should be cached as table columns
        """
        lon, lat = self.table['RA'], self.table['DEC']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @property
    def galactic(self):
        """Event Galactic sky coordinates (`~astropy.coordinates.SkyCoord`).

        Note: uses the ``GLON`` and ``GLAT`` columns.
        If only ``RA`` and ``DEC`` are present use the explicit
        ``event_list.radec.to('galactic')`` instead.
        """
        self.add_galactic_columns()
        lon, lat = self.table['GLON'], self.table['GLAT']
        return SkyCoord(lon, lat, unit='deg', frame='galactic')

    def add_galactic_columns(self):
        """Add Galactic coordinate columns to the table.

        Adds the following columns to the table if not already present:
        - "GLON" - Galactic longitude (deg)
        - "GLAT" - Galactic latitude (deg)
        """
        if set(['GLON', 'GLAT']).issubset(self.table.colnames):
            return

        galactic = self.radec.galactic
        self.table['GLON'] = galactic.l.degree
        self.table['GLAT'] = galactic.b.degree

    # TODO: the following properties are also present on the `DataStoreObservation` class.
    # This duplication should be removed.
    # Maybe the EventList or EventListDataset should have an `observation` object member?
    @property
    def observatory_earth_location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return _earth_location_from_dict(self.table.meta)

    @property
    def observation_time_duration(self):
        """Observation time duration in seconds (`~astropy.units.Quantity`).

        The wall time, including dead-time.
        """
        return Quantity(self.table.meta['ONTIME'], 'second')

    @property
    def observation_live_time_duration(self):
        """Live-time duration in seconds (`~astropy.units.Quantity`).

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return Quantity(self.table.meta['LIVETIME'], 'second')

    @property
    def observation_dead_time_fraction(self):
        """Dead-time fraction (float).

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        http://en.wikipedia.org/wiki/Dead_time
        http://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        return 1 - self.table.meta['DEADC']

    @property
    def altaz(self):
        """Event horizontal sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        time = self.time
        location = self.observatory_earth_location
        altaz_frame = AltAz(obstime=time, location=location)

        lon, lat = self.table['AZ'], self.table['ALT']
        return SkyCoord(lon, lat, unit='deg', frame=altaz_frame)

    @property
    def pointing_radec(self):
        """Pointing RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        info = self.table.meta
        lon, lat = info['RA_PNT'], info['DEC_PNT']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @property
    def offset(self):
        """Event offset from the array pointing position (`~astropy.coordinates.Angle`)."""
        position = self.radec
        center = self.pointing_radec
        offset = center.separation(position)
        return Angle(offset, unit='deg')

    @property
    def energy(self):
        """Event energies (`~astropy.units.Quantity`)."""
        return self.table['ENERGY'].quantity

    def select_row_subset(self, row_specifier):
        """Select table row subset.

        Parameters
        ----------
        row_specifier : slice, int, or array of ints
            Specification for rows to select,
            passed on to ``self.table[row_specifier]``.

        Returns
        -------
        event_list : `EventList`
            New event list with table row subset selected

        Examples
        --------
        Use a boolean mask as ``row_specifier``:

            mask = events.table['FOO'] > 42
            events2 = events.select_row_subset(mask)

        Use row index array as ``row_specifier``:

            idx = np.where(events.table['FOO'] > 42)[0]
            events2 = events.select_row_subset(idx)
        """
        table = self.table[row_specifier]
        return self.__class__(table=table)

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
        return self.select_row_subset(mask)

    def select_offset(self, offset_band):
        """Select events in offset band.

        Parameters
        ----------
        offset_band : `~astropy.coordinates.Angle`
            offset band ``[offset_min, offset_max)``

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        offset = self.offset
        mask = (offset_band[0] <= offset)
        mask &= (offset < offset_band[1])
        return self.select_row_subset(mask)

    def select_time(self, time_interval):
        """Select events in time interval.
        """
        time = self.time
        mask = (time_interval[0] <= time)
        mask &= (time < time_interval[1])
        return self.select_row_subset(mask)

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
        return self.select_row_subset(mask)

    def select_sky_ring(self, center, inner_radius, outer_radius):
        """Select events in ring region on the sky.

        Parameters
        ----------
        center : `~astropy.coordinates.SkyCoord`
            Sky ring center
        inner_radius, outer_radius : `~astropy.coordinates.Angle`
            Sky ring inner and outer radius

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
        return self.select_row_subset(mask)

    def select_sky_box(self, lon_lim, lat_lim, frame='icrs'):
        """Select events in sky box.

        TODO: move `gammapy.catalog.select_sky_box` to gammapy.utils.
        """
        from ..catalog import select_sky_box
        selected = select_sky_box(self.table, lon_lim, lat_lim, frame)
        return self.__class__(selected)

    def select_circular_region(self, region):
        """Select events in circular regions.

        TODO: Extend to support generic regions

        Parameters
        ----------
        region : `~regions.CircleSkyRegion` or list of `~regions.CircleSkyRegion`
            (List of) sky region(s)

        Returns
        -------
        event_list : `EventList`
            Copy of event list with selection applied.
        """
        if not isinstance(region, list):
            region = list([region])
        mask = self.filter_circular_region(region)
        return self.select_row_subset(mask)

    def filter_circular_region(self, region):
        """Create selection mask for event in given circular regions.

        TODO: Extend to support generic regions

        Parameters
        ----------
        region : list of `~region.SkyRegion`
            List of sky regions

        Returns
        -------
        index_array : `np.array`
            Index array of selected events
        """
        position = self.radec
        mask = np.array([], dtype=int)
        for reg in region:
            separation = reg.center.separation(position)
            temp = np.where(separation < reg.radius)[0]
            mask = np.union1d(mask, temp)
        return mask

    def peek(self):
        """Summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        self.plot_image_radec(ax=axes[0])
        #        self.plot_time_map(ax=axes[1])
        self.plot_time(ax=axes[1])

        # log-log scale for time map
        #        xlims = axes[1].set_xlim()
        #        ylims = axes[1].set_ylim()
        #        axes[1].set_xlim(1e-3, xlims[1])
        #        axes[1].set_ylim(1e-3, ylims[1])
        #        axes[1].loglog()
        # TODO: self.plot_energy_dependence(ax=axes[x])
        # TODO: self.plot_offset_dependence(ax=axes[x])
        plt.tight_layout()

    def plot_image_radec(self, ax=None, number_bins=50):
        """Plot a sky  counts image in RADEC coordinate.

        TODO: fix the histogramming ... this example shows that it's currently incorrect:
        gammapy-data-show ~/work/hess-host-analyses/hap-hd-example-files/run023000-023199/run023037/hess_events_023037.fits.gz events -p
        Maybe we can use the FOVCube class for this with one energy bin.
        Or add a separate FOVImage class.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import PowerNorm

        ax = plt.gca() if ax is None else ax

        # max_x = max(self.table['RA'])
        # min_x = min(self.table['RA'])
        # max_y = max(self.table['DEC'])
        # min_y = min(self.table['DEC'])
        #
        # x_edges = np.linspace(min_x, max_x, number_bins)
        # y_edges = np.linspace(min_y, max_y, number_bins)

        count_image, x_edges, y_edges = np.histogram2d(
            self.table[:]['RA'], self.table[:]['DEC'], bins=number_bins)

        ax.set_title('# Photons')

        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')

        ax.plot(self.pointing_radec.ra.value, self.pointing_radec.dec.value, '+', ms=20, mew=3, color='white')

        im = ax.imshow(count_image, interpolation='nearest', origin='low',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       norm=PowerNorm(gamma=0.5))

        ax.invert_xaxis()
        ax.grid()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    def plot_image(self, ax=None, number_bins=50):
        """Plot the counts as a function of x and y camera coordinate.

        TODO: fix the histogramming ... this example shows that it's currently incorrect:
        gammapy-data-show ~/work/hess-host-analyses/hap-hd-example-files/run023000-023199/run023037/hess_events_023037.fits.gz events -p
        Maybe we can use the FOVCube class for this with one energy bin.
        Or add a separate FOVImage class.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        max_x = max(self.table['DETX'])
        min_x = min(self.table['DETX'])
        max_y = max(self.table['DETY'])
        min_y = min(self.table['DETY'])

        x_edges = np.linspace(min_x, max_x, number_bins)
        y_edges = np.linspace(min_y, max_y, number_bins)

        count_image, x_edges, y_edges = np.histogram2d(
            self.table[:]['DETY'], self.table[:]['DETX'],
            bins=(x_edges, y_edges)
        )

        ax.set_title('# Photons')

        ax.set_xlabel('x / deg')
        ax.set_ylabel('y / deg')
        ax.imshow(count_image, interpolation='nearest', origin='low',
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

    def plot_energy_hist(self, ax=None, ebounds=None, **kwargs):
        """Plot counts as a function of energy."""
        from ..spectrum import CountsSpectrum

        if ebounds is None:
            emin = np.min(self['ENERGY'].quantity)
            emax = np.max(self['ENERGY'].quantity)
            ebounds = EnergyBounds.equal_log_spacing(emin, emax, 100)

        spec = CountsSpectrum(energy=ebounds)
        spec.fill(self)
        spec.plot(ax=ax, **kwargs)
        return ax

    def plot_offset_hist(self, ax=None):
        """Plot counts as a function of camera offset."""
        raise NotImplementedError

    def plot_energy_offset(self, ax=None):
        """Plot energy dependence as a function of camera offset."""
        raise NotImplementedError

    def plot_time(self, ax=None):
        """Plots an event rate time curve.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None
            Axes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes

        Examples
        --------
        Plot the rate of the events:

        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            from gammapy.data import DataStore

            ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
            events = ds.obs(obs_id=23523).events
            events.plot_time_map()
            plt.show()

        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        time = self.table['TIME']
        first_event_time = np.min(time)

        # Note the events are not necessarily in time order
        relative_event_times = time - first_event_time

        ax.set_title('Event rate ')

        ax.set_xlabel('seconds')
        ax.set_ylabel('Events / s')
        rate, t = np.histogram(relative_event_times, bins=50)
        t_center = (t[1:] + t[:-1]) / 2

        ax.plot(t_center, rate)

        return ax

    def plot_time_map(self, ax=None):
        """A time map showing for each event the time between the previous and following event.

        The use and implementation are described here:
        https://districtdatalabs.silvrback.com/time-maps-visualizing-discrete-events-across-many-timescales

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` or None
            Axes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes

        Examples
        --------
        Plot a time map of the events:

        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            from gammapy.data import DataStore

            ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
            events = ds.obs(obs_id=23523).events
            events.plot_time_map()
            plt.show()
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        time = self.table['TIME']
        first_event_time = np.min(time)

        # Note the events are not necessarily in time order
        relative_event_times = time - first_event_time

        diffs = relative_event_times[1:] - relative_event_times[:-1]

        xcoords = diffs[:-1]  # all differences except the last
        ycoords = diffs[1:]  # all differences except the first

        ax.set_title('Time Map')

        ax.set_xlabel('time before event / s')
        ax.set_ylabel('time after event / s')
        ax.scatter(xcoords, ycoords)

        return ax


class EventListDataset(object):
    """Event list dataset (event list plus some extra info).

    TODO: I'm not sure if IRFs should be included in this
    class or if an extra container class should be added.

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list table
    gti : `~gammapy.data.GTI`
        Good time interval table
    """

    def __init__(self, event_list, gti=None):
        self.event_list = event_list
        self.gti = gti

    @classmethod
    def from_hdu_list(cls, hdu_list):
        """Create `EventList` from a `~astropy.io.fits.HDUList`."""
        # TODO: This doesn't work because FITS / Table is not integrated.
        # Maybe the easiest solution for now it to write the hdu_list
        # to an in-memory buffer with StringIO and then read it
        # back using Table.read()?
        raise NotImplementedError
        event_list = EventList.from_hdu(hdu_list['EVENTS'])
        gti = GTI.from_hdu(hdu_list['GTI'])

        return cls(event_list=event_list, gti=gti)

    @classmethod
    def read(cls, filename):
        """Read event list from FITS file."""
        event_list = EventList.read(filename)

        try:
            gti = GTI.read(filename, hdu='GTI')
        except KeyError:
            gti = None

        return cls(event_list=event_list, gti=gti)

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
        for filename in ProgressBar(filenames):
            # logger.info('Reading {}'.format(filename))
            event_list = Table.read(filename, hdu='EVENTS')

            # TODO: Remove and modify header keywords for stacked event list
            meta_del = ['OBS_ID', 'OBJECT']
            meta_mod = ['DATE_OBS', 'DATE_END', 'TIME_OBS', 'TIME_END']

            gti = Table.read(filename, hdu='GTI')
            event_lists.append(event_list)
            gtis.append(gti)

        total_event_list = vstack_tables(event_lists, metadata_conflicts='silent')
        total_gti = vstack_tables(gtis, metadata_conflicts='silent')

        total_event_list.meta['EVTSTACK'] = 'yes'
        total_gti.meta['EVTSTACK'] = 'yes'

        return cls(event_list=total_event_list, gti=total_gti)

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

        data = self.gti.as_array()
        header = fits.Header()
        header.update(self.gti.meta)
        hdu_list.append(fits.BinTableHDU(data, header=header, name='GTI'))

        return hdu_list

    def __str__(self):
        ss = 'Event list dataset info:\n'
        ss += str(self.event_list)
        ss += str(self.gti)
        return ss

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

    Data format specification: ref:`gadf:iact-events`

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
        missing_meta = set(required_meta) - set(self.dset.event_list.table.meta)
        if missing_meta:
            ok = False
            self.logger.error('Missing meta info: {}'.format(missing_meta))

        # TODO: implement more basic checks that all required info is present.

        return ok

    def _check_times_gtis(self):
        """Check GTI info."""
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
        # https://hess-confluence.desy.de/confluence/display/HESS/HESS+FITS+data+-+References+and+checks#HESSFITSdata-Referencesandchecks-Time
        telescope_met_refs = OrderedDict(
            FERMI=Time('2001-01-01T00:00:00'),
            HESS=Time('2001-01-01T00:00:00'),
        )

        meta = self.dset.event_list.table.meta
        telescope = meta['TELESCOP']
        met_ref = time_ref_from_dict(meta)

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
            if colname not in event_list.table.colnames:
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

        for colname in ['RA', 'DEC', 'AZ', 'ALT']:
            if colname not in event_list.table.colnames:
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
        """Check if DETX / DETY matches ALT / AZ."""
        # TODO: implement
        return True

    def _check_separation(self, separation, tag1, tag2):
        max_separation = separation.max()

        if max_separation > self.accuracy['angle']:
            # TODO: probably we need to print run number and / or other
            # things for this to be useful in a pipeline ...
            fmt = '{} not consistent with {}. Max separation: {}'
            args = [tag1, tag2, max_separation]
            self.logger.warning(fmt.format(*args))
            return False
        else:
            return True
