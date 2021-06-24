# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.time import Time
import astropy.units as u
from gammapy.maps import MapAxis

class TimeMapAxis:
    """Class representing a time axis.

    Provides methods for transforming to/from axis and pixel coordinates.
    A time axis can represent non-contiguous sequences of time intervals.

    Parameters
    ----------
    edges_min : `~astropy.units.Quantity`
        Array of edge time values. This the time delta w.r.t. to the reference time.
    edges_max : ``~astropy.units.Quantity`
        Array of edge time values. This the time delta w.r.t. to the reference time.
    reference_time : `~astropy.time.Time`
        Reference time to use.
    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  Valid options are 'log', 'lin', and 'sqrt'.
    name : str
        Axis name
    """
    node_type = "edges"
    def __init__(self, edges_min, edges_max, reference_time, name="time", interp="lin",):
        self._name = name

        edges_min = u.Quantity(edges_min, ndmin=1)
        edges_max = u.Quantity(edges_max, ndmin=1)

        if not isinstance(reference_time, Time):
            raise TypeError(f"TimeAxis reference time must be Time object. Got {type(reference_time).__name__}.")

        invalid = [_.unit.name for _ in (edges_min, edges_max) if not _.unit.is_equivalent("d")]
        if len(invalid)>0:
            raise TypeError(f"TimeAxis edges must be time-like quantities. Got {invalid}")

        # Note: flatten is there to deal with scalr Time objects
        if not len(edges_min.flatten()) == len(edges_max.flatten()):
            raise ValueError("Time min and time max must have the same length.")

        if not (np.all(edges_min == np.sort(edges_min))
                and np.all(edges_max == np.sort(edges_max))):
            raise ValueError("TimeAxis: edges values must be sorted")

        self._edges_min = u.Quantity(edges_min)
        self._edges_max = u.Quantity(edges_max)
        self._reference_time = reference_time
        self._interp = interp

        self._pix_offset = -0.5
        self._nbin = len(edges_min.flatten())

    @property
    def reference_time(self):
        """Return reference time used for the axis."""
        return self._reference_time

    @property
    def name(self):
        """Return axis name."""
        return self._name

    @property
    def nbin(self):
        """Return number of bins in the axis."""
        return self._nbin

    @property
    def time_min(self):
        """Return axis lower edges as Time objects."""
        return self._edges_min + self.reference_time

    @property
    def time_max(self):
        """Return axis upper edges as Time objects."""
        return self._edges_max + self.reference_time

    @property
    def time_delta(self):
        """Return axis time bin width (`~astropy.time.TimeDelta`)."""
        return self._edges_max - self._edges_min

    @property
    def time_mid(self):
        """Return time bin center (`~astropy.time.Time`)."""
        return self.time_min + 0.5 * self.time_delta

    def coord_to_idx(self, time):
        """Transform from axis time coordinate to bin index.

        Indices of time values falling outside time bins will be
        set to -1.

        Parameters
        ----------
        coord : `~astropy.time.Time`
            Array of axis coordinate values.

        Returns
        -------
        idx : `~numpy.ndarray`
            Array of bin indices.
        """

        time = Time(time[..., np.newaxis])
        delta_plus = (time - self.time_min).value > 0.
        delta_minus = (time - self.time_max).value <= 0.
        mask = np.logical_and(delta_plus, delta_minus)

        idx = np.asanyarray(np.argmax(mask, axis=-1))
        idx[~np.any(mask, axis=-1)] = -1
        return idx

    def time_to_pix(self, coord):
        return self.coord_to_idx(coord)

    def pix_to_idx(self, pix, clip=False):
        return pix

    @property
    def center(self):
        """Return `~astropy.time.Time` at interval centers."""
        return self.time_mid

    @property
    def bin_width(self):
        """Return time interval width."""
        return self.time_delta

    def __repr__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        fmt = "\t{:<10s} : {:<10s}\n"
        str_ += fmt.format("name", self.name)
        str_ += fmt.format("nbins", str(self.nbin))
        str_ += fmt.format("node type", self.node_type)
        return str_.expandtabs(tabsize=2)

    def upsample(self):
        raise NotImplementedError

    def slice(self, idx):
        """Create a new axis object by extracting a slice from this axis.

        Parameters
        ----------
        idx : slice
            Slice object selecting a subselection of the axis.

        Returns
        -------
        axis : `~TimeMapAxis`
            Sliced axis object.
        """
        return TimeMapAxis(
            self._edges_min[idx].copy(),
            self._edges_max[idx].copy(),
            self.reference_time,
            interp=self._interp,
            name=self.name,
        )

    def squash(self):
        """Create a new axis object by squashing the axis into one bin.

        Returns
        -------
        axis : `~MapAxis`
            Sliced axis object.
        """
        return TimeMapAxis(
            self._edges_min[0],
            self._edges_max[-1],
            self.reference_time,
            interp=self._interp,
            name=self._name,
        )

    # TODO: if we are to allow log or sqrt bins the reference time should always
    # be strictly lower than all times
    # Should we define a mechanism to ensure this is always correct?
    @classmethod
    def from_time_edges(cls, time_min, time_max, unit="d", interp="lin", name="time"):
        """Create TimeMapAxis from the time interval edges defined as `~astropy.time.Time`.

        The reference time is defined as the lower edge of the first interval.

        Parameters
        ----------
        time_min : `~astropy.time.Time`
            Array of lower edge times.
        time_max : ``~astropy.time.Time`
            Array of lower edge times.
        unit : `~astropy.units.Unit` or str
            The unit to convert the edges to. Default is 'd' (day).
        interp : str
            Interpolation method used to transform between axis and pixel
            coordinates.  Valid options are 'log', 'lin', and 'sqrt'.
        name : str
            Axis name
        """
        unit = u.Unit(unit)
        reference_time = time_min[0]
        edges_min = time_min - reference_time
        edges_max = time_max - reference_time

        return cls(edges_min.to(unit), edges_max.to(unit), reference_time, interp=interp, name=name)

    #TODO: how configurable should that be? column names?
    @classmethod
    def from_table(cls, table, reference_time=None, name="time"):
        if "TIMESYS" not in table.meta:
            print("No TIMESYS information. Assuming UTC scale.")
            scale = "utc"
        else:
            scale = table.meta["TIMESYS"]
        format = "mjd"

        # TODO: improve and correct
        tmin = Time(table["time_min"], scale=scale, format=format)
        tmax = Time(table["time_max"], scale=scale, format=format)
        if not reference_time:
            reference_time = tmin[0]
        return cls((tmin-reference_time).to('d'), (tmax-reference_time).to('d'), reference_time, name)

    @classmethod
    def from_gti(cls, gti, name="time"):
        """Create a time axis from an input GTI."""
        tmin = gti.time_start - gti.time_ref
        tmax = gti.time_stop - gti.time_ref

        return cls(tmin.to('s'), tmax.to('s'), gti.time_ref, name)