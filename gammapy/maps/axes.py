# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.time import Time


class TimeAxis:
    """Class representing a time axis.

    Provides methods for transforming to/from axis and pixel coordinates.
    A time axis can represent non-contiguous sequences of time intervals.

    Parameters
    ----------
    time_min : `~astropy.time.Time`
        Array of edge time values.
    time_max : `~astropy.time.Time`
        Array of edge time values.
    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  Valid options are 'log', 'lin', and 'sqrt'.
    name : str
        Axis name
    """
    node_type = "edges"
    def __init__(self, time_min, time_max, name="", interp="lin",):
        self._name = name

        invalid = [type(_).__name__ for _ in (time_min, time_max) if not isinstance(_, Time)]
        if len(invalid)>0:
            raise TypeError(f"TimeAxis edges must be Time objects. Got {invalid}")

        if not len(time_min) == len(time_max):
            raise ValueError("Time min and time max must have the same length.")

        if not (np.all(time_min == time_min.sort()) and np.all(time_max == time_max.sort()) ):
            raise ValueError("TimeAxis: edge values must be sorted")

        self._time_min = Time(time_min)
        self._time_max = Time(time_max)

        self._interp = interp

        self._pix_offset = -0.5
        self._nbin = len(time_min)

    @property
    def name(self):
        return self._name

    @property
    def nbin(self):
        return len(self._time_min)

    @property
    def time_min(self):
        return self._time_min

    @property
    def time_max(self):
        return self._time_max

    @property
    def time_delta(self):
        """Time bin width (`~astropy.time.TimeDelta`)."""
        return self.time_max - self.time_min

    @property
    def time_mid(self):
        """Time bin center (`~astropy.time.Time`)."""
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
        return self.time_mid

    @property
    def bin_width(self):
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

    #TODO: how configurable should that be? column names?
    @classmethod
    def from_table(cls, table, name="time"):
        if "TIMESYS" not in table.meta:
            print("No TIMESYS information. Assuming UTC scale.")
            scale = "utc"
        else:
            scale = table.meta["TIMESYS"]
        format = "mjd"

        tmin = Time(table["time_min"], scale=scale, format=format)
        tmax = Time(table["time_max"], scale=scale, format=format)
        return cls(tmin, tmax, name)

    @classmethod
    def from_gti(cls, gti, name="time"):
        """Create a time axis from an input GTI."""
        tmin = gti.time_start
        tmax = gti.time_stop

        return cls(tmin, tmax, name)