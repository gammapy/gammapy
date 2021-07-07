# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import inspect
import numpy as np
from astropy.time import Time
import astropy.units as u
from gammapy.utils.interpolation import interpolation_scale
from .utils import INVALID_INDEX

__all__ = ["TimeMapAxis"]


class TimeMapAxis:
    """Class representing a time axis.

    Provides methods for transforming to/from axis and pixel coordinates.
    A time axis can represent non-contiguous sequences of non-overlapping time intervals.

    Time intervals must be provided in increasing order.

    Parameters
    ----------
    edges_min : `~astropy.units.Quantity`
        Array of edge time values. This the time delta w.r.t. to the reference time.
    edges_max : ``~astropy.units.Quantity`
        Array of edge time values. This the time delta w.r.t. to the reference time.
    reference_time : `~astropy.time.Time`
        Reference time to use.
    name : str
        Axis name
    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  For now only 'lin' is supported.
    """
    node_type = "intervals"
    def __init__(self, edges_min, edges_max, reference_time, name="time", interp="lin"):
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
        self._pix_offset = -0.5
        self._nbin = len(edges_min.flatten())

        if self.nbin>1:
            delta = self._edges_min[1:] - self._edges_max[:-1]
            if np.any(delta.to_value("s")<0):
                raise ValueError("TimeMapAxis: time intervals must not overlap.")
#        else:
#            self._edges_min.reshape((1,))
#            self._edges_max.reshape((1,))

        if interp != "lin":
            raise ValueError("TimeMapAxis: non-linear scaling scheme are not supported yet.")
        self._interp = interp


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

    def assert_name(self, required_name):
        """Assert axis name if a specific one is required.

        Parameters
        ----------
        required_name : str
            Required
        """
        if self.name != required_name:
            raise ValueError(
                "Unexpected axis name,"
                f' expected "{required_name}", got: "{self.name}"'
            )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self._edges_min.shape != other._edges_min.shape:
            return False
        # This will test equality at microsec level.
        delta_min = self.time_min - other.time_min
        delta_max = self.time_max - other.time_max

        return (
            np.allclose(delta_min.to_value("s"), 0., atol=1e-6)
            and np.allclose(delta_max.to_value("s"), 0., atol=1e-6)
            and self._interp == other._interp
            and self.name.upper() == other.name.upper()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)


    def is_aligned(self, other, atol=2e-2):
        raise NotImplementedError

    @property
    def iter_by_edges(self):
        """Iterate by intervals defined by the edges"""
        for time_min, time_max in zip(self.time_min, self.time_max):
            yield (time_min, time_max)

    def coord_to_idx(self, coord):
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

        time = Time(coord[..., np.newaxis])
        delta_plus = (time - self.time_min).value > 0.
        delta_minus = (time - self.time_max).value <= 0.
        mask = np.logical_and(delta_plus, delta_minus)

        idx = np.asanyarray(np.argmax(mask, axis=-1))
        idx[~np.any(mask, axis=-1)] = INVALID_INDEX.int
        return idx

    def coord_to_pix(self, coord):
        """Transform from time to coordinate to pixel position.

        Pixels of time values falling outside time bins will be
        set to -1.

        Parameters
        ----------
        coord : `~astropy.time.Time`
            Array of axis coordinate values.

        Returns
        -------
        pix : `~numpy.ndarray`
            Array of pixel positions.
        """
        idx = np.atleast_1d(self.coord_to_idx(coord))

        valid_pix = np.where(idx!=INVALID_INDEX.int)
        pix = np.asarray(idx, dtype = 'float')
        if pix.shape == ():
            pix = pix.reshape((1,))

        if coord.shape == ():
            coord = coord.reshape((1,))
        relative_time = coord[valid_pix]-self.reference_time

        print(valid_pix)
        scale = interpolation_scale(self._interp)
        valid_idx = idx[valid_pix]
        s_min = scale(self._edges_min[valid_idx])
        s_max = scale(self._edges_max[valid_idx])
        s_coord = scale(relative_time.to(self._edges_min.unit))

        pix[valid_pix] += (s_coord - s_min) / (s_max - s_min)
        return pix

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
        str_ += fmt.format("reference time", self.reference_time.iso)
        str_ += fmt.format("total time", self.time_delta.sum())
        return str_.expandtabs(tabsize=2)

    def upsample(self):
        raise NotImplementedError

    def downsample(self):
        raise NotImplementedError

    def _init_copy(self, **kwargs):
        """Init map axis instance by copying missing init arguments from self.
        """
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy `MapAxis` instance and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map axis constructor.

        Returns
        -------
        copy : `MapAxis`
            Copied map axis.
        """
        return self._init_copy(**kwargs)

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


