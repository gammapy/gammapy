# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from operator import le, lt
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.units import Quantity
from gammapy.utils.scripts import make_path
from gammapy.utils.time import (
    time_ref_from_dict,
    time_ref_to_dict,
    time_relative_to_ref,
)

__all__ = ["GTI"]


class GTI:
    """Good time intervals (GTI) `~astropy.table.Table`.

    Data format specification: :ref:`gadf:iact-gti`

    Note: at the moment dead-time and live-time is in the
    EVENTS header ... the GTI header just deals with
    observation times.

    Parameters
    ----------
    table : `~astropy.table.Table`
        GTI table

    Examples
    --------
    Load GTIs for a H.E.S.S. event list:

    >>> from gammapy.data import GTI
    >>> gti = GTI.read('$GAMMAPY_DATA/hess-dl3-dr1//data/hess_dl3_dr1_obs_id_023523.fits.gz')
    >>> print(gti)
    GTI info:
    - Number of GTIs: 1
    - Duration: 1687.0 s
    - Start: 53343.92234009259 MET
    - Start: 2004-12-04T22:08:10.184 (time standard: TT)
    - Stop: 53343.94186555556 MET
    - Stop: 2004-12-04T22:36:17.184 (time standard: TT)

    Load GTIs for a Fermi-LAT event list:

    >>> gti = GTI.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
    >>> print(gti)
    GTI info:
    - Number of GTIs: 39042
    - Duration: 183139597.9032163 s
    - Start: 54682.65603794185 MET
    - Start: 2008-08-04T15:44:41.678 (time standard: TT)
    - Stop: 57236.96833546296 MET
    - Stop: 2015-08-02T23:14:24.184 (time standard: TT)
    """

    def __init__(self, table):
        self.table = table

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def create(cls, start, stop, reference_time="2000-01-01"):
        """Creates a GTI table from start and stop times.

        Parameters
        ----------
        start : `~astropy.units.Quantity`
            start times w.r.t. reference time
        stop : `~astropy.units.Quantity`
            stop times w.r.t. reference time
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        """
        start = Quantity(start, ndmin=1)
        stop = Quantity(stop, ndmin=1)
        reference_time = Time(reference_time)
        meta = time_ref_to_dict(reference_time)
        table = Table({"START": start.to("s"), "STOP": stop.to("s")}, meta=meta)
        return cls(table)

    @classmethod
    def read(cls, filename, hdu="GTI"):
        """Read from FITS file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        hdu : str
            hdu name. Default GTI.
        """
        filename = make_path(filename)
        table = Table.read(filename, hdu=hdu)
        return cls(table)

    def write(self, filename, **kwargs):
        """Write to file."""
        self.table.write(make_path(filename), **kwargs)

    def __str__(self):
        return (
            "GTI info:\n"
            f"- Number of GTIs: {len(self.table)}\n"
            f"- Duration: {self.time_sum}\n"
            f"- Start: {self.time_start[0]} MET\n"
            f"- Start: {self.time_start[0].fits} (time standard: {self.time_start[-1].scale.upper()})\n"
            f"- Stop: {self.time_stop[-1]} MET\n"
            f"- Stop: {self.time_stop[-1].fits} (time standard: {self.time_stop[-1].scale.upper()})\n"
        )

    @property
    def time_delta(self):
        """GTI durations in seconds (`~astropy.units.Quantity`)."""
        start = self.table["START"].astype("float64")
        stop = self.table["STOP"].astype("float64")
        return Quantity(stop - start, "second")

    @property
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)."""
        return time_ref_from_dict(self.table.meta)

    @property
    def time_sum(self):
        """Sum of GTIs in seconds (`~astropy.units.Quantity`)."""
        return self.time_delta.sum()

    @property
    def time_start(self):
        """GTI start times (`~astropy.time.Time`)."""
        met = Quantity(self.table["START"].astype("float64"), "second")
        return self.time_ref + met

    @property
    def time_stop(self):
        """GTI end times (`~astropy.time.Time`)."""
        met = Quantity(self.table["STOP"].astype("float64"), "second")
        return self.time_ref + met

    @property
    def time_intervals(self):
        """List of time intervals"""
        return [
            (t_start, t_stop)
            for t_start, t_stop in zip(self.time_start, self.time_stop)
        ]

    @classmethod
    def from_time_intervals(cls, time_intervals, reference_time="2000-01-01"):
        """From list of time intervals

        Parameters
        ----------
        time_intervals : list of `~astropy.time.Time` objects
            Time intervals
        reference_time : `~astropy.time.Time`
            Reference time to use in GTI definition

        Returns
        -------
        gti : `GTI`
            GTI table.
        """
        reference_time = Time(reference_time)
        start = Time([_[0] for _ in time_intervals]) - reference_time
        stop = Time([_[1] for _ in time_intervals]) - reference_time
        meta = time_ref_to_dict(reference_time)
        table = Table({"START": start.to("s"), "STOP": stop.to("s")}, meta=meta)
        return cls(table=table)

    def select_time(self, time_interval):
        """Select and crop GTIs in time interval.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time for the selection.

        Returns
        -------
        gti : `GTI`
            Copy of the GTI table with selection applied.
        """
        # get GTIs that fall within the time_interval
        mask = self.time_start < time_interval[1]
        mask &= self.time_stop > time_interval[0]
        gti_within = self.table[mask]

        # crop the GTIs
        start_met = time_relative_to_ref(time_interval[0], self.table.meta)
        stop_met = time_relative_to_ref(time_interval[1], self.table.meta)
        np.clip(
            gti_within["START"],
            start_met.value,
            stop_met.value,
            out=gti_within["START"],
        )
        np.clip(
            gti_within["STOP"], start_met.value, stop_met.value, out=gti_within["STOP"]
        )

        return self.__class__(gti_within)

    def stack(self, other):
        """Stack with another GTI in place.

        This simply changes the time reference of the second GTI table
        and stack the two tables. No logic is applied to the intervals.

        Parameters
        ----------
        other : `~gammapy.data.GTI`
            GTI to stack to self

        """
        start = (other.time_start - self.time_ref).sec
        end = (other.time_stop - self.time_ref).sec
        table = Table({"START": start, "STOP": end}, names=["START", "STOP"])
        self.table = vstack([self.table, table])

    def union(self, overlap_ok=True, merge_equal=True):
        """Union of overlapping time intervals.

        Returns a new `~gammapy.data.GTI` object.

        Parameters
        ----------
        overlap_ok : bool
            Whether to raise an error when overlapping time bins are found.
        merge_equal : bool
            Whether to merge touching time bins e.g. ``(1, 2)`` and ``(2, 3)``
            will result in ``(1, 3)``.
        """
        # Algorithm to merge overlapping intervals is well-known,
        # see e.g. https://stackoverflow.com/a/43600953/498873

        table = self.table.copy()
        table.sort("START")

        compare = lt if merge_equal else le

        # We use Python dict instead of astropy.table.Row objects,
        # because on some versions modifying Row entries doesn't behave as expected
        merged = [{"START": table[0]["START"], "STOP": table[0]["STOP"]}]
        for row in table[1:]:
            interval = {"START": row["START"], "STOP": row["STOP"]}
            if compare(merged[-1]["STOP"], interval["START"]):
                merged.append(interval)
            else:
                if not overlap_ok:
                    raise ValueError("Overlapping time bins")

                merged[-1]["STOP"] = max(interval["STOP"], merged[-1]["STOP"])

        merged = Table(rows=merged, names=["START", "STOP"], meta=self.table.meta)
        return self.__class__(merged)

    def group_table(self, time_intervals, atol="1e-6 s"):
        """Compute the table with the info on the group to which belong each time interval.

        The t_start and t_stop are stored in MJD from a scale in "utc".

        Parameters
        ----------
        time_intervals : list of `astropy.time.Time`
            Start and stop time for each interval to compute the LC
        atol : `~astropy.units.Quantity`
            Tolerance value for time comparison with different scale. Default 1e-6 sec.

        Returns
        -------
        group_table : `~astropy.table.Table`
            Contains the grouping info.
        """
        atol = Quantity(atol)

        group_table = Table(
            names=("group_idx", "time_min", "time_max", "bin_type"),
            dtype=("i8", "f8", "f8", "S10"),
        )
        time_intervals_lowedges = Time(
            [time_interval[0] for time_interval in time_intervals]
        )
        time_intervals_upedges = Time(
            [time_interval[1] for time_interval in time_intervals]
        )

        for t_start, t_stop in zip(self.time_start, self.time_stop):
            mask1 = t_start >= time_intervals_lowedges - atol
            mask2 = t_stop <= time_intervals_upedges + atol
            mask = mask1 & mask2
            if np.any(mask):
                group_index = np.where(mask)[0]
                bin_type = ""
            else:
                group_index = -1
                if np.any(mask1):
                    bin_type = "overflow"
                elif np.any(mask2):
                    bin_type = "underflow"
                else:
                    bin_type = "outflow"
            group_table.add_row(
                [group_index, t_start.utc.mjd, t_stop.utc.mjd, bin_type]
            )

        return group_table
