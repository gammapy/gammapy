# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.units import Quantity
from astropy.table import Table, vstack
from ..utils.time import time_ref_from_dict, time_relative_to_ref
from ..utils.scripts import make_path

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
    - Start: 2004-12-04T22:08:10.184(TT)
    - Stop: 53343.94186555556 MET
    - Stop: 2004-12-04T22:36:17.184(TT)

    Load GTIs for a Fermi-LAT event list:

    >>> gti = GTI.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
    >>> print(gti)
    GTI info:
    - Number of GTIs: 39042
    - Duration: 183139597.9032163 s
    - Start: 54682.65603794185 MET
    - Start: 2008-08-04T15:44:41.678(TT)
    - Stop: 57236.96833546296 MET
    - Stop: 2015-08-02T23:14:24.184(TT)
    """

    def __init__(self, table):
        self.table = table

    def copy(self):
        return self.__class__(self.table)

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from FITS file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        kwargs.setdefault("hdu", "GTI")
        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    def __str__(self):
        ss = "GTI info:\n"
        ss += "- Number of GTIs: {}\n".format(len(self.table))
        ss += "- Duration: {}\n".format(self.time_sum)
        ss += "- Start: {} MET\n".format(self.time_start[0])
        ss += "- Start: {}\n".format(self.time_start[0].fits)
        ss += "- Stop: {} MET\n".format(self.time_stop[-1])
        ss += "- Stop: {}\n".format(self.time_stop[-1].fits)
        return ss

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
        """Stack with another GTI.

        This simply changes the time reference of the second GTI table
        and stack the two tables. No logic is applied to the intervals.

        Parameters
        ----------
        other : `~gammapy.data.GTI`
            GTI to stack to self

        Returns
        -------
        new_gti : `~gammapy.data.GTI`
            New GTI
        """
        start = (other.time_start - self.time_ref).sec
        end = (other.time_stop - self.time_ref).sec
        table = Table({"START": start, "STOP": end}, names=["START", "STOP"])
        return self.__class__(vstack([self.table, table]))

    def union(self):
        """Union of overlapping time intervals.

        Returns a new `~gammapy.data.GTI` object.

        Intervals that touch will be merged, e.g.
        ``(1, 2)`` and ``(2, 3)`` will result in ``(1, 3)``.
        """
        # Algorithm to merge overlapping intervals is well-known,
        # see e.g. https://stackoverflow.com/a/43600953/498873

        table = self.table.copy()
        table.sort("START")

        # We use Python dict instead of astropy.table.Row objects,
        # because on some versions modifying Row entries doesn't behave as expected
        merged = [{"START": table[0]["START"], "STOP": table[0]["STOP"]}]
        for row in table[1:]:
            interval = {"START": row["START"], "STOP": row["STOP"]}
            if merged[-1]["STOP"] <= interval["START"]:
                merged.append(interval)
            else:
                merged[-1]["STOP"] = max(interval["STOP"], merged[-1]["STOP"])

        merged = Table(rows=merged, names=["START", "STOP"], meta=self.table.meta)
        return self.__class__(merged)
