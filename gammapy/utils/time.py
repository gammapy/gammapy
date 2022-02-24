# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time related utility functions."""
import numpy as np
from astropy.time import Time
import astropy.units as u


__all__ = [
    "absolute_time",
    "reference_time_from_header",
    "reference_time_to_header",
    "time_relative_to_ref",
]


TIME_REF_FERMI = Time("2001-01-01T00:00:00")


def reference_time_from_header(meta, default_scale="tt"):
    """Calculate the time reference from fits header keywords.

    See FITS Standard Section 9.2.2.

    Parameters
    ----------
    meta : dict
        FITS time standard header info
    default_scale: str
        Scale to assume when TIMESYS is not present

    Returns
    -------
    time : `~astropy.time.Time`
        Time object with ``format='MJD'``
    """
    scale = meta.get("TIMESYS", default_scale).lower()

    # there are 5 different possible formats for the time reference
    # * MJDREF, * MJDREFI + MJDREFF, * JDREF, * JDREFI + JDREFF, * DATEREF
    # we try them in this order and return the first we find.
    # if REF, REFI and REFF are present, REFI + REFF should be used according
    # to the standard, so we check that first

    for fmt in ('MJD', 'JD'):
        ref_i = f"{fmt}REFI"
        ref_f = f"{fmt}REFF"
        ref = f"{fmt}REF"

        if ref_i in meta and ref_f in meta:
            return Time(float(meta[ref_i]), float(meta[ref_f]), format=fmt.lower(), scale=scale)

        if ref in meta:
            return Time(meta[ref], format=fmt.lower(), scale=scale)

    if 'DATEREF' in meta:
        return Time(meta['DATEREF'], scale=scale)

    # FIXME: from the standard:
    # > If none of the three keywords is present, [...] MJDREF = 0.0 must be assumed
    # return Time(0.0, format='mjd', scale=scale)
    # but currently, dataset loading somehow relies on this KeyError being raised
    raise KeyError("MJDREFF")


def reference_time_to_header(time, scale="tt", format="mjd"):
    """
    Convert an astropy time object to the FITS Header specs

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time object with ``format='MJD'``

    Returns
    -------
    header: dict
        FITS time standard header info
    """
    format = format.lower()

    if not isinstance(time, Time):
        raise TypeError("time must be an astropy.time.Time instance")

    # convert into target scale
    try:
        time = getattr(time, scale.lower())
    except AttributeError:
        raise ValueError(f"Unsupported scale: {scale}")

    d = {"TIMESYS": scale.upper()}

    if format == "mjd":
        mjd_f, mjd_i = np.modf(time.mjd)
        d["MJDREFI"] = int(mjd_i)
        d["MJDREFF"] = mjd_f
        return d

    if format == "jd":
        d["JDREFI"] = int(time.jd1)
        d["JDREFF"] = time.jd2
        return d

    if format == "date":
        d["DATEREF"] = time.iso
        return d

    raise ValueError("format must be one of 'jd', 'mjd' or 'date'")


def time_relative_to_ref(time, reference_time):
    """Convert a time using an existing reference.

    The time reference is built as MJDREFI + MJDREFF in units of MJD.
    The time will be converted to seconds after the reference.

    Parameters
    ----------
    time : `~astropy.time.Time`
        time to be converted
    meta : dict
        dictionary with the keywords ``MJDREFI`` and ``MJDREFF``

    Returns
    -------
    relative_time: astropy.units.Quantity
        time in seconds after the reference
    """
    return (time - reference_time).to(u.s)


def absolute_time(time_delta, reference_time):
    """Convert a MET into human readable date and time.

    Parameters
    ----------
    time_delta : `~astropy.time.TimeDelta` or `~astropy.units.Quanity`
        time in seconds after the MET reference
    reference_time : `astropy.time.Time`
        The MET reference time

    Returns
    -------
    time : `~astropy.time.Time`
        time in same scale as reference time
    """
    return reference_time + time_delta
