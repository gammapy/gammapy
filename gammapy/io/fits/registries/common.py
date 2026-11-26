# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Generic FITS keyword blocks, reusable by any format (GADF, OGIP, ...).

All keywords are optional here; each format decides which ones are mandatory
with ``compose_header(..., required=...)``.
"""

from gammapy.io.fits.registries import _kw

__all__ = [
    "FITS_TIME_KEYWORDS",
    "FITS_TIME_CONVENIENCE_KEYWORDS",
    "GENERAL_KEYWORDS",
    "HDU_KEYWORDS",
    "HDU_RESPONSE_KEYWORDS",
]

# FITS reference time (FITS standard, "Representations of time coordinates").
FITS_TIME_KEYWORDS = {
    "MJDREFI": _kw("int"),
    "MJDREFF": _kw("float"),
    "TIMEUNIT": _kw("str"),
    "TIMESYS": _kw("str"),
    "TIMEREF": _kw("str"),
}

# Date and time strings, for convenience and human readers.
FITS_TIME_CONVENIENCE_KEYWORDS = {
    key: _kw("str")
    for key in ("DATE-OBS", "DATE-BEG", "DATE-AVG", "TIME-OBS", "DATE-END", "TIME-END")
}

GENERAL_KEYWORDS = {
    key: _kw("str") for key in ("TELESCOP", "INSTRUME", "ORIGIN", "CREATOR")
}

# HDU classification (HDUCLASS hierarchy).
HDU_KEYWORDS = {
    key: _kw("str") for key in ("HDUCLASS", "HDUVERS", "HDUDOC", "HDUCLAS1")
}

HDU_RESPONSE_KEYWORDS = {
    key: _kw("str") for key in ("HDUCLAS2", "HDUCLAS3", "HDUCLAS4")
}
