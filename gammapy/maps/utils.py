# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.io import fits
import json


def coordsys_to_frame(coordsys):
    if coordsys in ["CEL", "C"]:
        return "icrs"
    elif coordsys in ["GAL", "G"]:
        return "galactic"
    else:
        raise ValueError(f"Unknown coordinate system: '{coordsys}'")


def frame_to_coordsys(frame):
    if frame in ["icrs", "fk5", "fk4"]:
        return "CEL"
    elif frame in ["galactic"]:
        return "GAL"
    else:
        raise ValueError(f"Unknown coordinate frame '{frame}'")


class InvalidValue:
    """Class to define placeholder for invalid array values."""

    float = np.nan
    int = np.nan
    bool = np.nan

    def __getitem__(self, dtype):
        if np.issubdtype(dtype, np.integer):
            return self.int
        elif np.issubdtype(dtype, np.floating):
            return self.float
        elif np.issubdtype(dtype, np.dtype(bool).type):
            return self.bool
        else:
            raise ValueError(f"No invalid value placeholder defined for {dtype}")


class InvalidIndex:
    """Class to define placeholder for invalid array indices."""

    float = np.nan
    int = -1
    bool = False


INVALID_VALUE = InvalidValue()
INVALID_INDEX = InvalidIndex()


def find_bands_hdu(hdu_list, hdu):
    """Discover the extension name of the BANDS HDU.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`

    hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`

    Returns
    -------
    hduname : str
        Extension name of the BANDS HDU.  None if no BANDS HDU was found.
    """
    if "BANDSHDU" in hdu.header:
        return hdu.header["BANDSHDU"]

    has_cube_data = False

    if (
        isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU))
        and hdu.header.get("NAXIS", None) == 3
    ):
        has_cube_data = True
    elif isinstance(hdu, fits.BinTableHDU):
        if (
            hdu.header.get("INDXSCHM", "") in ["EXPLICIT", "IMPLICIT", ""]
            and len(hdu.columns) > 1
        ):
            has_cube_data = True

    if has_cube_data:
        if "EBOUNDS" in hdu_list:
            return "EBOUNDS"
        elif "ENERGIES" in hdu_list:
            return "ENERGIES"

    return None


def find_hdu(hdulist):
    """Find the first non-empty HDU."""
    for hdu in hdulist:
        if hdu.data is not None:
            return hdu

    raise AttributeError("No Image or BinTable HDU found.")


def find_bintable_hdu(hdulist):
    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.BinTableHDU):
            return hdu

    raise AttributeError("No BinTable HDU found.")


def edges_from_lo_hi(edges_lo, edges_hi):
    if np.isscalar(edges_lo.value) and np.isscalar(edges_hi.value):
        return u.Quantity([edges_lo, edges_hi])

    edges = edges_lo.copy()
    try:
        edges = edges.insert(len(edges), edges_hi[-1])
    except AttributeError:
        edges = np.insert(edges, len(edges), edges_hi[-1])
    return edges


def slice_to_str(slice_):
    return f"{slice_.start}:{slice_.stop}"


def str_to_slice(slice_str):
    start, stop = slice_str.split(":")
    return slice(int(start), int(stop))


class JsonQuantityEncoder(json.JSONEncoder):
    """Support for quantities that JSON default encoder"""
    def default(self, obj):
        if isinstance(obj, u.Quantity):
            return obj.to_string()

        return json.JSONEncoder.default(self, obj)


class JsonQuantityDecoder(json.JSONDecoder):
    """Support for quantities that JSON default encoder"""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(data):
        for key, value in data.items():
            try:
                data[key] = u.Quantity(value)
            except TypeError:
                continue
        return data
