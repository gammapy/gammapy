# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from ..utils.random import get_random_state


def fill_poisson(map_in, mu, random_state="random-seed"):
    """Fill a map object with a poisson random variable.

    This can be useful for testing, to make a simulated counts image.
    E.g. filling with ``mu=0.5`` fills the map so that many pixels
    have value 0 or 1, and a few more "counts".

    Parameters
    ----------
    map_in : `~gammapy.maps.Map`
        Input map
    mu : scalar or `~numpy.ndarray`
        Expectation value
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """
    random_state = get_random_state(random_state)
    idx = map_in.geom.get_idx(flat=True)
    mu = random_state.poisson(mu, idx[0].shape)
    map_in.fill_by_idx(idx, mu)


def swap_byte_order(arr_in):
    """Swap the byte order of a numpy array to the native one.

    Parameters
    ----------
    arr_in : `~numpy.ndarray`
        Input array.

    Returns
    -------
    arr_out : `~numpy.ndarray`
        Array with native byte order.
    """
    if arr_in.dtype.byteorder not in ("=", "|"):
        return arr_in.byteswap().newbyteorder()

    return arr_in


def interp_to_order(interp):
    """Convert interpolation string to order."""
    if isinstance(interp, int):
        return interp

    order_map = {None: 0, "nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3}
    return order_map.get(interp, None)


def unpack_seq(seq, n=1):
    """Utility to unpack the first N values of a tuple or list.  Remaining
    values are put into a single list which is the last element of the
    return value.  This partially simulates the extended unpacking
    functionality available in Python 3.

    Parameters
    ----------
    seq : list or tuple
        Input sequence to be unpacked.
    n : int
        Number of elements of ``seq`` to unpack.  Remaining elements
        are put into a single tuple.
    """
    for row in seq:
        yield [e for e in row[:n]] + [row[n:]]


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


def find_image_hdu(hdulist):
    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.ImageHDU):
            return hdu

    raise AttributeError("No Image HDU found.")


def find_bintable_hdu(hdulist):
    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.BinTableHDU):
            return hdu

    raise AttributeError("No BinTable HDU found.")
