# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.io import fits
from gammapy.utils.random import get_random_state


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


def interp_to_order(interp):
    """Convert interpolation string to order."""
    if isinstance(interp, int):
        return interp

    order_map = {None: 0, "nearest": 0, "linear": 1, "quadratic": 2, "cubic": 3}
    return order_map.get(interp, None)


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
