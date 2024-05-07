# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.io import fits


def find_bands_hdu(hdu_list, hdu):
    """Discover the extension name of the BANDS HDU.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
        The FITS HDU list.
    hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
        The HDU to check.
    Returns
    -------
    hduname : str
        Extension name of the BANDS HDU. Returns None if no BANDS HDU was found.
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
