# Licensed under a 3-clause BSD style license - see LICENSE.rst
import json
from astropy import units as u
from astropy.io import fits


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
