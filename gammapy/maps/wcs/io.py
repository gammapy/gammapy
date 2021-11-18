# Licensed under a 3-clause BSD style license - see LICENSE.rst


def identify_wcs_format(hdu):
    if hdu is None:
        return "gadf"
    elif hdu.name == "ENERGIES":
        return "fgst-template"
    elif hdu.name == "EBOUNDS":
        return "fgst-ccube"
    else:
        return "gadf"
