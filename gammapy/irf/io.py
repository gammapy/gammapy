# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.io import fits
from gammapy.data.hdu_index_table import HDUIndexTable
from gammapy.utils.fits import HDULocation
from gammapy.utils.scripts import make_path

__all__ = ["load_irf_dict_from_file"]

log = logging.getLogger(__name__)


IRF_DL3_AXES_SPECIFICATION = {
    "THETA": {"name": "offset", "interp": "lin"},
    "ENERG": {"name": "energy_true", "interp": "log"},
    "ETRUE": {"name": "energy_true", "interp": "log"},
    "RAD": {"name": "rad", "interp": "lin"},
    "DETX": {"name": "fov_lon", "interp": "lin"},
    "DETY": {"name": "fov_lat", "interp": "lin"},
    "MIGRA": {"name": "migra", "interp": "lin"},
}

COMMON_HEADERS = {
    "HDUCLASS": "GADF",
    "HDUDOC": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
    "HDUVERS": "0.2",
}

COMMON_IRF_HEADERS = {
    **COMMON_HEADERS,
    "HDUCLAS1": "RESPONSE",
}


# The key is the class tag.
# TODO: extend the info here with the minimal header info
IRF_DL3_HDU_SPECIFICATION = {
    "bkg_3d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "BKG_3D",
            "FOVALIGN": "RADEC",
        },
    },
    "bkg_2d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "BKG_2D",
        },
    },
    "edisp_2d": {
        "extname": "ENERGY DISPERSION",
        "column_name": "MATRIX",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "EDISP",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "EDISP_2D",
        },
    },
    "psf_table": {
        "extname": "PSF_2D_TABLE",
        "column_name": "RPSF",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_TABLE",
        },
    },
    "psf_3gauss": {
        "extname": "PSF_2D_GAUSS",
        "column_name": {
            "sigma_1": "SIGMA_1",
            "sigma_2": "SIGMA_2",
            "sigma_3": "SIGMA_3",
            "scale": "SCALE",
            "ampl_2": "AMPL_2",
            "ampl_3": "AMPL_3",
        },
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_3GAUSS",
        },
    },
    "psf_king": {
        "extname": "PSF_2D_KING",
        "column_name": {
            "sigma": "SIGMA",
            "gamma": "GAMMA",
        },
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_KING",
        },
    },
    "aeff_2d": {
        "extname": "EFFECTIVE AREA",
        "column_name": "EFFAREA",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "EFF_AREA",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "AEFF_2D",
        },
    },
    "rad_max_2d": {
        "extname": "RAD_MAX",
        "column_name": "RAD_MAX",
        "mandatory_keywords": {
            **COMMON_IRF_HEADERS,
            "HDUCLAS2": "RAD_MAX",
            "HDUCLAS3": "POINT-LIKE",
            "HDUCLAS4": "RAD_MAX_2D",
        },
    },
}


IRF_MAP_HDU_SPECIFICATION = {
    "edisp_kernel_map": "edisp",
    "edisp_map": "edisp",
    "psf_map": "psf",
    "psf_map_reco": "psf",
}


def gadf_is_pointlike(header):
    """Check if a GADF IRF is pointlike based on the header."""
    return header.get("HDUCLAS3") == "POINT-LIKE"


class UnknownHDUClass(IOError):
    """Raised when a file contains an unknown HDUCLASS."""


def _get_hdu_type_and_class(header):
    """Get gammapy hdu_type and class from FITS header.

    Contains a workaround to support CTA 1DC irf file.
    """
    hdu_clas2 = header.get("HDUCLAS2", "")
    hdu_clas4 = header.get("HDUCLAS4", "")

    clas2_to_type = {"rpsf": "psf", "eff_area": "aeff"}
    hdu_type = clas2_to_type.get(hdu_clas2.lower(), hdu_clas2.lower())
    hdu_class = hdu_clas4.lower()

    if hdu_type not in HDUIndexTable.VALID_HDU_TYPE:
        raise UnknownHDUClass(f"HDUCLAS2={hdu_clas2}, HDUCLAS4={hdu_clas4}")

    # workaround for CTA 1DC files with non-compliant HDUCLAS4 names
    if hdu_class not in HDUIndexTable.VALID_HDU_CLASS:
        hdu_class = f"{hdu_type}_{hdu_class}"

    if hdu_class not in HDUIndexTable.VALID_HDU_CLASS:
        raise UnknownHDUClass(f"HDUCLAS2={hdu_clas2}, HDUCLAS4={hdu_clas4}")

    return hdu_type, hdu_class


def load_irf_dict_from_file(filename):
    """Load all available IRF components from given file into a dictionary.

    If multiple IRFs of the same type are present, the first encountered is returned.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Path to the file containing the IRF components, if EVENTS and GTI HDUs
        are included in the file, they are ignored.

    Returns
    -------
    irf_dict : dict of `~gammapy.irf.IRF`
        Dictionary with instances of the Gammapy objects corresponding
        to the IRF components.
    """
    from .rad_max import RadMax2D

    filename = make_path(filename)
    irf_dict = {}
    is_pointlike = False

    with fits.open(filename) as hdulist:
        for hdu in hdulist:
            hdu_clas1 = hdu.header.get("HDUCLAS1", "").lower()

            # not an IRF component
            if hdu_clas1 != "response":
                continue

            is_pointlike |= hdu.header.get("HDUCLAS3") == "POINT-LIKE"

            try:
                hdu_type, hdu_class = _get_hdu_type_and_class(hdu.header)
            except UnknownHDUClass as e:
                log.warning("File has unknown class %s", e)
                continue

            loc = HDULocation(
                hdu_class=hdu_class,
                hdu_name=hdu.name,
                file_dir=filename.parent,
                file_name=filename.name,
            )

            if hdu_type in irf_dict.keys():
                log.warning(f"more than one HDU of {hdu_type} type found")
                log.warning(
                    f"loaded the {irf_dict[hdu_type].meta['EXTNAME']} HDU in the dictionary"
                )
                continue

            data = loc.load()
            irf_dict[hdu_type] = data

    if is_pointlike and "rad_max" not in irf_dict:
        irf_dict["rad_max"] = RadMax2D.from_irf(irf_dict["aeff"])

    return irf_dict
