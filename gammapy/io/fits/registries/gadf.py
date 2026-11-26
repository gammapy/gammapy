# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""GADF registries: versioned column and header-keyword definitions.

Pure data (plus the small builders that assemble it). Everything is keyed by
format version and HDU class key, e.g. ``GADF_PRODUCTS_HEADER_DEFINITION["0.3"]["EVENTS"]``.
This module imports only from ``gammapy.io.fits.registries`` and registers GADF
there (definitions and HDU class-key resolver).

Column ``dtype`` entries are given only where the GADF specification gives a
column type (event lists, GTI, pointing, WCS BANDS columns), and they are then
checked. The IRF and most BANDS column specifications give ``ndim`` and ``unit``
only, so their definitions carry no ``dtype``.
"""

import copy

from gammapy.io.fits.registries import DATA_FORMATS_MODELS, _kw, compose_header
from gammapy.io.fits.registries.common import (
    FITS_TIME_CONVENIENCE_KEYWORDS,
    FITS_TIME_KEYWORDS,
    GENERAL_KEYWORDS,
    HDU_KEYWORDS,
    HDU_RESPONSE_KEYWORDS,
)

# --------------------------------------------------------------------------
# DL3 event-level tables
# --------------------------------------------------------------------------
# Column types as in the GADF events page: EVENT_ID int64, TIME float64,
# EVENT_TYPE a bit field (FITS TFORM "32X"); "float" / "int" without a width.
GADF_V02_EVENT_TABLE_DEFINITION = {
    "EVENT_ID": {"dtype": "int64", "required": True, "unit": None},
    "TIME": {"dtype": "float64", "required": True, "unit": "s"},
    "RA": {"dtype": "float", "required": True, "unit": "deg"},
    "DEC": {"dtype": "float", "required": True, "unit": "deg"},
    "ENERGY": {"dtype": "float", "required": True, "unit": "TeV"},
    "EVENT_TYPE": {"dtype": "bit"},
    "MULTIP": {"dtype": "int"},
    "GLON": {"dtype": "float", "unit": "deg"},
    "GLAT": {"dtype": "float", "unit": "deg"},
    "ALT": {"dtype": "float", "unit": "deg"},
    "AZ": {"dtype": "float", "unit": "deg"},
    "DETX": {"dtype": "float", "unit": "deg"},
    "DETY": {"dtype": "float", "unit": "deg"},
    "THETA": {"dtype": "float", "unit": "deg"},
    "PHI": {"dtype": "float", "unit": "deg"},
    "DIR_ERR": {"dtype": "float", "unit": "deg"},
    "ENERGY_ERR": {"dtype": "float", "unit": "TeV"},
    "COREX": {"dtype": "float", "unit": "m"},
    "COREY": {"dtype": "float", "unit": "m"},
    "CORE_ERR": {"dtype": "float", "unit": "m"},
    "XMAX": {"dtype": "float", "unit": "m"},
    "XMAX_ERR": {"dtype": "float", "unit": "m"},
    "HIL_MSW": {"dtype": "float", "unit": ""},
    "HIL_MSW_ERR": {"dtype": "float", "unit": ""},
    "HIL_MSL": {"dtype": "float", "unit": ""},
    "HIL_MSL_ERR": {"dtype": "float", "unit": ""},
}

GADF_V02_GTI_TABLE_DEFINITION = {
    "START": {"dtype": "float64", "required": True, "unit": "s"},
    "STOP": {"dtype": "float64", "required": True, "unit": "s"},
}

GADF_V02_POINTING_TABLE_DEFINITION = {
    "TIME": {"dtype": "float64", "required": True, "unit": "s"},
    "RA_PNT": {"dtype": "float", "required": True, "unit": "deg"},
    "DEC_PNT": {"dtype": "float", "required": True, "unit": "deg"},
    "ALT_PNT": {"dtype": "float", "unit": "deg"},
    "AZ_PNT": {"dtype": "float", "unit": "deg"},
}

# --------------------------------------------------------------------------
# DL3 IRF tables
# --------------------------------------------------------------------------
# The GADF IRF pages give ndim and unit only: no column dtype.
GADF_V02_AEFF_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "EFFAREA": {"required": True, "unit": "m2", "ndim": 2},
}

GADF_V02_EDISP_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "MIGRA_LO": {"required": True, "unit": "", "ndim": 1},
    "MIGRA_HI": {"required": True, "unit": "", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "MATRIX": {"required": True, "unit": "", "ndim": 3},
}

GADF_V02_PSF_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "RAD_LO": {"required": True, "unit": "deg", "ndim": 1},
    "RAD_HI": {"required": True, "unit": "deg", "ndim": 1},
    "RPSF": {"required": True, "unit": "sr-1", "ndim": 3},
}

GADF_V02_PSF_3GAUSS_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "SCALE": {"required": True, "unit": "", "ndim": 2},
    "SIGMA_1": {"required": True, "unit": "deg", "ndim": 2},
    "SIGMA_2": {"required": True, "unit": "deg", "ndim": 2},
    "SIGMA_3": {"required": True, "unit": "deg", "ndim": 2},
    "AMPL_2": {"required": True, "unit": "", "ndim": 2},
    "AMPL_3": {"required": True, "unit": "", "ndim": 2},
}

GADF_V02_PSF_KING_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "GAMMA": {"required": True, "unit": "", "ndim": 2},
    "SIGMA": {"required": True, "unit": "", "ndim": 2},
}

GADF_V02_PSF_TABLE_DEFINITION = {
    "ENERGY": {"required": True, "unit": "MeV", "ndim": 1},
    "EXPOSURE": {"required": True, "unit": "cm2 s", "ndim": 1},
    "PSF": {"required": True, "unit": "", "ndim": 2},
}

GADF_V02_THETA_TABLE_DEFINITION = {
    "THETA": {"required": True, "unit": "deg", "ndim": 1},
}

GADF_V02_BKG_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "BKG": {"required": True, "unit": "s-1 MeV-1 sr-1", "ndim": 2},
}

GADF_V02_BKG_3D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "DETX_LO": {"required": True, "unit": "deg", "ndim": 1},
    "DETX_HI": {"required": True, "unit": "deg", "ndim": 1},
    "DETY_LO": {"required": True, "unit": "deg", "ndim": 1},
    "DETY_HI": {"required": True, "unit": "deg", "ndim": 1},
    "BKG": {"required": True, "unit": "s-1 MeV-1 sr-1", "ndim": 3},
}

GADF_V02_RAD_MAX_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"required": True, "unit": "deg", "ndim": 1},
    "RAD_MAX": {"required": True, "unit": "deg", "ndim": 2},
}

# --------------------------------------------------------------------------
# DL4 Map tables
# --------------------------------------------------------------------------
# BANDS: GADF gives ndim (and unit) only, except the WCS columns below.
GADF_V02_BANDS_TABLE_DEFINITION = {
    "CHANNEL": {"required": True, "unit": "", "ndim": 1},
    "E_MIN": {"required": False, "unit": "keV", "ndim": 1},
    "E_MAX": {"required": False, "unit": "keV", "ndim": 1},
    "ENERGY": {"required": False, "unit": "keV", "ndim": 1},
    "EVENT_TYPE": {"required": False, "unit": "", "ndim": 1},
}

GADF_V02_WCS_TABLE_DEFINITION = {
    "NPIX": {"dtype": "int", "required": False, "unit": "", "ndim": 2},
    "CRPIX": {"dtype": "float", "required": False, "unit": "deg", "ndim": 2},
    "CDELT": {"dtype": "float", "required": False, "unit": "deg", "ndim": 2},
}

GADF_V02_HPX_TABLE_DEFINITION = {
    "NSIDE": {"required": True, "unit": "", "ndim": 1},
}

# --------------- IRF DL3 HDU SPECIFICATION ---------------
# The key is the class tag.
GADF_IRF_DL3_HDU_SPECIFICATION = {
    "bkg_3d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "BKG_3D",
            "FOVALIGN": "RADEC",
        },
    },
    "bkg_2d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "BKG_2D",
        },
    },
    "edisp_2d": {
        "extname": "ENERGY DISPERSION",
        "column_name": "MATRIX",
        "mandatory_keywords": {
            "HDUCLAS2": "EDISP",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "EDISP_2D",
        },
    },
    "psf_table": {
        "extname": "PSF_2D_TABLE",
        "column_name": "RPSF",
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
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
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "PSF_3GAUSS",
        },
    },
    "psf_king": {
        "extname": "PSF_2D_KING",
        "column_name": {"sigma": "SIGMA", "gamma": "GAMMA"},
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "PSF_KING",
        },
    },
    "psf_gtpsf": {
        "extname": "PSF_GTPSF",
        "mandatory_keywords": {
            "HDUCLAS2": "PSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "GTPSF",
        },
    },
    "aeff_2d": {
        "extname": "EFFECTIVE AREA",
        "column_name": "EFFAREA",
        "mandatory_keywords": {
            "HDUCLAS2": "EFF_AREA",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "AEFF_2D",
        },
    },
    "rad_max_2d": {
        "extname": "RAD_MAX",
        "column_name": "RAD_MAX",
        "mandatory_keywords": {
            "HDUCLAS2": "RAD_MAX",
            "HDUCLAS3": "POINT-LIKE",
            "HDUCLAS4": "RAD_MAX_2D",
        },
    },
}


# EXTNAME -> IRF tag, used when HDUCLAS4 is missing. EXTNAME values shared by
# several tags (e.g. BACKGROUND for bkg_2d and bkg_3d) are left out: such an
# HDU cannot be identified from its EXTNAME alone.
_EXTNAMES = [spec["extname"] for spec in GADF_IRF_DL3_HDU_SPECIFICATION.values()]
GADF_EXTNAME_TO_TAG = {
    spec["extname"]: tag
    for tag, spec in GADF_IRF_DL3_HDU_SPECIFICATION.items()
    if "extname" in spec and _EXTNAMES.count(spec["extname"]) == 1
}


GADF_IRF_MAP_HDU_SPECIFICATION = {
    "edisp_kernel_map": "edisp",
    "edisp_map": "edisp",
    "psf_map": "psf",
    "psf_map_reco": "psf",
}

# --------------- VERSION / DOC REGISTRIES ---------------
GADF_DEFAULT_VERSION = "0.3"

_GADF_V02_TABLE = {
    "EVENTS": GADF_V02_EVENT_TABLE_DEFINITION,
    "GTI": GADF_V02_GTI_TABLE_DEFINITION,
    "POINTING": GADF_V02_POINTING_TABLE_DEFINITION,
    "AEFF_2D": GADF_V02_AEFF_2D_TABLE_DEFINITION,
    "EDISP_2D": GADF_V02_EDISP_2D_TABLE_DEFINITION,
    "PSF_TABLE": GADF_V02_PSF_2D_TABLE_DEFINITION,
    "PSF_3GAUSS": GADF_V02_PSF_3GAUSS_TABLE_DEFINITION,
    "PSF_KING": GADF_V02_PSF_KING_TABLE_DEFINITION,
    "PSF": GADF_V02_PSF_TABLE_DEFINITION,
    "THETA": GADF_V02_THETA_TABLE_DEFINITION,
    "BKG_2D": GADF_V02_BKG_2D_TABLE_DEFINITION,
    "BKG_3D": GADF_V02_BKG_3D_TABLE_DEFINITION,
    "RAD_MAX_2D": GADF_V02_RAD_MAX_2D_TABLE_DEFINITION,
    "BANDS": GADF_V02_BANDS_TABLE_DEFINITION,
    "IMAGE": GADF_V02_WCS_TABLE_DEFINITION,
    "SKYMAP": GADF_V02_HPX_TABLE_DEFINITION,
}

_GADF_V03_TABLE = copy.deepcopy(_GADF_V02_TABLE)
_GADF_V03_TABLE["PSF_3GAUSS"]["SCALE"]["unit"] = "sr-1"
_GADF_V03_TABLE["EVENTS"]["GAMMANESS"] = {"dtype": "float"}

# Nested registry: version -> HDU -> column-definition YAML.
GADF_PRODUCTS_TABLE_DEFINITION = {
    "0.2": _GADF_V02_TABLE,
    "0.3": _GADF_V03_TABLE,
}

GADF_HDUDOC = {
    "0.2": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
    "0.3": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
}

# =====================================================================
# GADF HEADER KEYWORD DEFINITIONS
# Same layout as the table definitions: version -> HDU class key -> {keyword: spec}.
# Spec entries (dtype, required, required_if, required_unless, allowed,
# default, unit, comment) are documented in gammapy.io.fits.registries.
# =====================================================================

# --------------- GADF keyword blocks (all optional here) ---------------
GADF_EARTH_LOCATION_KEYWORDS = {
    key: _kw("float") for key in ("GEOLON", "GEOLAT", "ALTITUDE")
}

GADF_POINTING_KEYWORDS = {
    key: _kw("float") for key in ("RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT")
}

GADF_OBS_KEYWORDS = {
    "OBS_ID": _kw(["int", "str"]),
    "OBS_MODE": _kw("str"),
    "TSTART": _kw("float"),
    "TSTOP": _kw("float"),
    "ONTIME": _kw("float"),
    "LIVETIME": _kw("float"),
    "DEADC": _kw("float"),
    "EQUINOX": _kw(["float", "str"]),
    "RADECSYS": _kw("str"),
}

GADF_EVENTS_OPTIONAL_KEYWORDS = {
    "OBJECT": _kw("str"),
    "RA_OBJ": _kw("float"),
    "DEC_OBJ": _kw("float"),
    "OBSERVER": _kw("str"),
    "EV_CLASS": _kw(["int", "str"]),
    "TELAPSE": _kw("float"),
    "TASSIGN": _kw("str"),
    "TELLIST": _kw("str"),  # comma-separated string
    "N_TELS": _kw("int"),
    "DST_VER": _kw(["int", "str"]),
    "ANA_VER": _kw(["int", "str"]),
    "CAL_VER": _kw(["int", "str"]),
    **{
        key: _kw("float")
        for key in (
            "CONV_DEP",
            "CONV_RA",
            "CONV_DEC",
            "TRGRATE",
            "ZTRGRATE",
            "MUONEFF",
            "BROKPIX",
            "AIRTEMP",
            "PRESSURE",
            "RELHUM",
            "NSBLEVEL",
        )
    },
}

GADF_RESPONSE_OPTIONAL_KEYWORDS = {
    "EXTNAME": _kw("str"),
    "OBS_ID": _kw("int"),
    "LO_THRES": _kw("float"),
    "HI_THRES": _kw("float"),
    "FOVALIGN": _kw("str"),
}

GADF_WCS_SKYMAP_KEYWORDS = {
    "BANDSHDU": _kw("str"),  # name of the companion BANDS HDU
}

GADF_HPX_SKYMAP_KEYWORDS = {
    "PIXTYPE": _kw("str"),  # must be HEALPIX
    "ORDERING": _kw("str"),  # NESTED | RING
    "INDXSCHM": _kw("str"),  # IMPLICIT | EXPLICIT | SPARSE (default IMPLICIT)
    "ORDER": _kw("int"),  # log2(NSIDE) or -1
    "NSIDE": _kw("int"),  # superseded by NSIDE column if BANDS defined
    "COORDSYS": _kw("str"),  # CEL | GAL
    "BANDSHDU": _kw("str"),
}

# --------------- mandatory sets ---------------
_HDU_REQUIRED = ("HDUCLASS", "HDUDOC", "HDUVERS", "HDUCLAS1")

# FITS reference time: mandatory for EVENTS, GTI and POINTING ("The standard FITS
# reference time header keywords should be used"; general/time: "New files should
# always be written with all five header keys").
_TIME_REQUIRED = tuple(FITS_TIME_KEYWORDS)

# Observatory Earth location: mandatory for EVENTS and POINTING ("An observatory
# Earth location should be given as well"; keys from general/coordinates).
_EARTH_LOCATION_REQUIRED = tuple(GADF_EARTH_LOCATION_KEYWORDS)

_EVENTS_REQUIRED = (
    _HDU_REQUIRED
    + _TIME_REQUIRED
    + _EARTH_LOCATION_REQUIRED
    + (
        "OBS_ID",
        "TSTART",
        "TSTOP",
        "ONTIME",
        "LIVETIME",
        "DEADC",
        "EQUINOX",
        "RADECSYS",
        "ORIGIN",
        "TELESCOP",
        "INSTRUME",
        "CREATOR",
    )
)

_IRF_REQUIRED = _HDU_REQUIRED + ("HDUCLAS2", "HDUCLAS3", "HDUCLAS4")

# Per-IRF-type mandatory keywords, on top of _IRF_REQUIRED.
_IRF_TAG_REQUIRED = {
    "bkg_2d": ("FOVALIGN",),
    "bkg_3d": ("FOVALIGN",),
}

# HDU class keys of the DL3 IRFs (e.g. AEFF_2D, BKG_3D).
GADF_IRF_CLASS_KEYS = frozenset(
    spec["mandatory_keywords"]["HDUCLAS4"]
    for spec in GADF_IRF_DL3_HDU_SPECIFICATION.values()
) | {"PSF"}  # "PSF": class key used for GTPSF tables in the table registry


def _gadf_hdu_keywords(version, hduclas1=None):
    """HDU keywords with the GADF defaults for ``version``."""
    updates = {
        "HDUCLASS": {"default": "GADF"},
        "HDUVERS": {"default": version, "allowed": list(GADF_HDUDOC)},
        "HDUDOC": {"default": GADF_HDUDOC[version]},
    }
    if hduclas1:
        updates["HDUCLAS1"] = {"default": hduclas1}
    return compose_header(HDU_KEYWORDS, updates=updates)


def _gadf_header_definitions(version):
    """Part of the GADF header registry that is common to all versions."""

    def hdu(hduclas1=None):
        return _gadf_hdu_keywords(version, hduclas1)

    headers = {
        "EVENTS": compose_header(
            hdu("EVENTS"),
            GENERAL_KEYWORDS,
            GADF_OBS_KEYWORDS,
            FITS_TIME_KEYWORDS,
            FITS_TIME_CONVENIENCE_KEYWORDS,
            GADF_EARTH_LOCATION_KEYWORDS,
            GADF_POINTING_KEYWORDS,
            GADF_EVENTS_OPTIONAL_KEYWORDS,
            required=_EVENTS_REQUIRED,
        ),
        "GTI": compose_header(
            hdu("GTI"),
            FITS_TIME_KEYWORDS,
            FITS_TIME_CONVENIENCE_KEYWORDS,
            required=_TIME_REQUIRED,
        ),
        "POINTING": compose_header(
            hdu("POINTING"),
            FITS_TIME_KEYWORDS,
            FITS_TIME_CONVENIENCE_KEYWORDS,
            GADF_EARTH_LOCATION_KEYWORDS,
            required=_TIME_REQUIRED + _EARTH_LOCATION_REQUIRED,
        ),
        "BANDS": hdu("BANDS"),
        "IMAGE": compose_header(hdu("IMAGE"), GADF_WCS_SKYMAP_KEYWORDS),
        "SKYMAP": compose_header(
            hdu("SKYMAP"),
            GADF_HPX_SKYMAP_KEYWORDS,
            required=("PIXTYPE",),
            updates={"PIXTYPE": {"default": "HEALPIX"}},
        ),
    }
    for tag, spec in GADF_IRF_DL3_HDU_SPECIFICATION.items():
        class_key = spec["mandatory_keywords"]["HDUCLAS4"]
        headers[class_key] = compose_header(
            hdu("RESPONSE"),
            HDU_RESPONSE_KEYWORDS,
            GENERAL_KEYWORDS,
            GADF_RESPONSE_OPTIONAL_KEYWORDS,
            required=_IRF_REQUIRED + _IRF_TAG_REQUIRED.get(tag, ()),
        )
    headers["PSF"] = headers["GTPSF"]  # GTPSF table class key, see GADF_IRF_CLASS_KEYS
    # THETA axis HDU of a GTPSF file: no keywords specified for it beyond HDUCLASn
    headers["THETA"] = hdu()
    return headers


# --------------- v0.2 ---------------
# RA_PNT/DEC_PNT unconditionally mandatory; OBS_MODE not required; no DRIFT support.
_GADF_V02_HEADER = _gadf_header_definitions("0.2")
_GADF_V02_HEADER["EVENTS"] = compose_header(
    _GADF_V02_HEADER["EVENTS"], required=("RA_PNT", "DEC_PNT")
)

# --------------- v0.3 ---------------
# OBS_MODE mandatory, with defined values; OBS_MODE=DRIFT -> ALT_PNT/AZ_PNT
# required, RA_PNT/DEC_PNT not; otherwise -> RA_PNT/DEC_PNT required, ALT_PNT/AZ_PNT not.
_GADF_V03_HEADER = _gadf_header_definitions("0.3")
_GADF_V03_HEADER["EVENTS"] = compose_header(
    _GADF_V03_HEADER["EVENTS"],
    required=("OBS_MODE",),
    updates={
        "OBS_MODE": {"allowed": ["POINTING", "RASTER", "SLEW", "SCAN", "DRIFT"]},
        "RA_PNT": {"required_unless": {"OBS_MODE": "DRIFT"}},
        "DEC_PNT": {"required_unless": {"OBS_MODE": "DRIFT"}},
        "ALT_PNT": {"required_if": {"OBS_MODE": "DRIFT"}},
        "AZ_PNT": {"required_if": {"OBS_MODE": "DRIFT"}},
    },
)

# Nested registry: version -> HDU class key -> keyword definition.
GADF_PRODUCTS_HEADER_DEFINITION = {
    "0.2": _GADF_V02_HEADER,
    "0.3": _GADF_V03_HEADER,
}


# --------------- HDU CLASS KEY RESOLUTION ---------------
# Class keys taken as they are from HDUCLAS4 / HDUCLAS1 / EXTNAME.
GADF_DIRECT_CLASS_KEYS = frozenset(
    {
        "EVENTS",
        "GTI",
        "POINTING",
        "AEFF_2D",
        "EDISP_2D",
        "PSF_TABLE",
        "PSF_3GAUSS",
        "PSF_KING",
        "BKG_2D",
        "BKG_3D",
        "RAD_MAX_2D",
        "PSF_MAP",
    }
)


def resolve_irf_tag_from_meta(meta):
    hduclas4 = meta.get("HDUCLAS4")
    if hduclas4 and hduclas4.lower() in GADF_IRF_DL3_HDU_SPECIFICATION:
        return hduclas4.lower()
    return GADF_EXTNAME_TO_TAG.get(meta.get("EXTNAME"))


def gadf_hdu_class_key(meta):
    """HDU class key of a GADF header (selects the TABLE and HEADER definitions)."""
    key = meta.get("HDUCLAS4") or meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    if key in GADF_DIRECT_CLASS_KEYS:
        return key
    tag = None
    if not meta.get("HDUCLAS4"):  # only guess from EXTNAME if HDUCLAS4 missing
        tag = GADF_EXTNAME_TO_TAG.get(meta.get("EXTNAME"))
    if tag:
        return GADF_IRF_DL3_HDU_SPECIFICATION[tag]["mandatory_keywords"]["HDUCLAS4"]

    hdu_class = meta.get("HDUCLAS2", "unknown_clas")
    if hdu_class in ["EFF_AREA", "RPSF", "EDISP", "BKG"] and meta.get("HDUCLAS4"):
        # CTA-1DC: non-standard HDUCLAS4 -> bkg_3d, ... (lazy: only on this path)
        from gammapy.irf.io import _get_hdu_type_and_class

        _, hdu_class = _get_hdu_type_and_class(meta)
    return hdu_class.upper()


# --------------- REGISTRATION ---------------
DATA_FORMATS_MODELS["GADF"] = {
    "TABLE": GADF_PRODUCTS_TABLE_DEFINITION,  # version -> class key -> columns
    "HEADER": GADF_PRODUCTS_HEADER_DEFINITION,  # version -> class key -> keywords
    "HDU_CLASS_KEY": gadf_hdu_class_key,  # header -> class key
}
