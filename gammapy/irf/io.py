# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.io import fits
from gammapy.data.hdu_index_table import HDUIndexTable
from gammapy.utils.fits import HDULocation
from gammapy.utils.scripts import make_path

__all__ = ["load_cta_irfs", "load_irf_dict_from_file"]

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
    """Check if a GADF IRF is pointlike based on the header"""
    return header.get("HDUCLAS3") == "POINT-LIKE"


def load_cta_irfs(filename):
    """load CTA instrument response function and return a dictionary container.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.

    The various IRFs are accessible with the following keys:

    - 'aeff' is a `~gammapy.irf.EffectiveAreaTable2D`
    - 'edisp'  is a `~gammapy.irf.EnergyDispersion2D`
    - 'psf' is a `~gammapy.irf.EnergyDependentMultiGaussPSF`
    - 'bkg' is a  `~gammapy.irf.Background3D`

    Parameters
    ----------
    filename : str
        the input filename. Default is

    Returns
    -------
    cta_irf : dict
        the IRF dictionary

    Examples
    --------
    Access the CTA 1DC IRFs stored in the gammapy datasets

    >>> from gammapy.irf import load_cta_irfs
    >>> filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    >>> cta_irf = load_cta_irfs(filename)
    >>> print(cta_irf['aeff'])
    EffectiveAreaTable2D
    --------------------
    <BLANKLINE>
      axes  : ['energy_true', 'offset']
      shape : (42, 6)
      ndim  : 2
      unit  : m2
      dtype : >f4
    <BLANKLINE>
    """
    from .background import Background3D
    from .edisp import EnergyDispersion2D
    from .effective_area import EffectiveAreaTable2D
    from .psf import EnergyDependentMultiGaussPSF

    aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")
    bkg = Background3D.read(filename, hdu="BACKGROUND")
    edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    return dict(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf)


def load_irf_dict_from_file(filename):
    """Open a fits file and generate a dictionary containing the Gammapy objects
    corresponding to the IRF components stored

    Parameters
    ----------
    filename : str, Path
        path to the file containing the IRF components, if EVENTS and GTI HDUs
        are included in the file, they are ignored

    Returns
    -------
    irf_dict : dict of `~gammapy.irf.IRF`
        dictionary with instances of the Gammapy objects corresponding
        to the IRF components
    """
    from .rad_max import RadMax2D

    filename = make_path(filename)

    hdulist = fits.open(make_path(filename))

    irf_dict = {}

    is_pointlike = False

    for hdu in hdulist:
        hdu_class = hdu.header.get("HDUCLAS1", "").lower()

        if hdu_class == "response":
            hdu_class = hdu.header.get("HDUCLAS4", "").lower()

            is_pointlike |= hdu.header["HDUCLAS3"] == "POINT-LIKE"

            loc = HDULocation(
                hdu_class=hdu_class,
                hdu_name=hdu.name,
                file_dir=filename.parent,
                file_name=filename.name,
            )

            for name in HDUIndexTable.VALID_HDU_TYPE:
                if name in hdu_class:
                    if name in irf_dict.keys():
                        log.warning(f"more than one HDU of {name} type found")
                        log.warning(
                            f"loaded the {irf_dict[name].meta['EXTNAME']} HDU in the dictionary"
                        )
                        continue
                    data = loc.load()
                    # TODO: maybe introduce IRF.type attribute...
                    irf_dict[name] = data
        else:  # not an IRF component
            continue

    if is_pointlike and "rad_max" not in irf_dict:
        irf_dict["rad_max"] = RadMax2D.from_irf(irf_dict["aeff"])

    return irf_dict
