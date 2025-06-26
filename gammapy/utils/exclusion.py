# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.table import Table
from regions import CircleSkyRegion
from astropy.coordinates import SkyCoord
import os

"""Exclusion regions for source position, stars, and other gamma-ray sources"""


def make_exclusion_mask(
    source_region, geom, max_star_mag=6, star_rad=3 * u.deg, other_exc=[]
):
    """
    Create an exclusion mask for the source position, any stars in the field of view,
    and any user-provided regions.

    Parameters
    ----------
    source_region : `~regions.SkyRegion`
        Source exclusion region coordinates (can be any geometry).
    geom: `~gammapy.maps.WcsGeom`
        Map geometry for the exclusion mask.
    max_star_mag :
        The maximum stellar magnitude (Johnson-Cousins B-band) to exclude. Maximum value is 8th magnitude.
    star_rad : `~astropy.units.Quantity`
        Radius to exclude for each star.
    other_exc : list of `~regions.SkyRegion`
        Other regions to exclude (can be any geometry).

    Returns
    -------
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        2D exclusion mask
    """
    exc = [source_region]

    for src in other_exc:
        exc.append(src)
    star_cat = Table.read(
        os.environ.get("GAMMAPY_DATA")
        + "/veritas/crab-point-like-ED/Hipparcos_MAG8_1997.dat",
        format="ascii.commented_header",
    )
    star_mask = geom.contains(
        SkyCoord(star_cat["_RAJ2000"] * u.deg, star_cat["_DEJ2000"] * u.deg)
    )

    for src in star_cat[
        (star_mask) & (star_cat["B-V"] + star_cat["Vmag"] < max_star_mag)
    ]:
        exc.append(
            CircleSkyRegion(
                center=SkyCoord(
                    src["_RAJ2000"], src["_DEJ2000"], unit="deg", frame="icrs"
                ),
                radius=star_rad,
            )
        )

    exclusion_mask = ~geom.region_mask(exc)

    return exclusion_mask

