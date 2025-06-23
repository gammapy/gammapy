import astropy.units as u
import numpy as np
from astropy.table import Table
from regions import *
from astropy.coordinates import SkyCoord
import os

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Exclusion regions for source position, stars, and other gamma-ray sources"""

def make_exclusion_mask(source_position,geom,rad=3*u.deg,max_star_mag=6,other_exc=[]):
    """
    Create an exclusion mask for the source position, any stars in the field of view,
    and any user-provided regions.

    Parameters
    ----------
    source_position : `~regions.SkyRegion`
        Source exclusion region coordinates (can be any geometry).
    geom: `~gammapy.maps.WcsGeom`
        Map geometry for the exclusion mask. 
    rad : `~astropy.units.Quantity`
        Radius around `source_position` to search for stellar exclusion regions.
    max_star_mag : 
        The maximum stellar magnitude (Johnson-Cousins B-band) to exclude. Maximum value is 8th magnitude.  
    other_exc : list of `~regions.SkyRegion`
        Other regions to exclude (can be any geometry). 

    Returns
    -------
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        2D exclusion mask 
    """
    exc = [source_position]

    for src in other_exc:
        exc.append(src)
    star_data = np.loadtxt(os.environ.get("GAMMAPY_DATA")+"/veritas/crab-point-like-ED/Hipparcos_MAG8_1997.dat",usecols=(0, 1, 2, 3))
    star_cat = Table(
    {
        "ra": star_data[:, 0],
        "dec": star_data[:, 1],
        "id": star_data[:, 2],
        "mag": star_data[:, 3],
    }
    )
    star_mask = (
    np.sqrt(
        (star_cat["ra"] - source_position.center.ra.deg) ** 2
        + (star_cat["dec"] - source_position.center.dec.deg) ** 2
    ) < rad.value
    )

    for src in star_cat[(star_mask) & (star_cat["mag"] < max_star_mag)]:
        exc.append(
            CircleSkyRegion(
                center=SkyCoord(src["ra"], src["dec"], unit="deg", frame="icrs"),
                radius=0.3 * u.deg,
            )
        )

    exclusion_mask = ~geom.region_mask(exc)

    return exclusion_mask
