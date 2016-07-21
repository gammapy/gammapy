# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser

__all__ = ['image_coordinates']

log = logging.getLogger(__name__)


def image_coordinates_main(args=None):
    parser = get_parser(image_coordinates)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('outfile', type=str,
                        help='Output FITS file name')
    parser.add_argument('--make_coordinate_maps', action='store_true',
                        help='Create coordinate maps')
    parser.add_argument('--make_distance_map', action='store_true',
                        help='Create distance to mask map')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    image_coordinates(**vars(args))


def image_coordinates(infile,
                      outfile,
                      make_coordinate_maps,
                      make_distance_map,
                      overwrite):
    """Make maps that can be used to create profiles.

    The following images can be created:
    * LON -- Longitude coordinate
    * LAT -- Latitude coordinate
    * DIST -- Distance to mask
    * SOLID_ANGLE -- Solid angle
    """
    from astropy.io import fits
    from gammapy.utils.fits import get_hdu
    from gammapy.image import SkyImage, ExclusionMask

    log.info('Reading {0}'.format(infile))
    hdu = get_hdu(infile)

    out_hdus = fits.HDUList()

    if make_coordinate_maps:
        image = SkyImage.empty_like(hdu)
        log.info('Computing LON and LAT maps')
        lon, lat = image.coordinates()
        out_hdus.append(fits.ImageHDU(lon, hdu.header, 'LON'))
        out_hdus.append(fits.ImageHDU(lat, hdu.header, 'LAT'))

    if make_distance_map:
        excl = ExclusionMask.from_image_hdu(hdu)
        log.info('Computing DIST map')
        dist = excl.exclusion_distance
        out_hdus.append(fits.ImageHDU(dist, hdu.header, 'DIST'))

    log.info('Writing {0}'.format(outfile))
    out_hdus.writeto(outfile, clobber=overwrite)
