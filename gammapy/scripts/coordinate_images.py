# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['coordinate_images']


def main(args=None):
    parser = get_parser(coordinate_images)
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
    coordinate_images(**vars(args))


def coordinate_images(infile,
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
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from astropy.io import fits
    from gammapy.utils.fits import get_hdu

    logging.info('Reading {0}'.format(infile))
    hdu = get_hdu(infile)

    out_hdus = fits.HDUList()

    if make_coordinate_maps:
        from gammapy.image import coordinates
        logging.info('Computing LON and LAT maps')
        lon, lat = coordinates(hdu)
        out_hdus.append(fits.ImageHDU(lon, hdu.header, 'LON'))
        out_hdus.append(fits.ImageHDU(lat, hdu.header, 'LAT'))

    if make_distance_map:
        from gammapy.image import exclusion_distance
        logging.info('Computing DIST map')
        dist = exclusion_distance(hdu.data)
        out_hdus.append(fits.ImageHDU(dist, hdu.header, 'DIST'))

    logging.info('Writing {0}'.format(outfile))
    out_hdus.writeto(outfile, clobber=overwrite)
