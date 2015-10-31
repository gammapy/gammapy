# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.io import fits
from astropy.table import Table
from ..image.utils import bin_events_in_cube
from ..utils.scripts import get_parser

__all__ = ['cube_bin']

log = logging.getLogger(__name__)


def cube_bin_main(args=None):
    parser = get_parser(cube_bin)
    parser.add_argument('event_file', type=str,
                        help='Input FITS event file name')
    parser.add_argument('reference_file', type=str,
                        help='Input FITS reference cube file name')
    parser.add_argument('out_file', type=str,
                        help='Output FITS counts cube file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    cube_bin(**vars(args))


def cube_bin(event_file,
             reference_file,
             out_file,
             overwrite):
    """Bin events into a LON-LAT-Energy cube."""
    events = Table.read(event_file)
    reference_cube = fits.open(reference_file)
    energies = Table.read(reference_file, 'ENERGIES')
    out_cube = bin_events_in_cube(events, reference_cube, energies)
    out_cube.writeto(out_file, clobber=overwrite)
