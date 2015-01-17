# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['irf_root_to_fits']


# TODO: rewrite and make it work for the latest CTA IRFs
# TODO: make this work for HESS IRFs


def main(args=None):
    parser = get_parser(irf_root_to_fits)
    parser.add_argument('irf_root_file', type=str,
                        help='IRF ROOT file.')
    parser.add_argument('-w', '--write_output', action='store_true', default=False,
                        help='Write results to FITS files in current directory.')
    args = parser.parse_args(args)
    irf_root_to_fits(**vars(args))


def irf_root_to_fits(irf_root_file,
                     write_output):
    """Convert CTA IRF data from ROOT to FITS format.

    Read input file from command line
    irf_root_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/irf/cta/SubarrayE_IFAE_50hours_20101102.root'
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from gammapy.utils.pyfact import cta_irf_root_to_fits

    cta_irf_root_to_fits(irf_root_file_name=irf_root_file,
                         write_output=write_output)
