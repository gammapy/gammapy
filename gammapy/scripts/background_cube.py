# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['background_cube']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = background_cube.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
    parser.add_argument('run_list', type=str,
                        help='Input run list file name')
    parser.add_argument('exclusion_list', type=str,
                        help='Input exclusion list file name')
    parser.add_argument('reference_file', type=str,
                        help='Input FITS reference cube file name')
    parser.add_argument('out_file', type=str,
                        help='Output FITS counts cube file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    background_cube(**vars(args))


def background_cube(run_list,
                    exclusion_list,
                    reference_file,
                    out_file,
                    overwrite):
    """Create background model cube from off runs.

    TODO: explain a bit.
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    # TODO: implement
    raise NotImplementedError
