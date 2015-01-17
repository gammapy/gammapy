# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['detect']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = detect.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
    parser.add_argument('infile', type=str,
                        help='Input FITS file name')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    detect(**vars(args))


def detect(infile,
           overwrite):
    """Detect sources in images.

    TODO: explain.
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    # from gammapy import detect

    raise NotImplementedError
    # TODO: implement me
