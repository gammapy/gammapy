# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

# TODO: implement object listing and selecting by key names in ROOT file
# TODO: implement histogram conversion
# TODO: implement ntuple conversion

__all__ = ['root_to_fits']


def main(args=None):
    parser = get_parser(root_to_fits)
    args = parser.parse_args(args)
    root_to_fits(**vars(args))


def root_to_fits():
    """Convert ROOT files to FITS files (histograms and ntuples).

    TODO: explain a bit.
    """
    # TODO: implement
    raise NotImplementedError
