# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser

__all__ = []

log = logging.getLogger(__name__)

DEFAULT_DATADIR = '/Users/deil/work/_Data/hess/HESSFITS/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std/'



def main(args=None):
    parser = get_parser()
    parser.add_argument('--data_dir', default=DEFAULT_DATADIR,
                        help='Data directory')
    args = parser.parse_args(args)
    print('not implemented')
