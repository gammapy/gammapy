# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['xspec']


def main(args=None):
    parser = get_parser(xspec)
    args = parser.parse_args(args)
    xspec(**vars(args))


def xspec():
    """Perform various tasks with XSPEC files (PHA, ARF, RMF).

    Depending on the subcommand used, a variety of tasks
    are implemented for XSPEC files (PHA, ARF, RMF).

    * info : Print summary infos
    * plot : Make plots
    * fake : Fake some data or IRFs.

    TODO: describe
    """
    # TODO: implement
    print('error: this tool is not implemented')
