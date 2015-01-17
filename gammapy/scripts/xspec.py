# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['xspec']


def main(args=None):
    from gammapy.utils.scripts import argparse, GammapyFormatter
    description = xspec.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
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
    raise NotImplementedError
