# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
from __future__ import print_function, division

__all__ = ['argparse',
           'GammapyFormatter',
           ]

import argparse


class GammapyFormatter(argparse.ArgumentDefaultsHelpFormatter,
                       argparse.RawTextHelpFormatter):
    """ArgumentParser formatter_class argument.

    Examples
    --------
    >>> from gammapy.utils.scripts import argparse, GammapyFormatter
    >>> parser = argparse.ArgumentParser(description=__doc__,
    ...                                  formatter_class=GammapyFormatter)
    """
    pass
