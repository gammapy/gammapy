# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
from collections import OrderedDict
import importlib
import os
import glob

__all__ = ['GammapyFormatter',
           'get_parser',
           'get_installed_scripts',
           'get_all_main_functions',
           ]



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


def get_parser(function=None, description='N/A'):
    """Make an ArgumentParser how we like it.
    """
    if function:
        description = function.__doc__.split('\n')[0]
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=GammapyFormatter)
    return parser


def get_installed_scripts():
    """Get list of installed scripts via ``pkg-resources``.

    See http://peak.telecommunity.com/DevCenter/PkgResources#convenience-api

    TODO: not sure if this will be useful ... maybe to check if the list
    of installed packages matches the available scripts somehow?
    """
    from pkg_resources import get_entry_map
    console_scripts = get_entry_map('gammapy')['console_scripts']
    return console_scripts


def get_all_main_functions():
    """Get a dict with all scripts (used for testing).
    """
    # Could this work?
    # http://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package
    # import pkgutil
    # pkgutil.iter_modules(path=None, prefix='')

    path = os.path.join(os.path.dirname(__file__), '../scripts')
    names = glob.glob1(path, '*.py')
    names = [_.replace('.py', '') for _ in names]
    print(names)
    print(path)
    for name in ['__init__']:
        names.remove(name)

    out = OrderedDict()
    for name in names:
        module = importlib.import_module('gammapy.scripts.{}'.format(name))
        out[name] = module.main
    return out