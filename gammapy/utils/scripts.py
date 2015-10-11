# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import argparse
from collections import OrderedDict
import importlib
import os
import glob
import logging
import shutil

__all__ = [
    'GammapyFormatter',
    'get_parser',
    'get_installed_scripts',
    'get_all_main_functions',
    'set_up_logging_from_args',
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
    for name in ['__init__']:
        names.remove(name)

    out = OrderedDict()
    for name in names:
        module = importlib.import_module('gammapy.scripts.{}'.format(name))
        out[name] = module.main

    return out


def set_up_logging_from_args(args):
    """Set up logging from command line arguments.

    This is a helper function that should be called from
    all Gammapy command line tools.
    It executes the boilerplate that's involved in setting
    up the root logger the way we like it.
    """
    if hasattr(args, 'loglevel'):
        level = args.loglevel
        del args.loglevel
    else:
        level = 'info'
    _configure_root_logger(level=level)


def _configure_root_logger(level='info', format=None):
    """Configure root log level and format.

    This is a helper function that can be called form
    """
    log = logging.getLogger()  # Get root logger

    # Set log level
    # level = getattr(logging, level.upper())
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(level))
    log.setLevel(level=numeric_level)

    # Format log handler
    if not format:
        # format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        format = '%(levelname)-8s %(message)s [%(name)s]'
    formatter = logging.Formatter(format)

    # Not sure why there sometimes is a handler attached to the root logger,
    # and sometimes not, i.e. why this is needed:
    # https://github.com/gammapy/gammapy/pull/318/files#r36453321
    if len(log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(numeric_level)
        log.addHandler(handler)

    log.handlers[0].setFormatter(formatter)

    return log

def _create_dir(dirname, overwrite=False):
    """Create directory in file system.

    This is usefull for instance for creating output directories.

    Parameters
    ----------
    dirname : str
        Directory name to create. It can be full or relative path.
    overwrite : bool, optional
        Flag to remove previous directory with the same name (if existing).
    """
    if dirname == '':
            s_error = "Cannot continue: directory name \'{}\' empty.".format(dirname)
            raise ValueError(s_error)

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        if overwrite:
            # delete and create again
            shutil.rmtree(dirname) # recursively
            os.mkdir(dirname)
        else:
            # do not overwrite, hence exit
            s_error = "Cannot continue: directory \'{}\' exists.".format(dirname)
            raise RuntimeError(s_error)
