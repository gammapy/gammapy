# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from collections import OrderedDict
import importlib
import os
import glob
import logging
from os.path import expandvars
from ..extern.pathlib import Path

__all__ = [
    'get_installed_scripts',
    'get_all_main_functions',
    'set_up_logging_from_args',
    'read_yaml',
    'write_yaml',
    'make_path',
    'recursive_merge_dicts',
]


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

    TODO: this is brittle ... find a better solution to collect the scripts.
    """
    # Could this work?
    # http://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package
    # import pkgutil
    # pkgutil.iter_modules(path=None, prefix='')

    # TODO: use Path here
    path = os.path.join(os.path.dirname(__file__), '../scripts')
    names = glob.glob1(path, '*.py')
    names = [_.replace('.py', '') for _ in names]
    for name in ['__init__', 'setup_package']:
        names.remove(name)

    # names += ['data_browser']

    out = OrderedDict()
    for name in names:
        module = importlib.import_module('gammapy.scripts.{}'.format(name))
        if hasattr(module, 'main'):
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


def read_yaml(filename, logger=None):
    """
    Read YAML file

    Parameters
    ----------
    filename : `~gammapy.extern.pathlib.Path`, str
        File to read
    """
    import yaml

    filename = make_path(filename)
    if logger is not None:
        logger.info('Reading {}'.format(filename))
    with open(str(filename)) as fh:
        dictionary = yaml.safe_load(fh)

    return dictionary


def write_yaml(dictionary, filename, logger=None):
    """Write YAML file.

    Parameters
    ----------
    dictionary : dict
        Python dictionary
    filename : str, `~gammapy.exter.pathlib.Path`
        file to write
    """
    import yaml
    filename = make_path(filename)
    filename.parent.mkdir(exist_ok=True)
    if logger is not None:
        logger.info('Writing {}'.format(filename))
    with open(str(filename), 'w') as outfile:
        outfile.write(yaml.safe_dump(dictionary, default_flow_style=False))


def make_path(path):
    """Expand environment variables on `~pathlib.Path` construction.

    Parameters
    ----------
    path : str, `~gammapy.extern.pathlib.Path`
        path to expand
    """
    # TODO: raise error or warning if environment variables that don't resolve are used
    # e.g. "spam/$DAMN/ham" where `$DAMN` is not defined
    # Otherwise this can result in cryptic errors later on
    return Path(expandvars(str(path)))


def recursive_merge_dicts(a, b):
    """Recursively merge two dictionaries.

    Entries in b override entries in a. The built-in update function cannot be
    used for hierarchical dicts, see:
    http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/3233356#3233356

    Parameters
    ----------
    a : dict
        dictionary to be merged
    b : dict
        dictionary to be merged

    Returns
    -------
    c : dict
        merged dict

    Examples
    --------
    >>> from gammapy.utils.scripts import recursive_merge_dicts
    >>> a = dict(a=42, b=dict(c=43, e=44))
    >>> b = dict(d=99, b=dict(c=50, g=98))
    >>> c = recursive_merge_dicts(a, b)
    >>> print(c)
    {'a': 42, 'b': {'c': 50, 'e': 44, 'g': 98}, 'd': 99}
    """
    c = a.copy()
    for k, v in b.items():
        if k in c and isinstance(c[k], dict):
            c[k] = recursive_merge_dicts(c[k], v)
        else:
            c[k] = v
    return c
