# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""
import codecs
import os.path
from base64 import urlsafe_b64encode
from pathlib import Path
from uuid import uuid4
import yaml

__all__ = ["read_yaml", "write_yaml", "make_path", "recursive_merge_dicts"]


def read_yaml(filename, logger=None):
    """Read YAML file.

    Parameters
    ----------
    filename : `~pathlib.Path`
        Filename
    logger : `~logging.Logger`
        Logger

    Returns
    -------
    data : dict
        YAML file content as a dict
    """
    path = make_path(filename)
    if logger is not None:
        logger.info(f"Reading {path}")

    text = path.read_text()
    return yaml.safe_load(text)


def write_yaml(dictionary, filename, logger=None, sort_keys=True):
    """Write YAML file.

    Parameters
    ----------
    dictionary : dict
        Python dictionary
    filename : `~pathlib.Path`
        Filename
    logger : `~logging.Logger`
        Logger
    sort_keys : bool
        Whether to sort keys.
    """
    text = yaml.safe_dump(dictionary, default_flow_style=False, sort_keys=sort_keys)

    path = make_path(filename)
    path.parent.mkdir(exist_ok=True)
    if logger is not None:
        logger.info(f"Writing {path}")
    path.write_text(text)


def make_name(name=None):
    if name is None:
        return urlsafe_b64encode(codecs.decode(uuid4().hex, "hex")).decode()[:8]
    else:
        return name


def make_path(path):
    """Expand environment variables on `~pathlib.Path` construction.

    Parameters
    ----------
    path : str, `pathlib.Path`
        path to expand
    """
    # TODO: raise error or warning if environment variables that don't resolve are used
    # e.g. "spam/$DAMN/ham" where `$DAMN` is not defined
    # Otherwise this can result in cryptic errors later on
    return Path(os.path.expandvars(path))


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
