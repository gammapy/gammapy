# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to create scripts and command-line tools."""
import codecs
import os.path
import warnings
from base64 import urlsafe_b64encode
from pathlib import Path
from uuid import uuid4
import yaml
from gammapy.utils.check import add_checksum, verify_checksum
from gammapy.utils.deprecation import deprecated_renamed_argument

__all__ = [
    "from_yaml",
    "get_images_paths",
    "make_path",
    "read_yaml",
    "recursive_merge_dicts",
    "to_yaml",
    "write_yaml",
]

PATH_DOCS = Path(__file__).resolve().parent / ".." / ".." / "docs"
SKIP = ["_static", "_build", "_checkpoints", "docs/user-guide/model-gallery/"]
YAML_FORMAT = dict(sort_keys=False, indent=4, width=80, default_flow_style=False)


def get_images_paths(folder=PATH_DOCS):
    """Generator yields a Path for each image used in notebook.

    Parameters
    ----------
    folder : str
        Folder where to search.
    """
    for i in Path(folder).rglob("images/*"):
        if not any(s in str(i) for s in SKIP):
            yield i.resolve()


def from_yaml(text, sort_keys=False, checksum=False):
    """Read YAML file.

    Parameters
    ----------
    text : str
        yaml str
    sort_keys : bool, optional
        Whether to sort keys. Default is False.
    checksum : bool
        Whether to perform checksum verification. Default is False.

    Returns
    -------
    data : dict
        YAML file content as a dictionary.

    """
    data = yaml.safe_load(text)
    checksum_str = data.pop("checksum", None)
    if checksum:
        yaml_format = YAML_FORMAT.copy()
        yaml_format["sort_keys"] = sort_keys
        yaml_str = yaml.dump(data, **yaml_format)
        if not verify_checksum(yaml_str, checksum_str):
            warnings.warn("Checksum verification failed.", UserWarning)

    return data


def read_yaml(filename, logger=None, checksum=False):
    """Read YAML file.

    Parameters
    ----------
    filename : `~pathlib.Path`
        Filename.
    logger : `~logging.Logger`
        Logger.
    checksum : bool
        Whether to perform checksum verification. Default is False.

    Returns
    -------
    data : dict
        YAML file content as a dictionary.
    """
    path = make_path(filename)
    if logger is not None:
        logger.info(f"Reading {path}")

    text = path.read_text()
    return from_yaml(text, checksum=checksum)


def to_yaml(dictionary, sort_keys=False):
    """dict to yaml

    Parameters
    ----------
    dictionary : dict
        Python dictionary.
    sort_keys : bool, optional
        Whether to sort keys. Default is False.
    """
    from gammapy.utils.metadata import CreatorMetaData

    yaml_format = YAML_FORMAT.copy()
    yaml_format["sort_keys"] = sort_keys
    text = yaml.safe_dump(dictionary, **yaml_format)
    creation = CreatorMetaData()
    return text + creation.to_yaml()


@deprecated_renamed_argument("dictionary", "text", "v1.3")
def write_yaml(
    text, filename, logger=None, sort_keys=False, checksum=False, overwrite=False
):
    """Write YAML file.

    Parameters
    ----------
    text : str
        yaml str
    filename : `~pathlib.Path`
        Filename.
    logger : `~logging.Logger`, optional
        Logger. Default is None.
    sort_keys : bool, optional
        Whether to sort keys. Default is True.
    checksum : bool, optional
        Whether to add checksum keyword. Default is False.
    overwrite : bool, optional
        Overwrite existing file. Default is False.
    """

    if checksum:
        text = add_checksum(text, sort_keys=sort_keys)

    path = make_path(filename)
    path.parent.mkdir(exist_ok=True)
    if path.exists() and not overwrite:
        raise IOError(f"File exists already: {path}")
    if logger is not None:
        logger.info(f"Writing {path}")
    path.write_text(text)


def make_name(name=None):
    """Make a dataset name."""
    if name is None:
        name = urlsafe_b64encode(codecs.decode(uuid4().hex, "hex")).decode()[:8]
        while name[0] == "_":
            name = urlsafe_b64encode(codecs.decode(uuid4().hex, "hex")).decode()[:8]

    if not isinstance(name, str):
        raise ValueError(
            "Name argument must be a string, "
            f"got '{name}', which is of type '{type(name)}'"
        )

    return name


def make_path(path):
    """Expand environment variables on `~pathlib.Path` construction.

    Parameters
    ----------
    path : str, `pathlib.Path`
        Path to expand.
    """
    # TODO: raise error or warning if environment variables that don't resolve are used
    # e.g. "spam/$DAMN/ham" where `$DAMN` is not defined
    # Otherwise this can result in cryptic errors later on
    if path is None:
        return None
    else:
        return Path(os.path.expandvars(path))


def recursive_merge_dicts(a, b):
    """Recursively merge two dictionaries.

    Entries in 'b' override entries in 'a'. The built-in update function cannot be
    used for hierarchical dicts, see:
    http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth/3233356#3233356

    Parameters
    ----------
    a : dict
        Dictionary to be merged.
    b : dict
        Dictionary to be merged.

    Returns
    -------
    c : dict
        Merged dictionary.

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
