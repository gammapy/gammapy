# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to create scripts and command-line tools."""

import ast
import codecs
import operator
import os.path
import functools
import types
import warnings
import numpy as np
from base64 import urlsafe_b64encode
from pathlib import Path
from uuid import uuid4
import yaml
from gammapy.utils.check import add_checksum, verify_checksum

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
    """Dictionary to yaml file.

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


def requires_module(module_name):
    """
    Decorator that conditionally enables a method or property based on the availability of a module.

    If the specified module is available, the decorated method or property is returned as-is.
    If the module is not available:
      - For methods: replaces the method with one that raises ImportError when called.
      - For properties: replaces the property with one that raises ImportError when accessed.

    Parameters
    ----------
    module_name : str
        The name of the module to check for.

    Returns
    -------
    function or property
        The original object if the module is available, otherwise a fallback.
    """

    def decorator(obj):
        try:
            __import__(module_name)
            return obj  # Module is available
        except ImportError:
            if isinstance(obj, property):
                return property(
                    lambda self: raise_import_error(module_name, is_property=True)
                )
            elif isinstance(obj, (types.FunctionType, types.MethodType)):

                @functools.wraps(obj)
                def wrapper(*args, **kwargs):
                    raise_import_error(module_name)

                return wrapper
            else:
                raise TypeError(
                    "requires_module can only be used on methods or properties."
                )

    return decorator


def raise_import_error(module_name, is_property=False):
    """
    Raises an ImportError with a descriptive message about a missing module.

    Parameters
    ----------
    module_name : str
        The name of the required module.
    is_property : bool
        Whether the error is for a property (affects the error message).
    """
    kind = "property" if is_property else "method"
    raise ImportError(f"The '{module_name}' module is required to use this {kind}.")


# Mapping of AST operators to NumPy functions
_OPERATORS = {
    ast.And: np.logical_and,
    ast.Or: np.logical_or,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: np.isin(a, b),
    ast.NotIn: lambda a, b: ~np.isin(a, b),
}


def logic_parser(table, expression):
    """
    Parse and apply a logical expression to filter rows from an Astropy Table.

    This function evaluates a logical expression on each row of the input table
    and returns a new table containing only the rows that satisfy the expression.
    The expression can reference any column in the table by name and supports
    logical operators (`and`, `or`), comparison operators (`<`, `>`, `==`, `!=`, `in`),
    lists, and constants.

    Parameters
    ----------
    table : `~astropy.table.Table`
        The input table to filter.
    expression : str
        The logical expression to evaluate on each row. The expression can reference
        any column in the table by name.

    Returns
    -------
    table : `~astropy.table.Table`
        A table view containing only the rows that satisfy the expression. If no rows
        match the condition, an empty table with the same column names and data types
        as the input table is returned.

    Examples
    --------
    Given a table with columns 'OBS_ID' and 'EVENT_TYPE':

    >>> from astropy.table import Table
    >>> data = {'OBS_ID': [1, 2, 3, 4], 'EVENT_TYPE': ['1', '3', '4', '2']}
    >>> table = Table(data)
    >>> expression = '(OBS_ID < 3) and (OBS_ID > 1) and ((EVENT_TYPE in ["3", "4"]) or (EVENT_TYPE == "3"))'
    >>> filtered_table = logic_parser(table, expression)
    >>> print(filtered_table)
    OBS_ID EVENT_TYPE
    ------ ----------
         2          3

    """

    def eval_node(node):
        if isinstance(node, ast.BoolOp):
            op_func = _OPERATORS[type(node.op)]
            values = [eval_node(v) for v in node.values]
            result = values[0]
            for v in values[1:]:
                result = op_func(result, v)
            return result
        elif isinstance(node, ast.Compare):
            left = eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = eval_node(comparator)
                op_func = _OPERATORS[type(op)]
                left = op_func(left, right)
            return left
        elif isinstance(node, ast.Name):
            if node.id not in table.colnames:
                raise KeyError(
                    f"Column '{node.id}' not found in the table. Available columns: {table.colnames}"
                )
            return table[node.id]
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [eval_node(elt) for elt in node.elts]
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    expr_ast = ast.parse(expression, mode="eval")
    mask = eval_node(expr_ast.body)
    return table[mask]
