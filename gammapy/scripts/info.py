# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import platform
import sys
import warnings
import logging
import importlib
from collections import OrderedDict
import click
from .. import version

log = logging.getLogger(__name__)

GAMMAPY_DEPENDENCIES = [
    "numpy",
    "scipy",
    "matplotlib",
    "cython",
    "astropy",
    "astropy_healpix",
    "reproject",
    "sherpa",
    "pytest",
    "sphinx",
    "healpy",
    "regions",
    "iminuit",
    "naima",
    "uncertainties",
]

GAMMAPY_ENV_VARIABLES = [
    "GAMMAPY_EXTRA",
    "GAMMA_CAT",
    "GAMMAPY_FERMI_LAT_DATA",
    "CTADATA",
]


@click.command(name="info")
@click.option("--system/--no-system", default=True, help="Show system info")
@click.option("--version/--no-version", default=True, help="Show version info")
@click.option(
    "--dependencies/--no-dependencies", default=True, help="Show dependencies info"
)
@click.option("--envvar/--no-envvar", default=True, help="Show environment variables")
def cli_info(system, version, dependencies, envvar):
    """Display information about Gammapy
    """
    if system:
        info = get_info_system()
        print_info(info=info, title="System")

    if version:
        info = get_info_version()
        print_info(info=info, title="Gammapy package")

    if dependencies:
        info = get_info_dependencies()
        print_info(info=info, title="Other packages")

    if envvar:
        info = get_info_envvar()
        print_info(info=info, title="Gammapy environment variables")


def print_info(info, title):
    """Print Gammapy info."""
    info_all = "\n{}:\n\n".format(title)

    for key, value in info.items():
        info_all += "\t{:22s} : {:<10s} \n".format(key, value)

    print(info_all)


def get_info_system():
    """Get info about user system"""
    info = OrderedDict()
    info["python_executable"] = sys.executable
    info["python_version"] = platform.python_version()
    info["machine"] = platform.machine()
    info["system"] = platform.system()
    return info


def get_info_version():
    """Get detailed info about Gammapy version."""
    info = OrderedDict()
    try:
        path = sys.modules["gammapy"].__path__[0]
    except:
        path = "unknown"
    info["path"] = path
    info["version"] = version.version
    if not version.release:
        info["githash"] = version.githash
    return info


def get_info_dependencies():
    """Get info about Gammapy dependencies."""
    info = OrderedDict()
    for name in GAMMAPY_DEPENDENCIES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(name)

            module_version = getattr(module, "__version__", "no version info found")
        except ImportError:
            module_version = "not installed"
        info[name] = module_version
    return info


def get_info_envvar():
    """Get info about Gammapy environment variables."""
    info = OrderedDict()
    for name in GAMMAPY_ENV_VARIABLES:
        info[name] = os.environ.get(name, "not set")
    return info
