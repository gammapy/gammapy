# Licensed under a 3-clause BSD style license - see LICENSE.rst
import importlib
import logging
import os
import platform
import sys
import warnings
import click
from gammapy import __version__

log = logging.getLogger(__name__)

# Should be in sync with `docs/install/dependencies.rst`
GAMMAPY_DEPENDENCIES = [
    # required
    "numpy",
    "scipy",
    "astropy",
    "regions",
    "click",
    "yaml",
    # "pydantic",  # has no __version__
    # optional
    "IPython",
    # "jupyter",   # has no __version__
    "jupyterlab",
    "matplotlib",
    "pandas",
    "healpy",
    "iminuit",
    "sherpa",
    "naima",
    "emcee",
    "corner",
]

GAMMAPY_ENV_VARIABLES = ["GAMMAPY_DATA"]


@click.command(name="info")
@click.option("--system/--no-system", default=True, help="Show system info")
@click.option("--version/--no-version", default=True, help="Show version")
@click.option(
    "--dependencies/--no-dependencies", default=True, help="Show dependencies"
)
@click.option("--envvar/--no-envvar", default=True, help="Show environment variables")
def cli_info(system, version, dependencies, envvar):
    """Display information about Gammapy"""
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
    info_all = f"\n{title}:\n\n"

    for key, value in info.items():
        info_all += f"\t{key:22s} : {value:<10s} \n"

    print(info_all)


def get_info_system():
    """Get info about user system"""
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def get_info_version():
    """Get detailed info about Gammapy version."""
    info = {"version": __version__}
    try:
        path = sys.modules["gammapy"].__path__[0]
    except Exception:
        path = "unknown"
    info["path"] = path

    return info


def get_info_dependencies():
    """Get info about Gammapy dependencies."""
    info = {}
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
    return {name: os.environ.get(name, "not set") for name in GAMMAPY_ENV_VARIABLES}
