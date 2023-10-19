# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
from gammapy.scripts.download import RELEASE, cli_download_datasets
from gammapy.scripts.info import cli_info
from gammapy.version import version

log = logging.getLogger(__name__)


def check_tutorials_setup(download_datasets_path="./gammapy-data"):
    """Check tutorials setup and download data if not available.

    Parameters
    ----------
    download_datasets_path : str
        Path to download the data, if not present.
    """
    if "GAMMAPY_DATA" not in os.environ:
        log.info(
            "Missing example datasets, downloading to {download_datasets_path} now..."
        )
        cli_download_datasets.callback(out=download_datasets_path, release=RELEASE)
        os.environ["GAMMAPY_DATA"] = download_datasets_path

    cli_info.callback(system=True, version=True, dependencies=True, envvar=True)


def check_version(testversion):
    """Check if the current Gammapy version is identical, older or newer than the input.

    Parameters
    ----------
    testversion : str
        Version to test.

    Returns
    -------
    check : int
        0 if identical, -1 if the current version is older than `testversion` and +1 if newer.
    """

    if testversion == version:
        return 0
    elif "dev" in version:
        return 1
    elif float(testversion) < float(version):
        return 1
    else:
        return -1
