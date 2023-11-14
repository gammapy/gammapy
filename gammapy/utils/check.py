# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
from gammapy.scripts.download import RELEASE, cli_download_datasets
from gammapy.scripts.info import cli_info

log = logging.getLogger(__name__)


def check_tutorials_setup(download_datasets_path="./gammapy-data"):
    """Check tutorials setup and download data if not available.

    Parameters
    ----------
    download_datasets_path : str, optional
        Path to download the data. Default is "./gammapy-data".
    """
    if "GAMMAPY_DATA" not in os.environ:
        log.info(
            "Missing example datasets, downloading to {download_datasets_path} now..."
        )
        cli_download_datasets.callback(out=download_datasets_path, release=RELEASE)
        os.environ["GAMMAPY_DATA"] = download_datasets_path

    cli_info.callback(system=True, version=True, dependencies=True, envvar=True)


def yaml_checksum(yaml_content):
    """Compute a MD5 checksum for a given yaml string input."""
    import hashlib

    # Calculate the MD5 checksum
    checksum = hashlib.md5(yaml_content.encode("utf-8")).hexdigest()

    return checksum


def verify_checksum(yaml_content, checksum):
    """Compare MD5 checksum for yaml_content with input checksum."""
    return yaml_checksum(yaml_content) == checksum
