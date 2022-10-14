import logging
import os
from gammapy.scripts.download import RELEASE, cli_download_datasets
from gammapy.scripts.info import cli_info

log = logging.getLogger(__name__)


def check_tutorials_setup(download_datasets_path="./gammapy-data"):
    """Check tutorials setup and download data if not available

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
        os.env["GAMMAPY_DATA"] = download_datasets_path

    cli_info.callback(system=True, version=True, dependencies=True, envvar=True)
