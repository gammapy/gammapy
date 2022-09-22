import os
import logging
from gammapy.scripts.info import cli_info
from gammapy.scripts.download import cli_download_datasets

log = logging.getLogger(__name__)


def check_tutorials_setup(download_datasets_path="./gammapy-data"):
    """Check tutorials setup and download data if not available
    
    Parameters
    ----------
    download_datasets_path : str
        Path to download the data, if not present.
    """
    if not "GAMMAPY_DATA" in os.environ:
        log.info(
            "Missing example datasets, downloading to {download_datasets_path} now..."
        )
        cli_download_datasets.callback(out=download_datasets_path)
        os.env["GAMMAPY_DATA"] = download_datasets_path

    cli_info.callback(system=True, version=True, dependencies=True, envvar=True)