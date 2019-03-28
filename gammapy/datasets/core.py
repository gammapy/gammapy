# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import sys
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from astropy.table import Table
import astropy.utils.data

__all__ = ["gammapy_data"]

log = logging.getLogger(__name__)

# This is the cross-platform way to get the HOME directory, also in Windows
# https://docs.python.org/3/library/pathlib.html#pathlib.Path.home
# http://stackoverflow.com/a/4028943
DATASET_DIR = Path.home() / ".gammapy/datasets"


def download_file(
    url, filename, overwrite=False, mkdir=True, show_progress=True, timeout=None
):
    """Download a URL to a given filename.

    This is a wrapper for the `astropy.utils.data.download_file` function,
    that allows moving the file to a given location if the download is successful.

    This function also creates directories as needed.

    Parameters
    ----------
    TODO
    """
    filename = Path(filename)

    if filename.is_file() and not overwrite:
        return

    if not filename.parent.is_dir() and mkdir:
        filename.parent.mkdir(parents=True)

    # This saves the file to a temp folder, with `cache=False` the Astropy cache isn't touched!
    temp_filename = astropy.utils.data.download_file(
        remote_url=url, cache=False, show_progress=show_progress, timeout=timeout
    )

    shutil.move(temp_filename, str(filename))

    return filename


class GammapyDataNotFoundError(OSError):
    """The gammapy-data is not available.

    You have to set the GAMMAPY_DATA environment variable so that it's found.
    """

    pass


class _GammapyData:
    """Access files from gammapy-data.

    You have to set the `GAMMAPY_DATA` environment variable
    so that it's found.
    """

    @property
    def is_available(self):
        """Is gammapy-data available?"""
        if "GAMMAPY_DATA" in os.environ:
            # Make sure this is really pointing to a gammapy-data folder
            filename = Path(os.environ["GAMMAPY_DATA"]) / "gamma-cat/gammacat.fits.gz"
            if filename.is_file():
                return True

        return False

    @property
    def dir(self):
        """Path to the gammapy-data repo.

        Raises `GammapyDataNotFoundError` if gammapy-data isn't found.
        """
        if self.is_available:
            return Path(os.environ["GAMMAPY_DATA"])
        else:
            msg = "The gammapy-data repo is not available. "
            msg += "You have to set the GAMMAPY_DATA environment variable "
            msg += "to point to the location for it to be found."
            raise GammapyDataNotFoundError(msg)

    def filename(self, filename):
        """Filename in gammapy-data as string.
        """
        return str(self.dir / filename)


gammapy_data = _GammapyData()
"""Module-level variable to access gammapy-data.

TODO: usage examples
"""
