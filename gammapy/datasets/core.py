# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import sys
import os
import shutil
from collections import OrderedDict
from astropy.table import Table
import astropy.utils.data
from ..extern.pathlib import Path

__all__ = ["Datasets", "gammapy_extra"]

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


def make_dataset(config):
    """Dataset factory function.
    """
    # For not we just have simple datasets
    name = config["name"]
    filename = DATASET_DIR / config["filename"]
    url = config.get("url")
    description = config.get("description")
    tags = config.get("tags")
    ds = OneFileDataset(
        name=name, filename=filename, url=url, description=description, tags=tags
    )
    return ds


class OneFileDataset(object):
    """One file simple dataset."""

    def __init__(self, name, filename, url=None, description=None, tags=None):
        self.name = name
        self.filename = filename
        self.url = url
        self.description = description
        self.tags = tags

    def fetch(self, overwrite=False):
        download_file(url=self.url, filename=self.filename, overwrite=overwrite)

    def is_available(self):
        return Path(self.filename).is_file()

    def info(self, file=None):
        if not file:
            file = sys.stdout

        print(self.__dict__, file=file)
        self._print_status(file=file)

    def _print_status(self, file):
        available = "yes" if self.is_available() else "no"
        print("Available: {}".format(available), file=file)


class Datasets(object):
    """Download and access for all built-in datasets.

    TODO: this isn't used much at the moment and not documented.
    I added this before I decided to add `gammapy_extra`,
    and then this class wasn't needed to access datasets for tests.

    We still need something like this to manage files that aren't
    in gammapy-extra, e.g. large files from the web that we don't
    want to stick in gammapy-extra.

    Parameters
    ----------
    config : `~collections.OrderedDict`
        Data manager configuration.

    Attributes
    ----------
    datasets : list of `Dataset` objects
        List of datasets
    """

    # DEFAULT_CONFIG_FILE = Path.home() / '.gammapy/data-register.yaml'
    DEFAULT_CONFIG_FILE = astropy.utils.data.get_pkg_data_filename("datasets.yaml")

    def __init__(self, config=None):
        if not config:
            filename = Datasets.DEFAULT_CONFIG_FILE
            config = Datasets._load_config(filename)

        self.config = config

        self.datasets = OrderedDict()
        for dataset_config in config:
            dataset = make_dataset(dataset_config)
            self.datasets[dataset.name] = dataset

    @classmethod
    def from_yaml(cls, filename):
        """Create from a YAML config file.

        Parameters
        ----------
        filename : str
            YAML config file
        """
        config = Datasets._load_config(filename)
        return cls(config=config)

    @staticmethod
    def _load_config(filename):
        import yaml

        with Path(filename).open() as fh:
            config = yaml.safe_load(fh)
        return config

    def info(self, verbose=False, file=None):
        """Print basic info."""
        if not file:
            file = sys.stdout

        print("Number of datasets: {}".format(len(self.datasets)), file=file)

        self.info_table.pprint()

        if verbose:
            for dataset in self.datasets.values():
                dataset.info(file=file)

    @property
    def info_table(self):
        rows = []
        for ds in self.datasets.values():
            row = dict()
            row["Name"] = ds.name
            row["Available"] = "yes" if ds.is_available() else "no"
            row["Filename"] = ds.filename
            rows.append(row)

        table = Table(rows=rows, names=["Name", "Available", "Filename"])
        return table

    def __getitem__(self, name):
        return self.datasets[name]

    def fetch_one(self, name):
        """Fetch one dataset.
        """
        dataset = self.datasets[name]
        dataset.fetch()

    def fetch_all(self, tags="catalog"):
        """Fetch all datasets that match one of the tags.
        """
        for dataset in self.datasets.values():
            if not dataset.tags:
                continue
            if set(dataset.tags) & set(tags):
                dataset.fetch()


class GammapyExtraNotFoundError(OSError):
    """The gammapy-extra repo is not available.

    You have to set the GAMMAPY_EXTRA environment variable so that it's found.
    """

    pass


class _GammapyExtra(object):
    """Access files from gammapy-extra repo.

    You have to set the `GAMMAPY_EXTRA` environment variable
    so that it's found.
    """

    @property
    def is_available(self):
        """Is the gammapy-extra repo available?"""
        if "GAMMAPY_EXTRA" in os.environ:
            # Make sure this is really pointing to a gammapy-extra folder
            filename = Path(os.environ["GAMMAPY_EXTRA"]) / "logo/gammapy_logo.pdf"
            if filename.is_file():
                return True

        return False

    @property
    def dir(self):
        """Path to the gammapy-extra repo.

        Raises `GammapyExtraNotFoundError` if gammapy-extra isn't found.
        """
        if self.is_available:
            return Path(os.environ["GAMMAPY_EXTRA"])
        else:
            msg = "The gammapy-extra repo is not available. "
            msg += "You have to set the GAMMAPY_EXTRA environment variable "
            msg += "to point to the location for it to be found."
            raise GammapyExtraNotFoundError(msg)

    def filename(self, filename):
        """Filename in gammapy-extra as string.
        """
        return str(self.dir / filename)


gammapy_extra = _GammapyExtra()
"""Module-level variable to access gammapy-extra.

TODO: usage examples
"""
