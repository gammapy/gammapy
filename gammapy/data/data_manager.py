# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import sys
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from .data_store import DataStore

__all__ = [
    'DataManager',
]

log = logging.getLogger(__name__)


class DataManager(object):
    """Data manager.

    The data manager helps to locate, download, index, validate
    and generally manage data.

    The `DataManager` object just stores the configuration in a
    ``config`` attribute and that configuration can be read from
    and written to YAML files.

    TODO: link to a getting started section in the docs.

    Parameters
    ----------
    config : `~collections.OrderedDict`
        Data manager configuration.
    """
    DEFAULT_CONFIG_FILE = Path.home() / '.gammapy/data-register.yaml'

    def __init__(self, config=None):
        if not config:
            filename = DataManager.DEFAULT_CONFIG_FILE
            config = DataManager._load_config(filename)

        self.config = config

    @classmethod
    def from_yaml(cls, filename):
        """Create from a YAML config file.

        Parameters
        ----------
        filename : str
            YAML config file
        """
        filename = make_path(filename)
        config = DataManager._load_config(str(filename))
        return cls(config=config)

    @staticmethod
    def _load_config(filename):
        import yaml
        with Path(filename).open() as fh:
            config = yaml.safe_load(fh)
        return config

    def __getitem__(self, name):
        store_config = self.store_config(name)
        return DataStore.from_config(store_config)

    def store_config(self, name):
        for store_config in self.config['stores']:
            if store_config['name'] == name:
                return store_config

        msg = 'No data store with name {} found. '.format(name)
        msg += 'Available datastores: {}'.format(self.store_names)
        raise KeyError(msg)

    @property
    def store_names(self):
        """Data store names"""
        return [_['name'] for _ in self.config['stores']]

    @property
    def stores(self):
        """List of data stores."""
        stores = []
        for store_config in self.config['stores']:
            store = DataStore.from_config(store_config)
            stores.append(store)
        return stores

    def info(self, stream=None):
        """Print some info."""
        import yaml

        if not stream:
            stream = sys.stdout

        print(file=stream)
        print('*** DataManager info ***', file=stream)
        print('Number of data stores: {}'.format(len(self.config['stores'])), file=stream)
        print('Data store names: {}'.format(self.store_names), file=stream)
        yaml.safe_dump(self.config['stores'], indent=2, stream=stream)
        print('', file=stream)

        # TODO: implement checks (i.e. whether all index files or even all data files are present and valid)
        # def check_stores(self):
        #     ok = True
        #     for store_config in self.config['stores']:


def update_data():
    """Update data from server (using rsync)"""
    raise NotImplementedError
    # TODO: extract this from the `data-register.yaml` config file:
    cmd = ('rsync -uvrl {username}@{server}:{server_path} {local_path}'
           ''.format(locals()))
