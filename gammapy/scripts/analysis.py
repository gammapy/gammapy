# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import jsonschema
import logging
from pathlib import Path
from ..utils.scripts import read_yaml

__all__ = ["Analysis"]

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config"


class Analysis:
    """High-level interface analysis session class."""

    def __init__(self, configfile=CONFIG_PATH/"default.yaml"):
        self.schemafile = CONFIG_PATH / "schema.yaml"
        self.configfile = configfile
        self.params = read_yaml(self.configfile)

        self.validate_schema()
        self.set_log()

    def set_log(self):
        """Set logging parameters for API."""

        cfg = self.params['global']
        if "logging" in cfg:
            log_params = dict()
            for par, val in cfg['logging'].items():
                log_params[par] = val
            log_params['level'] = cfg['logging']['level'].upper()
            logging.basicConfig(**log_params)
            log.info("Setting logging parameters")

    def validate_schema(self):
        """Validate config parameters against schema."""
        schema = read_yaml(self.schemafile)

        try:
            jsonschema.validate(self.params, schema)

        # make error more specific based on param/value
        except jsonschema.exceptions.ValidationError as ex:
            log.error('Error when validating configuration parameters against schema.')
            log.error(ex.message)
            raise ex
