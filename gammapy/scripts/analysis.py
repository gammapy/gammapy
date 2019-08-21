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

        try:
            self.params = read_yaml(self.configfile)
            self._validate_schema()
        except Exception as ex:
            log.error(ex)

        self._set_log()

    def _set_log(self):
        """Set log level for API."""

        cfg = self.params['config']
        if "logging" in cfg:
            log_params = dict()
            for par, val in cfg['logging'].items():
                log_params[par] = val
            log_params['level'] = cfg['logging']['level'].upper()
            logging.basicConfig(**log_params)
            log.info("Setting log parameters")

    def _validate_schema(self):
        """Validate config parameters against schema."""
        schema = read_yaml(self.schemafile)

        try:
            jsonschema.validate(self.params, schema)
        except jsonschema.exceptions.ValidationError as ex:
            log.error('Invalid input file: {}'.format(self.configfile))
            raise ex
