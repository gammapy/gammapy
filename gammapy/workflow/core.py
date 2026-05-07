# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API."""

import html
import logging
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel, Models
from gammapy.utils.scripts import make_path
from gammapy.workflow.config import WorkflowConfig
from .steps import WorkflowStep

__all__ = ["Workflow"]

log = logging.getLogger(__name__)


class Workflow:
    """Config-driven high level workflow interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `~gammapy.workflow.WorkflowConfig`
        Configuration options following `WorkflowConfig` schema.
    """

    def __init__(self, config):
        self.log = log
        self.config = config
        self.config.set_logging()
        self.datastore = None
        self.observations = None
        self.datasets = None
        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None
        self.steps = []

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

    @property
    def config(self):
        """Workflow configuration as an `~gammapy.workflow.WorkflowConfig` object."""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = WorkflowConfig(**value)
        elif isinstance(value, WorkflowConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or WorkflowConfig.")

    def run(self, steps=None, overwrite=None, **kwargs):
        if steps is None:
            steps = self.config.general.steps
            overwrite = self.config.general.overwrite
        else:
            if overwrite is None:
                overwrite = [True] * len(steps)

        workflow_steps = []
        for k, step in enumerate(steps):
            if isinstance(overwrite, list):
                overwrite_step = overwrite[k]
            else:
                overwrite_step = overwrite
            workflow_step = WorkflowStep.create(
                step, self.config, log=self.log, overwrite=overwrite_step, **kwargs
            )
            workflow_step.run(self)
            workflow_steps.append(workflow_step)

        self.steps = workflow_steps

    # keep these methods to be backward compatible
    def get_observations(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        self.run(steps=["observations"])

    def get_datasets(self):
        """Produce reduced datasets."""
        self.run(steps=["datasets"])

    def run_fit(self):
        """Fitting reduced datasets to model."""
        self.run(steps=["fit"])

    def get_flux_points(self):
        """Calculate flux points for a specific model component."""
        self.run(steps=["flux-points"])

    def get_excess_map(self):
        """Calculate excess map with respect to the current model."""
        self.run(steps=["excess-map"])

    def get_light_curve(self):
        """Calculate light curve for a specific model component."""
        self.run(steps=["light-curve"])

    def set_models(self, models, extend=True):
        """Set models on datasets.

        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string.
        extend : bool, optional
            Extend the exiting models on the datasets or replace them.
            Default is True.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        self.log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models

        bkg_models = []
        for dataset in self.datasets:
            if dataset.tag == "MapDataset" and dataset.background_model is None:
                bkg_models.append(FoVBackgroundModel(dataset_name=dataset.name))
        if bkg_models:
            models.extend(bkg_models)
            self.datasets.models = models

        self.log.info(models)

    def read_models(self, path, extend=True):
        """Read models from YAML file.

        Parameters
        ----------
        path : str
            Path to the model file.
        extend : bool, optional
            Extend the exiting models on the datasets or replace them.
            Default is True.
        """
        path = make_path(path)
        models = Models.read(path)
        self.set_models(models, extend=extend)
        self.log.info(f"Models loaded from {path}.")

    def write_models(self, overwrite=True, write_covariance=True):
        """Write models to YAML file.

        File name is taken from the configuration file.
        """
        filename_models = self.config.general.models_file
        if filename_models is not None:
            self.models.write(
                filename_models, overwrite=overwrite, write_covariance=write_covariance
            )
            log.info(f"Models loaded from {filename_models}.")
        else:
            raise RuntimeError("Missing models_file in config.general")

    def read_datasets(self):
        """Read datasets from YAML file.

        File names are taken from the configuration file.
        """
        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets = Datasets.read(filename)
            log.info(f"Datasets loaded from {filename}.")
            if filename_models is not None:
                self.read_models(filename_models, extend=False)
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def write_datasets(self, overwrite=True, write_covariance=True):
        """Write datasets to YAML file.

        File names are taken from the configuration file.

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite existing file. Default is True.
        write_covariance : bool, optional
            Save covariance or not. Default is True.
        """
        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets.write(
                filename,
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
            )
            log.info(f"Datasets stored to {filename}.")
            log.info(f"Datasets stored to {filename_models}.")
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def update_config(self, config):
        """Update the configuration."""
        self.config = self.config.update(config=config)
