import pytest
from numpy.testing import assert_allclose
from gammapy.modeling import Parameter
from gammapy.modeling.models import ModelBase, Models
from gammapy.datasets import Dataset
from gammapy.modeling.sampler import Sampler
from gammapy.modeling.models import (
    GaussianPrior,
    UniformPrior,
    LogUniformPrior,
)


class MyModel(ModelBase):
    x = Parameter("x", 2)
    y = Parameter("y", 3e2)
    z = Parameter("z", 4e-2)
    name = "test"
    datasets_names = ["test"]
    type = "model"


class MyDataset(Dataset):
    tag = "MyDataset"

    def __init__(self, name="test"):
        self._name = name
        self._models = Models([MyModel(x=1.99, y=2.99e3, z=3.99e-2)])
        self.data_shape = (1,)
        self.mask_fit = None
        self.mask_safe = None

    @property
    def models(self):
        return self._models

    def stat_sum(self):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in self.models.parameters.unique_parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2

    def stat_array(self):
        """Statistic array, one value per data point."""
        return self.stat_sum()


@pytest.mark.parametrize("backend", ["ultranest"])
def test_run(backend):
    dataset = MyDataset()

    dataset.models.parameters["x"].prior = GaussianPrior(mu=2, sigma=1)
    dataset.models.parameters["y"].prior = UniformPrior(min=290, max=310)
    dataset.models.parameters["z"].prior = LogUniformPrior(min=1e-2, max=1e-1)

    sampler_opts = {
        "live_points": 200,
        "log_dir": None,
        "resume": "overwrite",
        "frac_remain": 0.5,
        "step_sampler": False,
    }
    result = Sampler(backend=backend, sampler_opts=sampler_opts).run(dataset)

    assert result.success
    assert (
        result.samples.shape[1]
        == dataset.models.parameters.free_parameters.value.shape[0]
    )
    required_keys = [
        "logz",
        "logzerr",
        "posterior",
        "samples",
        "ncall",
        "insertion_order_MWW_test",
    ]
    assert set(required_keys).issubset(result.sampler_results.keys())

    assert_allclose(dataset.models.parameters["x"].value, 2, rtol=0.1)
    assert_allclose(dataset.models.parameters["y"].value, 3e2, rtol=0.1)
    assert_allclose(dataset.models.parameters["z"].value, 4e-2, rtol=0.1)
