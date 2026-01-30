# Licensed under a 3-clause BSD style license - see LICENSE.rst
import time
import numpy as np
import pytest
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_dependency
from gammapy.workflow.steps import WorkflowStepBase
from gammapy.workflow.products import Products, Product


class SumStep(WorkflowStepBase):
    tag = "sum"
    parallel = True
    required_inputs = [{"name": "value"}]
    outputs_names = ["value", "log"]  # could be updated by the config in init

    def __init__(self, config=None, overwrite=False, **kwargs):
        super().__init__(config=None, overwrite=overwrite, **kwargs)

    def _sum(self):
        return np.sum(self.inputs.data)

    def _run(self):
        time.sleep(1)
        self.outputs[0].data = self._sum()
        self.outputs[1].data = "done"
        return self.outputs


def f_expected(n):
    values = []
    for _ in range(0, n - 1):
        s = np.sum([1, 2] + values)
        values.append(s)
    return s


@requires_dependency("ray")
def test_products():
    import ray

    sum1 = SumStep(name="sum1")
    assert sum1.outputs.data[0] is None
    assert sum1.outputs.data[1] is None
    assert sum1.outputs[0].step_name == "sum1"

    sum1.run(inputs=[Product(name="value", data=1), Product(name="value", data=2)])
    assert isinstance(sum1.outputs.data[0], ray.ObjectRef)
    assert isinstance(sum1.outputs.data[1], ray.ObjectRef)
    assert_allclose(sum1.inputs.data, [1, 2])

    sum1.run(inputs=sum1.outputs)
    assert_allclose(sum1.inputs.data[0], 1)
    assert_allclose(sum1.inputs.data[1], 2)
    assert isinstance(sum1.inputs.data[2], ray.ObjectRef)

    sum1.outputs.get()
    assert_allclose(sum1.outputs.data[0], 6)
    assert sum1.outputs.data[1] == "done"

    sum1.run(inputs=sum1.outputs)
    assert_allclose(sum1.inputs.data[3], 6)
    sum1.run(inputs=sum1.outputs)
    assert isinstance(sum1.inputs.data[4], ray.ObjectRef)

    sum2 = SumStep()
    assert sum2.outputs[0].step_name == sum2.name
    sum2.run(inputs=Product(name="value", data=1))
    sum2.run(inputs=Product(name="value", data=2))

    analysis_products = Products([*sum1.outputs, *sum2.outputs])
    analysis_products.get()
    results = analysis_products.select(name="value").data

    assert_allclose(results[0], f_expected(len(sum1.inputs)))
    assert_allclose(results[1], f_expected(len(sum2.inputs)))

    unique_names = analysis_products.unique_names
    assert len(unique_names) == 4
    assert unique_names[0] == f"sum1.value.{sum1.outputs[0].pid}"

    unique_names = analysis_products.select(step_name="sum1").unique_names
    for name in unique_names:
        assert "sum1" in name

    assert analysis_products.pids[0] == sum1.outputs[0].pid


def test_products_no_ray():
    sum1 = SumStep(name="sum1")
    sum1.parallel = False
    assert sum1.outputs.data[0] is None
    assert sum1.outputs.data[1] is None
    assert sum1.outputs[0].step_name == "sum1"

    sum1.run(inputs=[Product(name="value", data=1), Product(name="value", data=2)])
    assert_allclose(sum1.outputs.data[0], 3)
    assert sum1.outputs.data[1] == "done"
    assert_allclose(sum1.inputs.data, [1, 2])

    sum1.run(inputs=sum1.outputs)
    assert_allclose(sum1.inputs.data[0], 1)
    assert_allclose(sum1.inputs.data[1], 2)
    assert_allclose(sum1.inputs.data[2], 3)

    assert_allclose(sum1.outputs.data[0], 6)
    assert sum1.outputs.data[1] == "done"

    sum1.run(inputs=sum1.outputs)
    assert_allclose(sum1.inputs.data[3], 6)
    sum1.run(inputs=sum1.outputs)
    assert_allclose(sum1.inputs.data[4], 12)

    sum2 = SumStep()
    sum2.parallel = False

    assert sum2.outputs[0].step_name == sum2.name
    sum2.run(inputs=Product(name="value", data=1))
    sum2.run(inputs=Product(name="value", data=2))

    analysis_products = Products([*sum1.outputs, *sum2.outputs])
    results = analysis_products.select(name="value").data

    assert_allclose(results[0], f_expected(len(sum1.inputs)))
    assert_allclose(results[1], f_expected(len(sum2.inputs)))


def test_products_modification():
    products = Products([Product(name="value", data=1), Product(name="value", data=2)])
    products[0] = products[1]
    assert_allclose(products.data, 2)
    del products[0]
    assert len(products) == 1

    with pytest.raises(TypeError):
        products[0] = np.ones(1)
