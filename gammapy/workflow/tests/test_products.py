# Licensed under a 3-clause BSD style license - see LICENSE.rst
import time
import numpy as np
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_dependency
from gammapy.workflow.steps import WorkflowStepBase
from gammapy.workflow.products import Products, Product


class SumStep(WorkflowStepBase):
    tag = "sum"
    parallel = True
    required_data = [{"name": "value"}]
    products_names = ["value", "log"]  # could be update by the config in init

    def __init__(self, config=None, overwrite=False, **kwargs):
        super().__init__(config=None, overwrite=overwrite, **kwargs)

    def _sum(self):
        return np.sum(self.data.data)

    def _run(self):
        self.data.get()  # wait other remote and set results on self.data

        time.sleep(1)
        self.products[0].data = self._sum()
        self.products[1].data = "done"
        return self.products


def f_expected(n):
    values = []
    for k in range(0, n - 1):
        s = np.sum([1, 2] + values)
        values.append(s)
    return s


@requires_dependency("ray")
def test_product():
    import ray

    sum1 = SumStep()
    assert sum1.products.data[0] is None
    assert sum1.products.data[1] is None

    sum1.run(data=[Product(name="value", data=1), Product(name="value", data=2)])
    assert isinstance(sum1.products.data[0], ray.ObjectRef)
    assert isinstance(sum1.products.data[1], ray.ObjectRef)
    assert_allclose(sum1.data.data, [1, 2])

    sum1.run(data=sum1.products)
    assert_allclose(sum1.data.data[0], 1)
    assert_allclose(sum1.data.data[1], 2)
    assert isinstance(sum1.data.data[2], ray.ObjectRef)

    sum1.products.get()
    assert_allclose(sum1.products.data[0], 6)
    assert sum1.products.data[1] == "done"

    sum1.run(data=sum1.products)
    assert_allclose(sum1.data.data[3], 6)
    sum1.run(data=sum1.products)
    assert isinstance(sum1.data.data[4], ray.ObjectRef)

    sum2 = SumStep()
    sum2.run(data=Product(name="value", data=1))
    sum2.run(data=Product(name="value", data=2))

    analysis_products = Products([*sum1.products, *sum2.products])
    analysis_products.get()
    results = analysis_products.select(name="value").data

    assert_allclose(results[0], f_expected(len(sum1.data)))
    assert_allclose(results[1], f_expected(len(sum2.data)))
