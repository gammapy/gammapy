# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pickle
import numpy as np
import gammapy.utils.cache as cache
from gammapy.utils.testing import requires_dependency


class Dummy(cache.CacheInstanceMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return isinstance(other, Dummy) and self.a == other.a and self.b == other.b


class Dummy2(cache.CacheInstanceMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return isinstance(other, Dummy2) and self.a == other.a and self.b == other.b


@requires_dependency("ray")
def test_dummy_cache():
    cache.USE_INSTANCE_CACHE = True

    x = Dummy(1, np.array(2))
    y = Dummy(1, np.array(2))
    z = Dummy2(1, np.array(2))

    u = Dummy(1, np.array(3))
    v = Dummy2(1, np.array(3))
    w = Dummy2(1, np.array(3))

    assert x is y
    assert z is not y

    assert u is not x
    assert u is not v
    assert v is not z
    assert v is w

    # the cache is not propagated through pickle
    # (implementing that would be more complex)
    data = pickle.dumps(x)
    xnew = pickle.loads(data)
    assert xnew is not x

    cache.USE_INSTANCE_CACHE = False

    x = Dummy(1, np.array(2))
    y = Dummy(1, np.array(2))
    z = Dummy2(1, np.array(2))

    assert x is not y
    assert z is not y
