# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.cache import CacheEquivalentMixin


class Dummy(CacheEquivalentMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return isinstance(other, Dummy) and self.a == other.a and self.b == other.b

    def __repr__(self):
        return f"MyObject(x={self.x}, y={self.y})"


class Dummy2(CacheEquivalentMixin):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return isinstance(other, Dummy2) and self.a == other.a and self.b == other.b

    def __repr__(self):
        return f"MyObject(x={self.x}, y={self.y})"


def test_dummy_cache():
    x = Dummy(1, 2)
    y = Dummy(1, 2)
    z = Dummy2(1, 2)

    u = Dummy(1, 3)
    v = Dummy2(1, 3)
    w = Dummy2(1, 3)

    assert x is y
    assert z is not y

    assert u is not x
    assert v is not z
    assert v is w
