# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ..freeze_attr import freeze


class Other:
    def __init__(self):
        self.other = "other_attr"


@freeze
class Parent:
    tag = "p"

    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        self._a = val

    def some_function(self):
        return self.a + self.b


@freeze
class Child(Parent):
    tag = "c"
    counts = None

    def __init__(self, counts=None, g=3, h=4):
        super().__init__(1, 2)
        self.g = g
        self.h = h
        self.counts = counts

    def other_function(self):
        return self.g + self.h


@freeze
class GrandChild(Child, Other):
    tag = "g"

    def __init__(self, counts, g=5, h=6):
        Child.__init__(self, counts, g, h)
        Other.__init__(self)


def test_freeze_decorator():
    parent = Parent(1, 2)

    assert parent._allowed_attrs == {"tag", "a", "b"}
    assert parent.some_function() == 3
    assert parent.tag == "p"
    with pytest.raises(AttributeError):
        parent.bad = "bad"

    parent._frozen = False
    parent.good = "good"
    parent._frozen = True
    with pytest.raises(AttributeError):
        parent.bad = "bad"

    child = Child(counts="counts")
    assert child._allowed_attrs == {"tag", "a", "b", "counts", "g", "h"}
    assert child.other_function() == 7
    assert child.tag == "c"
    with pytest.raises(AttributeError):
        child.bad = "bad"

    grandchild = GrandChild(counts="counts")
    assert grandchild._allowed_attrs == {"tag", "a", "b", "counts", "g", "h", "other"}
    assert grandchild.other_function() == 11
    assert grandchild.tag == "g"
    with pytest.raises(AttributeError):
        grandchild.bad = "bad"
    assert grandchild.other == "other_attr"
