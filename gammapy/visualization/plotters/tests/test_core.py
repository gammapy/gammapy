# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import matplotlib
from gammapy.visualization.plotters import BasePlotter


class SubClassPlotter(BasePlotter):
    def __init__(self, rc_params=None):
        super().__init__()
        self.rc_params = {"image.cmap": "afmhot", "image.origin": "lower"}
        if rc_params:
            self.rc_params = rc_params


def test_init_empty():
    plotter = BasePlotter()
    assert isinstance(plotter.rc_params, matplotlib.RcParams)
    assert len(plotter.rc_params) == 0
    assert plotter.rc_params is not matplotlib.rcParams


def test_init_with_rc_params():
    plotter = BasePlotter({"image.cmap": "viridis"})
    assert plotter.rc_params["image.cmap"] == "viridis"
    assert len(plotter.rc_params) == 1


def test_does_not_mutate_global_rc_params():
    before = dict(matplotlib.rcParams)
    plotter = BasePlotter({"image.cmap": "afmhot"})
    plotter.rc_params = {"image.origin": "lower"}
    assert dict(matplotlib.rcParams) == before


def test_instances_are_independent():
    first = BasePlotter({"image.cmap": "viridis"})
    second = BasePlotter({"image.cmap": "afmhot"})
    first.rc_params = {"image.cmap": "gray"}
    assert second.rc_params["image.cmap"] == "afmhot"


def test_invalid_key_raises():
    with pytest.raises(KeyError):
        BasePlotter({"not.a.key": "x"})


def test_invalid_value_raises():
    with pytest.raises(ValueError):
        BasePlotter({"image.origin": "incorrect"})


@pytest.mark.parametrize("value", [["image.cmap", "afmhot"], "image.cmap"])
def test_non_dict_raises(value):
    plotter = BasePlotter()
    with pytest.raises(TypeError):
        plotter.rc_params = value


def test_setter_merges_with_existing_config():
    plotter = BasePlotter({"image.cmap": "viridis", "image.origin": "lower"})
    plotter.rc_params = {"image.cmap": "afmhot"}
    assert plotter.rc_params["image.cmap"] == "afmhot"
    assert plotter.rc_params["image.origin"] == "lower"


def test_setter_is_atomic_on_error():
    plotter = BasePlotter({"image.cmap": "viridis"})
    with pytest.raises(KeyError):
        plotter.rc_params = {"image.cmap": "afmhot", "not.a.key": "x"}
    assert plotter.rc_params["image.cmap"] == "viridis"
    assert "not.a.key" not in plotter.rc_params


def test_rc_context_applies_and_restores():
    original = matplotlib.rcParams["image.cmap"]
    plotter = BasePlotter({"image.cmap": "afmhot"})
    with plotter._rc_context():
        assert matplotlib.rcParams["image.cmap"] == "afmhot"
    assert matplotlib.rcParams["image.cmap"] == original


def test_rc_context_only_overrides_declared_keys():
    plotter = BasePlotter({"image.cmap": "afmhot"})
    with matplotlib.rc_context({"image.origin": "lower"}), plotter._rc_context():
        assert matplotlib.rcParams["image.cmap"] == "afmhot"
        assert matplotlib.rcParams["image.origin"] == "lower"


def test_subclass_sets_defaults():
    plotter = SubClassPlotter()
    assert plotter.rc_params["image.cmap"] == "afmhot"
    assert plotter.rc_params["image.origin"] == "lower"


def test_subclass_rc_params_override_defaults():
    plotter = SubClassPlotter({"image.cmap": "viridis"})
    assert plotter.rc_params["image.cmap"] == "viridis"
    assert plotter.rc_params["image.origin"] == "lower"
