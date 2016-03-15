# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals
from astropy.tests.helper import pytest

from ...datasets.core import GammapyExtraNotFoundError
from ...utils.scripts import make_path
from ...utils.testing import requires_dependency, requires_data, data_manager
from ...datasets import gammapy_extra


def get_list_of_chains():
    """Provide parametrization list for test_EffectiveArea

    Returns emtpy list if YAML or Gammapy extra are available, but the
    test only run if this is false anyway.
    """
    try:
        import yaml
    except ImportError:
        return []
    try:
        structure_file = gammapy_extra.filename(
            'test_datasets/reference/reference_info.yaml')
    except GammapyExtraNotFoundError:
        return []
    with open(structure_file) as fh:
        test_args = yaml.safe_load(fh)
    return test_args


@pytest.mark.parametrize('chain', get_list_of_chains())
@requires_dependency('scipy')
@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_EffectiveArea(data_manager, chain):
    ref_file = make_path(chain['aeff2D_reference_file'])
    ref_aeff = open(str(ref_file), 'r').read()
    store = data_manager[chain['store']]
    aeff = store.obs(obs_id=chain['obs']).aeff
    assert aeff.info() == ref_aeff
