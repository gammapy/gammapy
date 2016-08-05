# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import pytest
from ...datasets.core import GammapyExtraNotFoundError
from ...utils.scripts import make_path
from ...utils.testing import requires_dependency, requires_data, data_manager
from ...datasets import gammapy_extra


def get_list_of_chains():
    """Provide parametrization list for test_EffectiveArea

    Returns emtpy list if YAML or Gammapy extra are not available, but the
    test does not run in this case anyway.
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
def test_hess_chains(data_manager, chain):
    ref_file_aeff = make_path(chain['aeff2D_reference_file'])
    ref_aeff = np.loadtxt(str(ref_file_aeff))
    ref_file_edisp = make_path(chain['edisp2D_reference_file'])
    ref_edisp = open(str(ref_file_edisp), 'r').read()
    ref_file_psf = make_path(chain['psf_reference_file'])
    ref_psf = open(str(ref_file_psf), 'r').read()
    ref_file_obs = make_path(chain['obs_reference_file'])
    ref_obs = open(str(ref_file_obs), 'r').read()
    ref_file_loc = make_path(chain['location_reference_file'])
    ref_loc = open(str(ref_file_loc), 'r').read()

    obs_nr = chain['obs']
    obs_id = chain['obs_id']

    store = data_manager[chain['store']]
    obs = store.obs(obs_id=obs_id)

    assert str(obs.location(hdu_type='events').path(abs_path=False)) == ref_loc

    assert store.obs_table['OBS_ID'][obs_nr] == obs_id
    assert str(obs) == ref_obs

    assert obs.edisp.info() == ref_edisp
    assert obs.psf.info() == ref_psf

    # These values are copied from
    # gammapy-extra/test_datasets/reference/make_reference_files.py 
    test_energy = [0.1, 1, 5, 10] * u.TeV
    test_offset = [0.1, 0.2, 0.4] * u.deg

    aeff_val = obs.aeff.evaluate(offset=test_offset, energy=test_energy)
    assert_allclose(aeff_val.value, ref_aeff)
