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
def test_hess_chains(data_manager, chain):

    ref_file_aeff = make_path(chain['aeff2D_reference_file'])
    ref_aeff = open(str(ref_file_aeff), 'r').read()
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

    # Todo : Find a better way to capture stdout
    # see http://stackoverflow.com/questions/5136611/capture-stdout-from-a-script-in-python/10743550#10743550
    import sys
    from StringIO import StringIO

    backup = sys.stdout
    sys.stdout = StringIO()
    obs.info()
    obs_info = sys.stdout.getvalue()
    sys.stdout.close()
    sys.stdout = backup

    assert store.obs_table['OBS_ID'][obs_nr] == obs_id
    assert obs_info == ref_obs

    assert obs.aeff.info() == ref_aeff
    assert obs.edisp.info() == ref_edisp
    assert obs.psf.info() == ref_psf

