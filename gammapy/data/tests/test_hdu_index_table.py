# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ..hdu_index_table import HDUIndexTable
from ...utils.testing import requires_data


@requires_data('gammapy-extra')
def test_hdu_index_table_hd_hap():
    """Test HESS HAP-HD data access."""
    hdu_index = HDUIndexTable.read('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/hdu-index.fits.gz')
    hdu_index.summary()

    # A few valid queries

    location = hdu_index.hdu_location(obs_id=23523, hdu_type='events')
    location.info()
    hdu = location.get_hdu()
    assert hdu.name == 'EVENTS'

    assert str(location.path(abs_path=False)) == 'run023400-023599/run023523/hess_events_023523.fits.gz'
    path1 = str(location.path(abs_path=True))
    path2 = str(location.path(abs_path=False))
    assert path1.endswith(path2)

    location = hdu_index.hdu_location(obs_id=23523, hdu_class='psf_3gauss')
    assert str(location.path(abs_path=False)) == 'run023400-023599/run023523/hess_psf_3gauss_023523.fits.gz'

    location = hdu_index.hdu_location(obs_id=23523, hdu_type='psf')
    assert str(location.path(abs_path=False)) == 'run023400-023599/run023523/hess_psf_3gauss_023523.fits.gz'

    # A few invalid queries

    with pytest.raises(IndexError) as exc:
        hdu_index.hdu_location(obs_id=42, hdu_class='psf_3gauss')
    msg = 'No entry available with OBS_ID = 42'
    assert exc.value.args[0] == msg

    with pytest.raises(IndexError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_type='bkg')
    msg = 'No HDU found matching: OBS_ID = 23523, HDU_TYPE = bkg, HDU_CLASS = None'
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523)
    msg = 'You have to specify `hdu_type` or `hdu_class`.'
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_type='invalid')
    msg = "Invalid hdu_type: invalid. Valid values are: ['events', 'gti', 'aeff', 'edisp', 'psf', 'bkg']"
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError) as exc:
        hdu_index.hdu_location(obs_id=23523, hdu_class='invalid')
    msg = "Invalid hdu_class: invalid. Valid values are: ['events', 'gti', 'aeff_2d', 'edisp_2d', 'psf_table', 'psf_3gauss', 'psf_king', 'bkg_2d', 'bkg_3d']"
    assert exc.value.args[0] == msg


@requires_data('gammapy-extra')
def test_hdu_index_table_pa():
    """Test HESS ParisAnalysis data access."""
    hdu_index = HDUIndexTable.read('$GAMMAPY_EXTRA/datasets/hess-crab4-pa/hdu-index.fits.gz')
    hdu_index.summary()

    # A few valid queries

    location = hdu_index.hdu_location(obs_id=23523, hdu_type='psf')
    location.info()
    assert str(location.path(abs_path=False)) == 'run23400-23599/run23523/psf_king_23523.fits.gz'

    location = hdu_index.hdu_location(obs_id=23523, hdu_class='psf_king')
    assert str(location.path(abs_path=False)) == 'run23400-23599/run23523/psf_king_23523.fits.gz'
