# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest
from ...irf import OffsetDependentTableEffectiveArea,TableEffectiveArea, abramowski_effective_area
from ...datasets import load_arf_fits_table, load_aeff2D_fits_table

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_abramowski_effective_area():
    energy = Quantity(100, 'GeV')
    area_ref = Quantity(1.65469579e+07, 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

    energy = Quantity([0.1, 2], 'TeV')
    area_ref = Quantity([1.65469579e+07, 1.46451957e+09], 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit


@pytest.mark.skipif('not HAS_SCIPY')
def test_TableEffectiveArea():
    filename = get_pkg_data_filename('data/arf_info.txt')
    info_str = open(filename, 'r').read()
    arf = TableEffectiveArea.from_fits(load_arf_fits_table())
    assert arf.info() == info_str


def test_TableEffectiveArea_write():
    from tempfile import NamedTemporaryFile
    from astropy.io import fits

    # Read test psf file
    psf = TableEffectiveArea.from_fits(load_arf_fits_table())

    # Write it back to disk
    with NamedTemporaryFile(suffix='.fits') as psf_file:
        psf.write(psf_file.name)

        # Verify checksum
        hdu_list = fits.open(psf_file.name)
        # TODO: replace this assert with something else.
        # For unknown reasons this verify_checksum fails non-deterministically
        # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
        # assert hdu_list[1].verify_checksum() == 1
        assert len(hdu_list) == 2


@pytest.mark.skipif('not HAS_SCIPY')
def test_OffsetDependentTableEffectiveArea():
    
    print("Executing tests ...)")

    #Read test effective area file
    effarea = OffsetDependentTableEffectiveArea.from_fits(load_aeff2D_fits_table())


    #Check that nodes are evaluated correctly
    e_node   = 42
    off_node = 3
    actual  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node])
    desired =  effarea.eff_area[off_node,e_node]
    assert_equal(actual, desired)

    #Check that values between node make sense    
    upper  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node])
    lower  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node+1])
    e_val  =  (effarea.energy[e_node]+effarea.energy[e_node+1])/2
    actual =  effarea.evaluate(effarea.offset[off_node],e_val)
    assert_equal(lower > actual and actual > upper , True)
    
    #and the same for the spline interpolator
    effarea.set_interpolation_method('spline')
    actual  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node])
    assert_allclose(actual, desired)  #not exactly equal!
    upper  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node])
    lower  =  effarea.evaluate(effarea.offset[off_node],effarea.energy[e_node+1])
    e_val  =  (effarea.energy[e_node]+effarea.energy[e_node+1])/2
    actual =  effarea.evaluate(effarea.offset[off_node],e_val)
    assert_equal(lower > actual and actual > upper , True)


    #USECASE: SpectralAnalysis
    offset = Angle(0.234,'degree')

    #Get a 1D vector of effective area values
    nbins = 42
    energies = Quantity(np.logspace(3,4,nbins),'GeV')
    actual = effarea.eval_at_offset(offset,energies).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    

    #USECASE: ExposureMap 
    #TODO
