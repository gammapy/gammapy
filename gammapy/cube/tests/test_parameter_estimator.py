# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.cube import ParameterEstimator

@pytest.fixture
def crab_datasets_1d():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.from_ogip_files(filename)
    datasets = Datasets([dataset])
    return datasets

@pytest.fixture
def PLmodel():
    return PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)


@requires_data()
def test_parameter_estimator_simple(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ParameterEstimator(datasets)

    result = estimator.run(PLmodel.amplitude, steps="all")

    # First make sure that parameters are correctly set on the dataset.models object
    assert_allclose(result['value'], 5.142843823441639e-11, rtol=1e-4)
    assert_allclose(result['err'], 6.42467002840316e-12, rtol=1e-4)
    assert_allclose(result['delta_ts'], 353.2092043652601, rtol=1e-4)
    assert_allclose(result['errp'], 6.703e-12, rtol=1e-3)
    assert_allclose(result['errn'], 6.152e-12, rtol=1e-3)
    # Add test for scan
