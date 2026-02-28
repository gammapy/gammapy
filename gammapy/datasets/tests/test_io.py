# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from regions import CircleSkyRegion
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FermipyDatasetsReader,
)
from gammapy.modeling.models import DatasetModels
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@requires_data()
def test_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(
        filename=filedata,
        filename_models=filemodel,
    )

    assert len(datasets) == 2
    assert len(datasets.models) == 5
    dataset0 = datasets[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None

    assert_allclose(dataset0.npred_background().data.sum(), 15726.8, atol=0.1)

    assert dataset0.background_model.name == "gc-bkg"

    dataset1 = datasets[1]
    assert dataset1.name == "g09"
    assert dataset1.background_model.name == "g09-bkg"

    assert (
        dataset0.models["gll_iem_v06_cutout"] == dataset1.models["gll_iem_v06_cutout"]
    )

    assert isinstance(dataset0.models, DatasetModels)
    assert len(dataset0.models) == 4
    assert dataset0.models[0].name == "gc"
    assert dataset0.models[1].name == "gll_iem_v06_cutout"
    assert dataset0.models[2].name == "gc-bkg"

    assert (
        dataset0.models["gc"].parameters["reference"]
        is dataset1.models["g09"].parameters["reference"]
    )
    assert_allclose(dataset1.models["g09"].parameters["lon_0"].value, 0.9, atol=0.1)

    datasets.write(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    datasets.write(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
        overwrite=True,
    )

    datasets_read = Datasets.read(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    assert len(datasets.parameters) == 24

    assert len(datasets_read) == 2
    dataset0 = datasets_read[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None
    assert_allclose(dataset0.npred_background().data.sum(), 15726.8, atol=0.1)
    assert datasets[1].name == "g09"

    dataset_copy = dataset0.copy(name="dataset0-copy")
    assert dataset_copy.models is None


@requires_data()
def test_spectrum_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(
        filename=filedata,
        filename_models=filemodel,
    )
    reg = CircleSkyRegion(center=datasets[0]._geom.center_skydir, radius=1.0 * u.deg)
    dataset0 = datasets[0].to_spectrum_dataset(reg)
    datasets1 = Datasets([dataset0, datasets[1]])
    datasets1.write(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    datasets_read = Datasets.read(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    assert len(datasets_read.parameters) == 21

    assert len(datasets_read) == 2

    assert datasets_read[0].counts.data.sum() == 18429
    assert_allclose(datasets_read[0].exposure.data.sum(), 2.034089e10, atol=0.1)
    assert isinstance(datasets_read[0], SpectrumDataset)


@requires_data()
def test_ogip_writer(tmp_path):
    dataset = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits",
        format="ogip",
    )
    dataset.counts_off.data = np.zeros(dataset.counts_off.data.shape)
    datasets = Datasets(dataset)
    datasets.write(tmp_path / "written_datasets.yaml")
    new_datasets = datasets.read(tmp_path / "written_datasets.yaml")
    assert_allclose(
        new_datasets[0].counts_off.data, np.zeros(new_datasets[0].counts_off.data.shape)
    )


@requires_data()
def test_datasets_write_checksum(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(
        filename=filedata,
        filename_models=filemodel,
    )

    filename = tmp_path / "written_datasets.yaml"
    filename_models = tmp_path / "written_models.yaml"
    datasets.write(
        filename=filename,
        filename_models=filename_models,
        checksum=True,
    )

    assert "checksum" in filename.read_text()
    assert "checksum" in filename_models.read_text()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Datasets.read(filename=filename, filename_models=filename_models, checksum=True)

    # Remove checksum from datasets.yaml file
    yaml_content = filename.read_text()
    index = yaml_content.find("checksum")
    bad = make_path(tmp_path) / "bad_checksum.yaml"
    bad.write_text(yaml_content[:index])

    with pytest.warns(UserWarning):
        Datasets.read(filename=bad, filename_models=filename_models, checksum=True)

    # Modify models yaml file
    yaml_content = filename_models.read_text()
    yaml_content = yaml_content.replace("name: gc", "name: bad")
    bad = make_path(tmp_path) / "bad_models.yaml"
    bad.write_text(yaml_content)

    with pytest.warns(UserWarning):
        Datasets.read(filename=filename, filename_models=bad, checksum=True)


@requires_data()
def test_fermipy_datasets_reader():
    reader = FermipyDatasetsReader(
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_std_5deg_qr.yaml", edisp_bins=1
    )
    datasets = reader.read()

    assert len(datasets) == 2
    assert datasets[0].counts.geom.axes[0].unit == "MeV"
    assert_allclose(datasets[0].background, 0)
    assert datasets[0].counts.geom.to_image() == datasets[0].exposure.geom.to_image()
    assert_allclose(datasets[0].exposure.data[0, 0, 0], 1.54938e11)
    assert_allclose(datasets[0].edisp.edisp_map.data[0, 0, 0, 0], 0.020409, rtol=1e-4)
    assert_allclose(
        datasets[0]._psf_kernel.psf_kernel_map.data.sum(axis=(1, 2)), 1, rtol=1e-5
    )
    assert_allclose(
        datasets[0].edisp.exposure_map.data, datasets[0].psf.exposure_map.data
    )
    assert datasets[0].name == "P8R3_SOURCEVETO_V3_PSF0_v1"
    assert datasets[1].name == "P8R3_SOURCEVETO_V3_PSF1_v1"
    assert datasets.models.names[0] == "isotropic_P8R3_SOURCEVETO_V3_PSF0_v1"
    assert datasets.models.names[1] == "isotropic_P8R3_SOURCEVETO_V3_PSF1_v1"
    assert datasets.gti is not None
    assert_allclose(
        datasets.gti.time_start.value,
        [54682.65603794, 54682.65603794],
        rtol=1 / (24 * 60),
    )

    path = make_path("$GAMMAPY_DATA/tests/fermi")
    dataset = reader.create_dataset(
        path / "ccube_00.fits",
        path / "bexpmap_00.fits",
        path / "psf_00.fits",
        path / "drm_00.fits",
    )
    assert not dataset.models
    assert dataset.gti is None


@requires_data()
def test_fermipy_datasets_reader_no_components():
    files = [
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_minimal.yaml",
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_empty_component.yaml",
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_null_component.yaml",
    ]
    for file in files:
        reader = FermipyDatasetsReader(file, edisp_bins=1)

        datasets = reader.read()

        assert len(datasets) == 1
        assert datasets[0].counts.geom.axes[0].unit == "MeV"
        assert_allclose(datasets[0].background, 0)
        assert (
            datasets[0].counts.geom.to_image() == datasets[0].exposure.geom.to_image()
        )
        assert_allclose(datasets[0].exposure.data[0, 0, 0], 1.54938e11)
        assert_allclose(
            datasets[0].edisp.edisp_map.data[0, 0, 0, 0], 0.020409, rtol=1e-4
        )
        assert_allclose(
            datasets[0]._psf_kernel.psf_kernel_map.data.sum(axis=(1, 2)), 1, rtol=1e-5
        )
        assert_allclose(
            datasets[0].edisp.exposure_map.quantity,
            datasets[0].psf.exposure_map.quantity,
        )
        assert datasets[0].edisp.exposure_map.unit == datasets[0].psf.exposure_map.unit
        assert datasets[0].edisp.exposure_map.unit == datasets[0].exposure.unit

        assert datasets[0].name == "P8R3_SOURCEVETO_V3_PSF1_v1"
        assert datasets.models.names[0] == "isotropic_P8R3_SOURCEVETO_V3_PSF1_v1"

        assert datasets[0].gti.time_sum.unit == "s"
        assert_allclose(datasets[0].gti.time_sum.value, 426196949.14969766, rtol=60)


@requires_data()
def test_fermipy_datasets_reader_invalid_iso():
    files = [
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_invalid_iso_list_missing.yaml",
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_invalid_iso_list.yaml",
        "$GAMMAPY_DATA/tests/fermi/config_fermipy_invalid_iso_missing.yaml",
    ]
    for file in files:
        with pytest.raises(ValueError):
            FermipyDatasetsReader(file, edisp_bins=1).read()
