# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.datasets import MapDataset, MapDatasetOnOff, MapDatasetMetaData
from gammapy.io.asdf.converters.datasets.core import DatasetConverter


class MapDatasetConverter(DatasetConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/datasets/mapdataset-1.0.0"]
    types = ["gammapy.datasets.map.MapDataset"]
    dataset_class = MapDataset
    default_stat_type = "cash"
    metadata_class = MapDatasetMetaData
    extra_fields = ["background"]


class MapDatasetOnOffConverter(DatasetConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/datasets/mapdatasetonoff-1.0.0"]
    types = ["gammapy.datasets.map.MapDatasetOnOff"]
    dataset_class = MapDatasetOnOff
    default_stat_type = "wstat"
    metadata_class = MapDatasetMetaData
    extra_fields = ["counts_off", "acceptance", "acceptance_off"]
