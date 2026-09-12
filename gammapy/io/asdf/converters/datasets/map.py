# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class MapDatasetConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/datasets/mapdataset-1.0.0"]
    types = ["gammapy.datasets.map.MapDataset"]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "name": obj.name,
            "meta": obj.meta.model_dump(),
        }
        if obj.counts is not None:
            node["counts"] = obj.counts
        if obj.exposure is not None:
            node["exposure"] = obj.exposure
        if obj.background is not None:
            node["background"] = obj.background
        if obj.mask_fit is not None:
            node["mask_fit"] = obj.mask_fit
        if obj.mask_safe is not None:
            node["mask_safe"] = obj.mask_safe
        if obj.psf is not None:
            node["psf"] = obj.psf
        if obj.edisp is not None:
            node["edisp"] = obj.edisp
        if obj.gti is not None:
            node["gti"] = obj.gti
        if obj.meta_table is not None:
            node["meta_table"] = obj.meta_table
        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.datasets import MapDataset, MapDatasetMetaData

        meta_node = node.get("meta")
        return MapDataset(
            counts=node.get("counts"),
            exposure=node.get("exposure"),
            background=node.get("background"),
            psf=node.get("psf"),
            edisp=node.get("edisp"),
            mask_fit=node.get("mask_fit"),
            mask_safe=node.get("mask_safe"),
            gti=node.get("gti"),
            meta_table=node.get("meta_table"),
            name=node.get("name"),
            meta=MapDatasetMetaData(**meta_node) if meta_node is not None else None,
        )


class MapDatasetOnOffConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/datasets/mapdatasetonoff-1.0.0"]
    types = ["gammapy.datasets.map.MapDatasetOnOff"]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "name": obj.name,
            "meta": obj.meta.model_dump(),
        }
        if obj.counts is not None:
            node["counts"] = obj.counts
        if obj.counts_off is not None:
            node["counts_off"] = obj.counts_off
        if obj.exposure is not None:
            node["exposure"] = obj.exposure
        if obj.acceptance is not None:
            node["acceptance"] = obj.acceptance
        if obj.acceptance_off is not None:
            node["acceptance_off"] = obj.acceptance_off
        if obj.mask_fit is not None:
            node["mask_fit"] = obj.mask_fit
        if obj.mask_safe is not None:
            node["mask_safe"] = obj.mask_safe
        if obj.psf is not None:
            node["psf"] = obj.psf
        if obj.edisp is not None:
            node["edisp"] = obj.edisp
        if obj.gti is not None:
            node["gti"] = obj.gti
        if obj.meta_table is not None:
            node["meta_table"] = obj.meta_table
        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.datasets import MapDatasetOnOff, MapDatasetMetaData

        meta_node = node.get("meta")
        return MapDatasetOnOff(
            counts=node.get("counts"),
            counts_off=node.get("counts_off"),
            exposure=node.get("exposure"),
            acceptance=node.get("acceptance"),
            acceptance_off=node.get("acceptance_off"),
            gti=node.get("gti"),
            psf=node.get("psf"),
            edisp=node.get("edisp"),
            mask_fit=node.get("mask_fit"),
            mask_safe=node.get("mask_safe"),
            meta_table=node.get("meta_table"),
            name=node.get("name"),
            meta=MapDatasetMetaData(**meta_node) if meta_node is not None else None,
        )
