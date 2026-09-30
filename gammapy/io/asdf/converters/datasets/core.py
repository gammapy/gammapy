# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from asdf.extension import Converter


def _sanitize_map_meta(m):
    """Convert boolean Quantity in map meta['is_pointlike'] to a Python bool."""
    if "is_pointlike" in m.meta:
        val = m.meta["is_pointlike"]
        if isinstance(val, u.Quantity):
            m = m.copy()
            m.meta["is_pointlike"] = bool(val.value)
    return m


class DatasetConverter(Converter):
    """Base converter for Dataset classes."""

    dataset_class = None
    default_stat_type = None
    extra_fields = []
    common_fields = [
        "counts",
        "exposure",
        "mask_fit",
        "mask_safe",
        "psf",
        "edisp",
        "gti",
        "meta_table",
    ]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "name": obj.name,
            "meta": obj.meta.model_dump() if obj.meta is not None else None,
            "stat_type": obj.stat_type,
        }

        for field in self.common_fields + self.extra_fields:
            value = getattr(obj, field, None)
            if value is not None:
                target = getattr(value, "exposure_map", value)
                if hasattr(target, "meta") and isinstance(
                    target.meta.get("is_pointlike"), u.Quantity
                ):
                    target.meta["is_pointlike"] = bool(
                        target.meta["is_pointlike"].value
                    )
                node[field] = value
        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.datasets import MapDatasetMetaData

        kwargs = {}
        for field in self.common_fields + self.extra_fields:
            kwargs[field] = node.get(field)

        kwargs["name"] = node.get("name")
        kwargs["stat_type"] = node.get("stat_type", self.default_stat_type)
        meta_node = node.get("meta")

        if meta_node is not None:
            try:
                kwargs["meta"] = MapDatasetMetaData(**meta_node)
            except Exception:
                kwargs["meta"] = None

        return self.dataset_class(**kwargs)
