# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class WcsNDMapConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/wcsndmap-1.0.0"]
    types = ["gammapy.maps.wcs.ndmap.WcsNDMap"]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "geom": obj.geom,
            "data": obj.data,
            "unit": str(obj.unit),
        }
        if obj.meta:
            node["meta"] = obj.meta

        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import WcsNDMap

        return WcsNDMap(
            geom=node["geom"],
            data=node["data"],
            meta=node.get("meta", {}),
            unit=node.get("unit", ""),
        )
