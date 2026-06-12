# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class MapAxisConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0"]
    types = ["gammapy.maps.axes.MapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "name": obj.name,
            "nodes": obj._nodes,
            "unit": str(obj._unit),
            "interp": obj.interp,
            "node_type": obj.node_type,
            "boundary_type": obj._boundary_type,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import MapAxis

        return MapAxis(
            name=node.get("name", ""),
            nodes=node["nodes"],
            unit=node.get("unit", ""),
            interp=node.get("interp", "lin"),
            node_type=node.get("node_type", "edges"),
            boundary_type=node.get("boundary_type", "monotonic"),
        )
