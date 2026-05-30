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
            name=node["name"],
            nodes=node["nodes"],
            unit=node.get("unit", ""),
            interp=node["interp"],
            node_type=node["node_type"],
            boundary_type=node.get("boundary_type", "monotonic"),
        )
