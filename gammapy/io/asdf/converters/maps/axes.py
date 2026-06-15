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


class TimeMapAxisConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0"]
    types = ["gammapy.maps.axes.TimeMapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "name": obj.name,
            "edges_min": obj.edges_min,
            "edges_max": obj.edges_max,
            "reference_time": obj.reference_time,
            "interp": obj.interp,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import TimeMapAxis

        return TimeMapAxis(
            name=node.get("name", "time"),
            edges_min=node["edges_min"],
            edges_max=node["edges_max"],
            reference_time=node["reference_time"],
            interp=node.get("interp", "lin"),
        )


class LabelMapAxisConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0"]
    types = ["gammapy.maps.axes.LabelMapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "labels": list(obj._labels),
            "name": obj.name,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import LabelMapAxis

        return LabelMapAxis(
            labels=node["labels"],
            name=node.get("name", ""),
        )
