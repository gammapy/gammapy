# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class MapAxisConverter(Converter):
    """ASDF converter for the MapAxis class."""

    tags = ["asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0"]
    types = ["gammapy.maps.axes.MapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        """Convert a MapAxis object into a node suitable for YAML serialization."""
        return {
            "name": obj.name,
            "nodes": obj._nodes,
            "unit": str(obj._unit),
            "interp": obj.interp,
            "node_type": obj.node_type,
            "boundary_type": obj._boundary_type,
        }

    def from_yaml_tree(self, node, tag, ctx):
        """Reconstruct a MapAxis object from a YAML node."""
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
    """ASDF converter for the TimeMapAxis class."""

    tags = ["asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0"]
    types = ["gammapy.maps.axes.TimeMapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        """Convert a TimeMapAxis object into a node suitable for YAML serialization."""
        return {
            "name": obj.name,
            "edges_min": obj.edges_min,
            "edges_max": obj.edges_max,
            "reference_time": obj.reference_time,
            "interp": obj.interp,
        }

    def from_yaml_tree(self, node, tag, ctx):
        """Reconstruct a TimeMapAxis object from a YAML node."""
        from gammapy.maps import TimeMapAxis

        return TimeMapAxis(
            name=node.get("name", "time"),
            edges_min=node["edges_min"],
            edges_max=node["edges_max"],
            reference_time=node["reference_time"],
            interp=node.get("interp", "lin"),
        )


class LabelMapAxisConverter(Converter):
    """ASDF converter for the LabelMapAxis class."""

    tags = ["asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0"]
    types = ["gammapy.maps.axes.LabelMapAxis"]

    def to_yaml_tree(self, obj, tag, ctx):
        """Convert a LabelMapAxis object into a node suitable for YAML serialization."""
        return {
            "labels": obj.center.tolist(),
            "name": obj.name,
        }

    def from_yaml_tree(self, node, tag, ctx):
        """Reconstruct a LabelMapAxis object from a YAML node."""
        from gammapy.maps import LabelMapAxis

        return LabelMapAxis(
            labels=node["labels"],
            name=node.get("name", ""),
        )


class MapAxesConverter(Converter):
    """ASDF converter for the MapAxes class."""

    tags = ["asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0"]
    types = ["gammapy.maps.axes.MapAxes"]

    def to_yaml_tree(self, obj, tag, ctx):
        """Convert a MapAxes object into a node suitable for YAML serialization."""
        return {
            "axes": list(obj),
        }

    def from_yaml_tree(self, node, tag, ctx):
        """Reconstruct a MapAxes object from a YAML node."""
        from gammapy.maps import MapAxes

        return MapAxes(
            axes=node["axes"],
        )
