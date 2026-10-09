# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class IRFMapConverter(Converter):
    map_key = None
    attr_name = None

    def to_yaml_tree(self, obj, tag, ctx):
        node = {self.map_key: getattr(obj, self.attr_name or self.map_key)}
        if obj.exposure_map is not None:
            node["exposure_map"] = obj.exposure_map

        return node

    def from_yaml_tree(self, node, tag, ctx):
        return {
            self.map_key: node[self.map_key],
            "exposure_map": node.get("exposure_map"),
        }
