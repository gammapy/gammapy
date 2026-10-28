# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class GTIConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/data/gti-1.0.0"]
    types = ["gammapy.data.gti.GTI"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "table": obj.table,
            "reference_time": obj.time_ref,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.data import GTI

        return GTI(
            table=node["table"],
            reference_time=node.get("reference_time"),
        )
