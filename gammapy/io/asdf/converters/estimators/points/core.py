# Licensed under a 3-clause BSD style license - see LICENSE.rst

from asdf.extension import Converter


class FluxPointsConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/estimators/fluxpoints-1.0.0"]
    types = ["gammapy.estimators.points.FluxPoints"]

    def to_yaml_tree(self, obj, tag, ctx):
        from gammapy.modeling.models import Models

        node = {
            "data": obj._data,
            "reference_model": Models([obj.reference_model]),
            "filter_success_nan": bool(obj.filter_success_nan),
            "meta": obj.meta,
        }

        if obj.gti is not None:
            node["gti"] = obj.gti
        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.estimators.points import FluxPoints

        return FluxPoints(
            data=node["data"],
            reference_model=node["reference_model"][0],
            meta=node.get("meta"),
            gti=node.get("gti"),
            filter_success_nan=node.get("filter_success_nan", True),
        )
