# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class ModelsConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/modeling/models-1.0.0"]
    types = ["gammapy.modeling.models.core.Models"]

    def to_yaml_tree(self, obj, tag, ctx):
        node = obj.to_dict(full_output=True)
        node.pop("covariance", None)
        node["covariance_data"] = obj.covariance.data
        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.modeling.models import Models

        covariance_data = node.pop("covariance_data", None)
        models = Models.from_dict(node)
        if covariance_data is not None:
            models.covariance = covariance_data

        return models
