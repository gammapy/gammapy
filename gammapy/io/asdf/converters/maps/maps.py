# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class MapsConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/maps-1.0.0"]
    types = ["gammapy.maps.maps.Maps"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {key: value for key, value in obj.items()}

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import Maps

        return Maps(**node)
