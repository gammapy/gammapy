# Licensed under a 3-clause BSD style license - see LICENSE.rst


from gammapy.io.asdf.converters.irf.core import IRFMapConverter


class PSFMapConverter(IRFMapConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/irf/psfmap-1.0.0"]
    types = ["gammapy.irf.PSFMap"]
    map_key = "psf_map"

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.irf import PSFMap

        return PSFMap(**super().from_yaml_tree(node, tag, ctx))


class RecoPSFMapConverter(IRFMapConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/irf/recopsfmap-1.0.0"]
    types = ["gammapy.irf.RecoPSFMap"]
    map_key = "psf_map"

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.irf import RecoPSFMap

        return RecoPSFMap(**super().from_yaml_tree(node, tag, ctx))
