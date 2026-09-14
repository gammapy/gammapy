# Licensed under a 3-clause BSD style license - see LICENSE.rst

from gammapy.io.asdf.converters.irf.core import IRFMapConverter


class EDispMapConverter(IRFMapConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/irf/edispmap-1.0.0"]
    types = ["gammapy.irf.EDispMap"]
    map_key = "edisp_map"

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.irf import EDispMap

        return EDispMap(**super().from_yaml_tree(node, tag, ctx))


class EDispKernelMapConverter(IRFMapConverter):
    tags = ["asdf://gammapy.org/gammapy/tags/irf/edispkernelmap-1.0.0"]
    types = ["gammapy.irf.EDispKernelMap"]
    map_key = "edisp_kernel_map"
    attr_name = "edisp_map"

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.irf import EDispKernelMap

        return EDispKernelMap(**super().from_yaml_tree(node, tag, ctx))
