# Licensed under a 3-clause BSD style license - see LICENSE.rst
from asdf.extension import Converter


class WcsGeomConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/wcsgeom-1.0.0"]
    types = ["gammapy.maps.wcs.geom.WcsGeom"]

    def to_yaml_tree(self, obj, tag, ctx):
        return {
            "wcs": obj.wcs,
            "npix": [obj.npix[0], obj.npix[1]],
            "cdelt": [obj._cdelt[0], obj._cdelt[1]],
            "crpix": [obj._crpix[0], obj._crpix[1]],
            "axes": obj.axes,
        }

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import WcsGeom
        import numpy as np

        npix = tuple(np.array(x) for x in node["npix"])
        cdelt = tuple(np.array(x) for x in node["cdelt"])
        crpix = tuple(np.array(x) for x in node["crpix"])
        return WcsGeom(
            wcs=node["wcs"],
            npix=npix,
            cdelt=cdelt,
            crpix=crpix,
            axes=node.get("axes"),
        )
