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


class HpxGeomConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0"]
    types = ["gammapy.maps.hpx.geom.HpxGeom"]

    def to_yaml_tree(self, obj, tag, ctx):
        node = {
            "nside": obj.nside,
            "nest": obj.nest,
            "frame": obj.frame,
            "axes": obj.axes,
        }
        if obj.region is not None:
            node["region"] = obj.region
        if obj.region == "explicit":
            node["ipix"] = obj._ipix

        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import HpxGeom

        nside = node["nside"]
        region = node.get("region")
        if region == "explicit":
            import numpy as np
            from gammapy.maps.hpx.utils import unravel_hpx_index

            ipix = np.array(node["ipix"])
            npix_max = 12 * np.max(nside) ** 2
            region = unravel_hpx_index(ipix, np.array([npix_max]))

        return HpxGeom(
            nside=nside,
            nest=node.get("nest", True),
            frame=node.get("frame", "icrs"),
            region=region,
            axes=node.get("axes"),
        )
