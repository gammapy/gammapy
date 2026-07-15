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


class RegionGeomConverter(Converter):
    tags = ["asdf://gammapy.org/gammapy/tags/maps/regiongeom-1.0.0"]
    types = ["gammapy.maps.region.geom.RegionGeom"]

    def to_yaml_tree(self, obj, tag, ctx):
        from gammapy.utils.regions import compound_region_to_regions

        region = compound_region_to_regions(obj.region)
        ds9_strings = [_.serialize(format="ds9") for _ in region]
        node = {
            "region": ds9_strings,
            "axes": obj.axes,
            "wcs": obj.wcs,
        }

        return node

    def from_yaml_tree(self, node, tag, ctx):
        from gammapy.maps import RegionGeom
        from regions import Regions
        from gammapy.utils.regions import regions_to_compound_region

        ds9_strings = node["region"]

        region = [Regions.parse(_, format="ds9")[0] for _ in ds9_strings]
        for _ in region:
            _.meta.clear()
            _.visual.clear()

        region = regions_to_compound_region(region)
        return RegionGeom(
            region=region,
            axes=node.get("axes"),
            wcs=node.get("wcs"),
        )
