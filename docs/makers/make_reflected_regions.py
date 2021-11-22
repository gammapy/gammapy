from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.makers import ReflectedRegionsFinder
from gammapy.maps import RegionGeom, WcsNDMap

# Exclude a rectangular region
exclusion_mask = WcsNDMap.create(npix=(801, 701), binsz=0.01, skydir=(83.6, 23.0))

coords = exclusion_mask.geom.get_coord().skycoord
data = (Angle("23 deg") < coords.dec) & (coords.dec < Angle("24 deg"))
exclusion_mask.data = ~data

pos = SkyCoord(83.633, 22.014, unit="deg")
radius = Angle(0.3, "deg")
on_region = CircleSkyRegion(pos, radius)
center = SkyCoord(83.633, 24, unit="deg")

# One can impose a minimal distance between ON region and first reflected regions
finder = ReflectedRegionsFinder(
    region=on_region,
    center=center,
    exclusion_mask=exclusion_mask,
    min_distance_input="0.2 rad",
)
regions = finder.run()

fig, axes = plt.subplots(
    ncols=3,
    subplot_kw={"projection": exclusion_mask.geom.wcs},
    figsize=(12, 3),
)


def plot_regions(ax, regions, on_region, exclusion_mask):
    """Little helper function to plot off regions"""
    exclusion_mask.plot_mask(ax=ax, colors="gray")
    on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="tab:orange")
    geom = RegionGeom.from_regions(regions)
    geom.plot_region(ax=ax, color="tab:blue")


ax = axes[0]
ax.set_title("Min. distance first region")
plot_regions(ax=ax, regions=regions, on_region=on_region, exclusion_mask=exclusion_mask)


# One can impose a minimal distance between two adjacent regions
finder = ReflectedRegionsFinder(
    region=on_region,
    center=center,
    exclusion_mask=exclusion_mask,
    min_distance="0.1 rad",
)
regions = finder.run()

ax = axes[1]
ax.set_title("Min. distance all regions")
plot_regions(ax=ax, regions=regions, on_region=on_region, exclusion_mask=exclusion_mask)


# One can impose a maximal number of regions to be extracted
finder = ReflectedRegionsFinder(
    region=on_region,
    center=center,
    exclusion_mask=exclusion_mask,
    max_region_number=5,
    min_distance="0.1 rad",
)
regions = finder.run()

ax = axes[2]
ax.set_title("Max. number of regions")
plot_regions(ax=ax, regions=regions, on_region=on_region, exclusion_mask=exclusion_mask)
plt.show()
