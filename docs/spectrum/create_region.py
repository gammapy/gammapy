"""Example how to compute and plot reflected regions."""
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion, EllipseAnnulusSkyRegion
from gammapy.maps import WcsNDMap

position = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")

on_circle = CircleSkyRegion(position, 0.3 * u.deg)

on_ellipse_annulus = EllipseAnnulusSkyRegion(
    center=position,
    inner_width=1.5 * u.deg,
    outer_width=2.5 * u.deg,
    inner_height=3 * u.deg,
    outer_height=4 * u.deg,
    angle=130 * u.deg,
)

another_position = SkyCoord(80.3, 22.0, unit="deg")
on_rectangle = RectangleSkyRegion(
    center=another_position, width=2.0 * u.deg, height=4.0 * u.deg, angle=50 * u.deg
)

# Now we plot those regions. We first create an empty map
empty_map = WcsNDMap.create(
    skydir=position, width=10 * u.deg, binsz=0.1 * u.deg, proj="TAN"
)
empty_map.data += 1.0
empty_map.plot(cmap="gray", vmin=0, vmax=1)

# To plot the regions, we convert them to PixelRegion with the map wcs
on_circle.to_pixel(empty_map.geom.wcs).plot()
on_rectangle.to_pixel(empty_map.geom.wcs).plot()
on_ellipse_annulus.to_pixel(empty_map.geom.wcs).plot()

plt.show()
