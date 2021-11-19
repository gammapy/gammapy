"""Plot Fermi PSF."""
from gammapy.irf import PSFMap
from gammapy.maps import MapAxis, WcsGeom

filename = "$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz"
psf = PSFMap.read(filename, format="gtpsf")

axis = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=20, name="energy_true")
geom = WcsGeom.create(npix=50, binsz=0.01, axes=[axis])

# .to_image() computes the exposure weighted mean PSF
kernel = psf.get_psf_kernel(geom=geom).to_image()

kernel.psf_kernel_map.plot()
