"""
Using Gammapy IRFs
==================

`gammapy.irf` contains classes for handling Instrument Response
Functions typically stored as multi-dimensional tables. For a list of
IRF classes internally supported, see
https://gamma-astro-data-formats.readthedocs.io/en/v0.3/irfs/full_enclosure/index.html

This tutorial is intended for advanced users typically creating IRFs

"""

import numpy as np
import scipy.special
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.irf import (
    IRF,
    Background3D,
    EffectiveAreaTable2D,
    EnergyDependentMultiGaussPSF,
    EnergyDispersion2D,
)
from gammapy.irf.io import COMMON_IRF_HEADERS, IRF_DL3_HDU_SPECIFICATION
from gammapy.makers.utils import make_edisp_kernel_map, make_map_exposure_true_energy
from gammapy.maps import MapAxes, MapAxis, WcsGeom

######################################################################
# Inbuilt Gammapy IRFs
# --------------------
#

irf_filename = (
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

aeff = EffectiveAreaTable2D.read(irf_filename, hdu="EFFECTIVE AREA")
print(aeff)


######################################################################
# We can see that the Effective Area Table is defined in terms of
# “energy_true” and `offset` from the camera center
#

# To see the IRF axes binning, eg, offset
print(aeff.axes["offset"])

# To get the IRF data
print(aeff.data)

# the aeff is evaluated at a given energy and offset
print(aeff.evaluate(energy_true=[1, 10] * u.TeV, offset=[0.2, 2.5] * u.deg))


# The peek methods give quick looks into the irfs
aeff.peek()


######################################################################
# Similarly, we can access other IRFs as well
#

psf = EnergyDependentMultiGaussPSF.read(irf_filename, hdu="Point Spread Function")
bkg = Background3D.read(irf_filename, hdu="BACKGROUND")
edisp = EnergyDispersion2D.read(irf_filename, hdu="ENERGY DISPERSION")

print(bkg)

# To evaluate the irfs, pass the values for each axis
ev = bkg.evaluate(energy=1 * u.TeV, fov_lon=[1, 2] * u.deg, fov_lat=[1, 2] * u.deg)
print(ev)

# To evaluate the irfs, pass the values for each axis
ev = edisp.evaluate(energy_true=1 * u.TeV, offset=[0, 1] * u.deg, migra=[1, 1.2])
print(ev)

edisp.peek()

print(psf)


######################################################################
# The PointSpreadFunction for the CTA DC1 is stored as a combination of 3
# Gaussians. Other PSFs, like a `PSF_TABLE` and analytic expressions
# like KING function are also supported. All PSF claases inherit from a
# common base `PSF` class
#

print(psf.axes.names)

# To get the containment radius for a fixed fraction at a given position
print(
    psf.containment_radius(
        fraction=0.68, energy_true=1.0 * u.TeV, offset=[0.2, 4.0] * u.deg
    )
)

# Alternatively, to get the containment fraction for at a given position
print(
    psf.containment(
        rad=0.05 * u.deg, energy_true=1.0 * u.TeV, offset=[0.2, 4.0] * u.deg
    )
)


######################################################################
# Support for Asymmetric IRFs
# ---------------------------
#


######################################################################
# While Gammapy does not have inbuilt classes for supporting asymmetric
# IRFs (except for `Background3D`), custom classes can be created. For
# this to work correctly with the `MapDatasetMaker`, only variations
# with `fov_lon` and `fov_lat` can be allowed.
#
# The main idea is that the list of required axes should be correctly
# mentioned in the class definition.
#
# Asymmetric PSFs are not correctly supported at present in the data
# reduction scheme.
#


######################################################################
# Effective Area
# ~~~~~~~~~~~~~~
#


class EffectiveArea3D(IRF):
    tag = "aeff_3d"
    required_axes = ["energy_true", "fov_lon", "fov_lat"]
    default_unit = u.m**2


energy_axis = MapAxis.from_energy_edges(
    [0.1, 0.3, 1.0, 3.0, 10.0] * u.TeV, name="energy_true"
)

nbin = 7
fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

data = np.ones((4, 3, 3))
for i in range(1, 4):
    data[i] = data[i - 1] * 2.0
    data[i][-1] = data[i][0] * 1.2


aeff_3d = EffectiveArea3D(
    [energy_axis, fov_lon_axis, fov_lat_axis], data=data, unit=u.m**2
)
print(aeff_3d)

res = aeff_3d.evaluate(
    fov_lon=[-0.5, 0.8] * u.deg,
    fov_lat=[-0.5, 1.0] * u.deg,
    energy_true=[0.2, 8.0] * u.TeV,
)
print(res)

# to visualise at a given energy
# sphinx_gallery_thumbnail_number = 1
aeff_eval = aeff_3d.evaluate(energy_true=[1.0] * u.TeV)

ax = plt.subplot()
with quantity_support():
    caxes = ax.pcolormesh(
        fov_lat_axis.edges, fov_lon_axis.edges, aeff_eval.value.squeeze()
    )
fov_lat_axis.format_plot_xaxis(ax)
fov_lon_axis.format_plot_yaxis(ax)
ax.set_title("Asymmetric effective area")


######################################################################
# Serialisation
# ~~~~~~~~~~~~~
#
# For serialisation, we need to add the class definition to the
# `IRF_DL3_HDU_SPECIFICATION` dictionary
#

IRF_DL3_HDU_SPECIFICATION["aeff_3d"] = {
    "extname": "EFFECTIVE AREA",
    "column_name": "EFFAREA",
    "mandatory_keywords": {
        **COMMON_IRF_HEADERS,
        "HDUCLAS2": "EFF_AREA",
        "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
        "HDUCLAS4": "AEFF_3D",
    },
}

aeff_3d.write("test_aeff3d.fits", overwrite=True)

aeff_new = EffectiveArea3D.read("test_aeff3d.fits")
print(aeff_new)


######################################################################
# Create exposure map
# ~~~~~~~~~~~~~~~~~~~
#
# DL4 data products can be created from these IRFs.
#

axis = MapAxis.from_energy_bounds(0.1 * u.TeV, 10 * u.TeV, 6, name="energy_true")
pointing = SkyCoord(2, 1, unit="deg")
geom = WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis], skydir=pointing)

print(geom)

exposure_map = make_map_exposure_true_energy(
    pointing=pointing, livetime="42 h", aeff=aeff_3d, geom=geom
)

exposure_map.plot_grid(add_cbar=True)


######################################################################
# Energy Dispersion
# -----------------
#


class EnergyDispersion3D(IRF):
    tag = "edisp_3d"
    required_axes = ["energy_true", "migra", "fov_lon", "fov_lat"]
    default_unit = u.one

    @classmethod
    def from_gauss(
        cls,
        energy_axis_true,
        migra_axis,
        fov_lon_axis,
        fov_lat_axis,
        bias,
        sigma,
        pdf_threshold=1e-6,
    ):
        axes = MapAxes([energy_axis_true, migra_axis, fov_lon_axis, fov_lat_axis])
        coords = axes.get_coord(mode="edges", axis_name="migra")

        migra_min = coords["migra"][:, :-1, :]
        migra_max = coords["migra"][:, 1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra_max - 1 - bias) / s
        t2 = (migra_min - 1 - bias) / s
        pdf = (scipy.special.erf(t1) - scipy.special.erf(t2)) / 2
        pdf = pdf / (migra_max - migra_min)

        r1 = np.rollaxis(pdf, -1, 1)
        r2 = np.rollaxis(r1, 0, -1)
        data = r2 * np.ones(axes.shape)

        data[data < pdf_threshold] = 0

        return cls(
            axes=axes,
            data=data.value,
        )


# Make a test case
energy_axis_true = MapAxis.from_energy_bounds(
    "0.1 TeV", "100 TeV", nbin=3, name="energy_true"
)

migra_axis = MapAxis.from_bounds(0, 4, nbin=2, node_type="edges", name="migra")

fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

data = np.array(
    [
        [
            [
                [4.99582964e-01, 4.99582964e-01, 4.99582964e-01],
                [4.99582964e-01, 4.99582964e-01, 4.99582964e-01],
                [4.99582964e-01, 4.99582964e-01, 4.99582964e-01],
            ],
            [
                [2.06259405e-04, 2.06259405e-04, 2.06259405e-04],
                [2.06259405e-04, 2.06259405e-04, 2.06259405e-04],
                [2.06259405e-04, 2.06259405e-04, 2.06259405e-04],
            ],
        ],
        [
            [
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
            ],
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ],
        ],
        [
            [
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
                [5.00000000e-01, 5.00000000e-01, 5.00000000e-01],
            ],
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ],
        ],
    ]
)


edisp3d = EnergyDispersion3D(
    [energy_axis_true, migra_axis, fov_lon_axis, fov_lat_axis], data=data
)

print(edisp3d)

energy = [1, 2] * u.TeV
migra = np.array([0.98, 0.97, 0.7])
fov_lon = [0.1, 1.5] * u.deg
fov_lat = [0.0, 0.3] * u.deg

edisp3d.evaluate(
    energy_true=energy.reshape(-1, 1, 1, 1),
    migra=migra.reshape(1, -1, 1, 1),
    fov_lon=fov_lon.reshape(1, 1, -1, 1),
    fov_lat=fov_lat.reshape(1, 1, 1, -1),
)[0][2]


######################################################################
# Serialisation
# ~~~~~~~~~~~~~
#

IRF_DL3_HDU_SPECIFICATION["edisp_3d"] = {
    "extname": "ENERGY DISPERSION",
    "column_name": "MATRIX",
    "mandatory_keywords": {
        **COMMON_IRF_HEADERS,
        "HDUCLAS2": "EDISP",
        "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
        "HDUCLAS4": "EDISP_3D",
    },
}

edisp3d.write("test_edisp.fits", overwrite=True)

edisp_new = EnergyDispersion3D.read("test_edisp.fits")
edisp_new


######################################################################
# Create edisp kernel map
# ~~~~~~~~~~~~~~~~~~~~~~~
#

migra = MapAxis.from_edges(np.linspace(0.5, 1.5, 50), unit="", name="migra")
etrue = MapAxis.from_energy_bounds(0.5, 2, 6, unit="TeV", name="energy_true")
ereco = MapAxis.from_energy_bounds(0.5, 2, 3, unit="TeV", name="energy")
geom = WcsGeom.create(10, binsz=0.5, axes=[ereco, etrue], skydir=pointing)

edispmap = make_edisp_kernel_map(edisp3d, pointing, geom)

edispmap.peek()

edispmap.edisp_map.data[3][1][3]


######################################################################
# PSF
# ---
#
# A higher dimensional PSF Table can also be created, as shown here.
# Support for higher dimensional analystic PSF will come later
#


class PSFnD(IRF):
    tag = "psf_nd"
    required_axes = ["energy_true", "fov_lon", "fov_lat", "rad"]
    default_unit = u.sr**-1


energy_axis = MapAxis.from_energy_edges(
    [0.1, 0.3, 1.0, 3.0, 10.0] * u.TeV, name="energy_true"
)

nbin = 7
fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

rad_axis = MapAxis.from_edges([0, 1, 2], unit="deg", name="rad")

data = 0.1 * np.ones((4, 3, 3, 2))
for i in range(1, 4):
    data[i] = data[i - 1] * 1.5

psfnd = PSFnD(
    axes=[energy_axis, fov_lon_axis, fov_lat_axis, rad_axis],
    data=data,
)
print(psfnd)

energy = [1, 2] * u.TeV
rad = np.array([0.98, 0.97, 0.7]) * u.deg
fov_lon = [0.1, 1.5] * u.deg
fov_lat = [0.0, 0.3] * u.deg

psfnd.evaluate(
    energy_true=energy.reshape(-1, 1, 1, 1),
    rad=rad.reshape(1, -1, 1, 1),
    fov_lon=fov_lon.reshape(1, 1, -1, 1),
    fov_lat=fov_lat.reshape(1, 1, 1, -1),
)


######################################################################
# Serialisation
# ~~~~~~~~~~~~~
#

IRF_DL3_HDU_SPECIFICATION["psf_nd"] = {
    "extname": "POINT SPREAD FUNCTION",
    "column_name": "MATRIX",
    "mandatory_keywords": {
        **COMMON_IRF_HEADERS,
        "HDUCLAS2": "PSF",
        "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
        "HDUCLAS4": "PSFnD",
    },
}

psfnd.write("test_psf.fits.gz", overwrite=True)

psf_new = PSFnD.read("test_psf.fits.gz")

psf_new == psfnd


######################################################################
# Support for asymmetric IRFs is priliminary at the moment, and will
# evolve depending on feedback.
#
