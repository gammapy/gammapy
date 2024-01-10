"""
Using asymmetric IRFs
=====================

"""


######################################################################
# While Gammapy does not have inbuilt classes for supporting asymmetric
# IRFs (except for `Background3D`), custom classes can be created. For
# this to work correctly with the `MapDatasetMaker`, only variations
# with `fov_lon` and `fov_lat` can be allowed.
#
# Analytic models for the PSF, or anisotropic PSFs (ie, asymmetry around
# the source position) is not correctly supported during the data
# reduction
#

import numpy as np
import scipy.special
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import *
from gammapy.irf.io import COMMON_IRF_HEADERS, IRF_DL3_HDU_SPECIFICATION
from gammapy.makers.utils import *
from gammapy.maps import MapAxes, MapAxis, WcsGeom

######################################################################
# Effective Area
# --------------
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
    data[i] = data[i - 1] * 1.5

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


######################################################################
# Serialisation
# ~~~~~~~~~~~~~
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

axis = MapAxis.from_energy_bounds(0.1 * u.TeV, 10 * u.TeV, 6, name="energy_true")
pointing = SkyCoord(2, 1, unit="deg")
geom = WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis], skydir=pointing)

print(geom)

exposure_map = make_map_exposure_true_energy(
    pointing=pointing, livetime="42 h", aeff=aeff_3d, geom=geom
)


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
    "0.1 TeV", "100 TeV", nbin=50, name="energy_true"
)

migra_axis = MapAxis.from_bounds(0, 4, nbin=100, node_type="edges", name="migra")

fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

energy_true = energy_axis_true.edges[:-1]
sigma = 0.15 / (energy_true / (1 * u.TeV)).value ** 0.3
bias = 1e-3 * (energy_true - 1 * u.TeV).value

edisp3d = EnergyDispersion3D.from_gauss(
    energy_axis_true=energy_axis_true,
    migra_axis=migra_axis,
    fov_lon_axis=fov_lon_axis,
    fov_lat_axis=fov_lat_axis,
    bias=bias,
    sigma=sigma,
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
)


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


######################################################################
# PSF
# ---
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
