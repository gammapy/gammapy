"""Example plot showing stacking of two datasets."""

from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from gammapy.data import Observation, observatory_locations
from gammapy.datasets import SpectrumDataset
from gammapy.datasets.map import MIGRA_AXIS_DEFAULT
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

energy_true = MapAxis.from_energy_bounds(
    "0.1 TeV", "20 TeV", nbin=20, per_decade=True, name="energy_true"
)
energy_reco = MapAxis.from_energy_bounds("0.2 TeV", "10 TeV", nbin=10, per_decade=True)

aeff = EffectiveAreaTable2D.from_parametrization(
    energy_axis_true=energy_true, instrument="HESS"
)
offset_axis = MapAxis.from_bounds(0 * u.deg, 5 * u.deg, nbin=2, name="offset")

edisp = EnergyDispersion2D.from_gauss(
    energy_axis_true=energy_true,
    offset_axis=offset_axis,
    migra_axis=MIGRA_AXIS_DEFAULT,
    bias=0,
    sigma=0.2,
)

observation = Observation.create(
    obs_id=0,
    pointing=SkyCoord("0d", "0d", frame="icrs"),
    irfs={"aeff": aeff, "edisp": edisp},
    tstart=0 * u.h,
    tstop=0.5 * u.h,
    location=observatory_locations["hess"],
)

geom = RegionGeom.create("icrs;circle(0, 0, 0.1)", axes=[energy_reco])

stacked = SpectrumDataset.create(geom=geom, energy_axis_true=energy_true)

maker = SpectrumDatasetMaker(selection=["edisp", "exposure"])

dataset_1 = maker.run(stacked.copy(), observation=observation)
dataset_2 = maker.run(stacked.copy(), observation=observation)

pwl = PowerLawSpectralModel()
model = SkyModel(spectral_model=pwl, name="test-source")

dataset_1.mask_safe = geom.energy_mask(energy_min=2 * u.TeV)
dataset_2.mask_safe = geom.energy_mask(energy_min=0.6 * u.TeV)

dataset_1.models = model
dataset_2.models = model
dataset_1.counts = dataset_1.npred()
dataset_2.counts = dataset_2.npred()

stacked = dataset_1.copy(name="stacked")
stacked.stack(dataset_2)

stacked.models = model
npred_stacked = stacked.npred()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axes[0].set_title("Stacked Energy Dispersion Matrix")
axes[1].set_title("Predicted Counts")
stacked.edisp.get_edisp_kernel().plot_matrix(ax=axes[0])
npred_stacked.plot_hist(ax=axes[1], label="npred stacked")
stacked.counts.plot_hist(ax=axes[1], ls="--", label="stacked npred")
plt.legend()
plt.show()
