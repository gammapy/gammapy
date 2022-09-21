"""
Multi instrument joint 3D and 1D analysis
=========================================

Joint 3D analysis using 3D Fermi datasets, a H.E.S.S. reduced spectrum and HAWC flux points.

Prerequisites
-------------

-  Handling of Fermi-LAT data with gammapy `see the corresponding
   tutorial <../../data/fermi_lat.ipynb>`__
-  Knowledge of spectral analysis to produce 1D On-Off datasets, `see
   the following tutorial <../1D/spectral_analysis.ipynb>`__
-  Using flux points to directly fit a model (without forward-folding)
   `see the SED fitting tutorial <../1D/sed_fitting.ipynb>`__

Context
-------

Some science studies require to combine heterogeneous data from various
instruments to extract physical information. In particular, it is often
useful to add flux measurements of a source at different energies to an
analysis to better constrain the wide-band spectral parameters. This can
be done using a joint fit of heterogeneous datasets.

**Objectives: Constrain the spectral parameters of the gamma-ray
emission from the Crab nebula between 10 GeV and 100 TeV, using a 3D
Fermi dataset, a H.E.S.S. reduced spectrum and HAWC flux points.**

Proposed approach
-----------------

This tutorial illustrates how to perform a joint modeling and fitting of
the Crab Nebula spectrum using different datasets. The spectral
parameters are optimized by combining a 3D analysis of Fermi-LAT data, a
ON/OFF spectral analysis of HESS data, and flux points from HAWC.

In this tutorial we are going to use pre-made datasets. We prepared maps
of the Crab region as seen by Fermi-LAT using the same event selection
than the `3FHL catalog <https://arxiv.org/abs/1702.00664>`__ (7 years of
data with energy from 10 GeV to 2 TeV). For the HESS ON/OFF analysis we
used two observations from the `first public data
release <https://arxiv.org/abs/1810.04516>`__ with a significant signal
from energy of about 600 GeV to 10 TeV. These observations have an
offset of 0.5° and a zenith angle of 45-48°. The HAWC flux points data
are taken from a `recent
analysis <https://arxiv.org/pdf/1905.12518.pdf>`__ based on 2.5 years of
data with energy between 300 Gev and 300 TeV.

The setup
---------

"""

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling import Fit
from gammapy.modeling.models import Models, create_crab_spectral_model
from gammapy.datasets import Datasets, FluxPointsDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from gammapy.maps import MapAxis
from gammapy.utils.scripts import make_path
from pathlib import Path

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Data and models files
# ---------------------
# 
# The datasets serialization produce YAML files listing the datasets and
# models. In the following cells we show an example containning only the
# Fermi-LAT dataset and the Crab model.
# 
# Fermi-LAT-3FHL_datasets.yaml:
# 

path = make_path("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml")

with path.open("r") as f:
    print(f.read())


######################################################################
# We used as model a point source with a log-parabola spectrum. The
# initial parameters were taken from the latest Fermi-LAT catalog
# `4FGL <https://arxiv.org/abs/1902.10045>`__, then we have re-optimized
# the spectral parameters for our dataset in the 10 GeV - 2 TeV energy
# range (fixing the source position).
# 
# Fermi-LAT-3FHL_models.yaml:
# 

path = make_path("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml")

with path.open("r") as f:
    print(f.read())


######################################################################
# Reading different datasets
# --------------------------
# 
# Fermi-LAT 3FHL: map dataset for 3D analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For now we let’s use the datasets serialization only to read the 3D
# `MapDataset` associated to Fermi-LAT 3FHL data and models.
# 

path = Path("$GAMMAPY_DATA/fermi-3fhl-crab")
filename = path / "Fermi-LAT-3FHL_datasets.yaml"

datasets = Datasets.read(filename=filename)

models = Models.read(path / "Fermi-LAT-3FHL_models.yaml")
print(models)


######################################################################
# We get the Crab model in order to share it with the other datasets
# 

print(models["Crab Nebula"])


######################################################################
# HESS-DL3: 1D ON/OFF dataset for spectral fitting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The ON/OFF datasets can be read from PHA files following the `OGIP
# standards <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html>`__.
# We read the PHA files from each observation, and compute a stacked
# dataset for simplicity. Then the Crab spectral model previously defined
# is added to the dataset.
# 

datasets_hess = Datasets()

for obs_id in [23523, 23526]:
    dataset = SpectrumDatasetOnOff.read(
        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    )
    datasets_hess.append(dataset)

dataset_hess = datasets_hess.stack_reduce(name="HESS")

datasets.append(dataset_hess)

print(datasets)


######################################################################
# HAWC: 1D dataset for flux point fitting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The HAWC flux point are taken from https://arxiv.org/pdf/1905.12518.pdf
# Then these flux points are read from a pre-made FITS file and passed to
# a `FluxPointsDataset` together with the source spectral model.
# 

# read flux points from https://arxiv.org/pdf/1905.12518.pdf
filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
flux_points_hawc = FluxPoints.read(
    filename, reference_model=create_crab_spectral_model("meyer")
)

dataset_hawc = FluxPointsDataset(data=flux_points_hawc, name="HAWC")

datasets.append(dataset_hawc)

print(datasets)


######################################################################
# Datasets serialization
# ----------------------
# 
# The `datasets` object contains each dataset previously defined. It can
# be saved on disk as datasets.yaml, models.yaml, and several data files
# specific to each dataset. Then the `datasets` can be rebuild later
# from these files.
# 

path = Path("crab-3datasets")
path.mkdir(exist_ok=True)

filename = path / "crab_10GeV_100TeV_datasets.yaml"

datasets.write(filename, overwrite=True)

datasets = Datasets.read(filename)
datasets.models = models

print(datasets)


######################################################################
# Joint analysis
# --------------
# 
# We run the fit on the `Datasets` object that include a dataset for
# each instrument
# 

# %%time
fit_joint = Fit()
results_joint = fit_joint.run(datasets=datasets)
print(results_joint)


######################################################################
# Let’s display only the parameters of the Crab spectral model
# 

crab_spec = datasets[0].models["Crab Nebula"].spectral_model
print(crab_spec)


######################################################################
# We can compute flux points for Fermi-LAT and HESS datasets in order plot
# them together with the HAWC flux point.
# 

# compute Fermi-LAT and HESS flux points
energy_edges = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=5).edges

flux_points_fermi = FluxPointsEstimator(
    energy_edges=energy_edges,
    source="Crab Nebula",
).run([datasets["Fermi-LAT"]])


energy_edges = MapAxis.from_bounds(
    1, 15, nbin=6, interp="log", unit="TeV"
).edges

flux_points_hess = FluxPointsEstimator(
    energy_edges=energy_edges, source="Crab Nebula", selection_optional=["ul"]
).run([datasets["HESS"]])


######################################################################
# Now, Let’s plot the Crab spectrum fitted and the flux points of each
# instrument.
# 

# display spectrum and flux points
plt.figure(figsize=(8, 6))

energy_bounds = [0.01, 300] * u.TeV
sed_type = "e2dnde"

ax = crab_spec.plot(
    energy_bounds=energy_bounds, sed_type=sed_type, label="Model"
)
crab_spec.plot_error(ax=ax, energy_bounds=energy_bounds, sed_type=sed_type)

flux_points_fermi.plot(ax=ax, sed_type=sed_type, label="Fermi-LAT")
flux_points_hess.plot(ax=ax, sed_type=sed_type, label="HESS")
flux_points_hawc.plot(ax=ax, sed_type=sed_type, label="HAWC")

ax.set_xlim(energy_bounds)
plt.legend();

