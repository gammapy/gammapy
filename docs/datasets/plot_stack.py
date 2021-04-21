"""Example plot showing stacking of two datasets."""

import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.datasets import MapDataset, Datasets
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

region = CircleSkyRegion(center=SkyCoord(0, 0, unit="deg", frame="galactic"), radius=0.3 * u.deg)
m_dataset1 = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
dataset1 = m_dataset1.to_spectrum_dataset(region)
dataset1.mask_safe.data[0:2] = False

dataset2 = m_dataset1.to_spectrum_dataset(region)
dataset2.mask_safe.data[1:3] = False
dataset2.exposure = dataset2.exposure / 2.0
dataset2.edisp.exposure_map = dataset2.edisp.exposure_map / 2.0
dataset2.counts = dataset2.counts / 2.0
dataset2.background = dataset2.background / 2.0

datasets = Datasets([dataset1, dataset2])
dataset_stacked = datasets.stack_reduce()

pwl = PowerLawSpectralModel(index=4)
model = SkyModel(spectral_model=pwl, name="test")
datasets.models = model
dataset_stacked.models = model


plt.figure(figsize=(20, 5))
ax1 = plt.subplot(131, projection=dataset_stacked._geom.wcs)
ax2 = plt.subplot(132, projection=dataset_stacked._geom.wcs)
ax3 = plt.subplot(133, projection=dataset_stacked._geom.wcs)

c1 = (dataset1.counts * dataset1.mask_safe)
c2 = (dataset2.counts * dataset2.mask_safe)

c1.plot_hist(ax=ax1, label="dataset1")
c2.plot_hist(ax=ax1, label="dataset2")
(c1 + c2).plot(color="green", ax=ax1, fmt="o", label="dataset1 + dataset2")
dataset_stacked.counts.plot(ax=ax1, fmt="+", label="stacked", color="black")
ax1.legend()
ax1.set_title("counts")

b1 = (dataset1.background * dataset1.mask_safe)
b2 = (dataset2.background * dataset2.mask_safe)
b1.plot_hist(ax=ax2, label="dataset1")
b2.plot_hist(ax=ax2, label="dataset2")
(b1 + b2
).plot(color="green", ax=ax2, fmt="o", label="dataset1 + dataset2")
dataset_stacked.background.plot(
    ax=ax2, fmt="+", label="stacked", color="black"
)
ax2.legend()
ax2.set_title("background")

n1 = dataset1.npred() * dataset1.mask_safe
n2 = dataset2.npred() * dataset2.mask_safe
(n1).plot_hist(label="dataset1", ax=ax3)
(n2).plot_hist(label="dataset2", ax=ax3)
(n1+n2).plot(color="green", fmt="o", label="dataset1 + dataset2", ax=ax3)
dataset_stacked.npred().plot(fmt="+", label="stacked", color="black", ax=ax3)
ax3.legend()
ax3.set_title("npred")


