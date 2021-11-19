"""Example plot showing stacking of two datasets."""

from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

region = CircleSkyRegion(
    center=SkyCoord(0, 0, unit="deg", frame="galactic"), radius=0.3 * u.deg
)
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
ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)

dataset1.edisp.get_edisp_kernel().plot_matrix(ax=ax1)
ax1.set_title("Energy dispersion dataset1")

dataset2.edisp.get_edisp_kernel().plot_matrix(ax=ax2)
ax2.set_title("Energy dispersion dataset2")

dataset_stacked.edisp.get_edisp_kernel().plot_matrix(ax=ax3)
ax3.set_title("Energy dispersion stacked")

n1 = dataset1.npred() * dataset1.mask_safe
n2 = dataset2.npred() * dataset2.mask_safe
(n1).plot_hist(label="dataset1", ax=ax4)
(n2).plot_hist(label="dataset2", ax=ax4)
(n1 + n2).plot(color="green", fmt="o", label="dataset1 + dataset2", ax=ax4)
dataset_stacked.npred().plot(fmt="+", label="stacked", color="black", ax=ax4)
ax4.legend()
ax4.set_title("npred")
