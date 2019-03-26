Datasets
========

This directory is a place for the index datasets file ``gammapy-data-index.json`` listing the datasets needed
in the development version of Gammapy. This file is generated executing ``python make_dataset_index.py all``
from the command line and needs some environment variables declared (i.e. GAMMAPY_EXTRA, JOINT_CRAB,
GAMMA_CAT, GAMMAPY_FERMI_LAT_DATA) storing the local paths for the different datasets. For the moment these
paths point to local folders pulled from different GitHub repositories, it is recommended to update the local
content of these paths before generating the index datasets file.
