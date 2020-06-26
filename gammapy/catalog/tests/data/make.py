"""Make test reference data files."""

from gammapy.catalog import CATALOG_REGISTRY

cat = CATALOG_REGISTRY["3fgl"]()
open("3fgl_J0000.1+6545.txt", "w").write(str(cat["3FGL J0000.1+6545"]))
open("3fgl_J0001.4+2120.txt", "w").write(str(cat["3FGL J0001.4+2120"]))
open("3fgl_J0023.4+0923.txt", "w").write(str(cat["3FGL J0023.4+0923"]))
open("3fgl_J0835.3-4510.txt", "w").write(str(cat["3FGL J0835.3-4510"]))

cat = CATALOG_REGISTRY["4fgl"]()
open("4fgl_J0000.3-7355.txt", "w").write(str(cat["4FGL J0000.3-7355"]))
open("4fgl_J0001.5+2113.txt", "w").write(str(cat["4FGL J0001.5+2113"]))
open("4fgl_J0002.8+6217.txt", "w").write(str(cat["4FGL J0002.8+6217"]))
open("4fgl_J1409.1-6121e.txt", "w").write(str(cat["4FGL J1409.1-6121e"]))

cat = CATALOG_REGISTRY["2fhl"]()
open("2fhl_j1445.1-0329.txt", "w").write(str(cat["2FHL J1445.1-0329"]))
open("2fhl_j0822.6-4250e.txt", "w").write(str(cat["2FHL J0822.6-4250e"]))

cat = CATALOG_REGISTRY["3fhl"]()
open("3fhl_j2301.9+5855e.txt", "w").write(str(cat["3FHL J2301.9+5855e"]))

cat = CATALOG_REGISTRY["2hwc"]()
open("2hwc_j0534+220.txt", "w").write(str(cat["2HWC J0534+220"]))
open("2hwc_j0631+169.txt", "w").write(str(cat["2HWC J0631+169"]))

cat = CATALOG_REGISTRY["hgps"]()
open("hess_j1713-397.txt", "w").write(str(cat["HESS J1713-397"]))
open("hess_j1825-137.txt", "w").write(str(cat["HESS J1825-137"]))
open("hess_j1930+188.txt", "w").write(str(cat["HESS J1930+188"]))

cat = CATALOG_REGISTRY["gamma-cat"]()
open("gammacat_hess_j1813-178.txt", "w").write(str(cat["HESS J1813-178"]))
open("gammacat_hess_j1848-018.txt", "w").write(str(cat["HESS J1848-018"]))
open("gammacat_vela_x.txt", "w").write(str(cat["Vela X"]))
