"""Make test reference data files."""

from gammapy.catalog import source_catalogs

cat = source_catalogs["3fgl"]
open("3fgl_J0000.1+6545.txt", "w").write(str(cat["3FGL J0000.1+6545"]))
open("3fgl_J0001.4+2120.txt", "w").write(str(cat["3FGL J0001.4+2120"]))
open("3fgl_J0023.4+0923.txt", "w").write(str(cat["3FGL J0023.4+0923"]))
open("3fgl_J0835.3-4510.txt", "w").write(str(cat["3FGL J0835.3-4510"]))

cat = source_catalogs["3fhl"]
open("3fhl_j2301.9+5855e.txt", "w").write(str(cat["3FHL J2301.9+5855e"]))

cat = source_catalogs["2hwc"]
open("2hwc_j0534+220.txt", "w").write(str(cat["2HWC J0534+220"]))
open("2hwc_j0631+169.txt", "w").write(str(cat["2HWC J0631+169"]))

cat = source_catalogs["gamma-cat"]
open("gammacat_hess_j1813-178.txt", "w").write(str(cat["HESS J1813-178"]))
open("gammacat_hess_j1848-018.txt", "w").write(str(cat["HESS J1848-018"]))
open("gammacat_vela_x.txt", "w").write(str(cat["Vela X"]))
