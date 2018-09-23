"""Example of how to create an ObservationCTA from CTA's 1DC"""
from gammapy.data import ObservationCTA, EventList, GTI
from gammapy.irf import (
    EnergyDependentMultiGaussPSF,
    EffectiveAreaTable2D,
    EnergyDispersion2D,
    Background3D,
)

filename = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
event_list = EventList.read(filename)
gti = GTI.read(filename)

filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
aeff = EffectiveAreaTable2D.read(filename)
bkg = Background3D.read(filename)
edisp = EnergyDispersion2D.read(filename, hdu="Energy Dispersion")
psf = EnergyDependentMultiGaussPSF.read(filename, hdu="Point Spread Function")

obs = ObservationCTA(
    obs_id=event_list.table.meta["OBS_ID"],
    events=event_list,
    gti=gti,
    psf=psf,
    aeff=aeff,
    edisp=edisp,
    bkg=bkg,
    pointing_radec=event_list.pointing_radec,
    observation_live_time_duration=event_list.observation_live_time_duration,
    observation_dead_time_fraction=event_list.observation_dead_time_fraction,
)

print(obs)
