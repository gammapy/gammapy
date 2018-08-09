"""Example of how to create an ObservationCTA from CTA's 1DC"""
from gammapy.utils.scripts import make_path
from gammapy.data import ObservationCTA, EventList, GTI
from gammapy.irf import EnergyDependentMultiGaussPSF, EffectiveAreaTable2D, EnergyDispersion2D, Background3D

data_file = make_path('$GAMMAPY_EXTRA/datasets/cta-1dc/data/baseline/gps/gps_baseline_110380.fits')
cal_file = make_path('$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits')

event_list = EventList.read(data_file)

kwargs = dict(obs_id=event_list.table.meta['OBS_ID'],
              events=event_list,
              gti=GTI.read(data_file),
              psf=EnergyDependentMultiGaussPSF.read(cal_file, hdu='Point Spread Function'),
              aeff=EffectiveAreaTable2D.read(cal_file),
              edisp=EnergyDispersion2D.read(cal_file, hdu='Energy Dispersion'),
              bkg=Background3D.read(cal_file),
              pointing_radec=event_list.pointing_radec,
              observation_live_time_duration=event_list.observation_live_time_duration,
              observation_dead_time_fraction=event_list.observation_dead_time_fraction,
              )

obs = ObservationCTA(**kwargs)
print(obs)
