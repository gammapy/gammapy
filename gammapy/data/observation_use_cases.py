""" Now we can build an IACT observation from scratch (programmatically):
-------------------------------------------------------------------------
"""

from gammapy.data import EventList, GTI
from gammapy.irf import EnergyDependentMultiGaussPSF, EffectiveAreaTable2D, EnergyDispersion2D, Background3D
# ObservationIACT is a child class of a more generic Observation class.
# We could use it for different Observation types: ObservationFermi, ObservationHAWC in the future.
from gammapy.data import ObservationIACT

filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
bkg = Background3D.read(filename, hdu='BACKGROUND')

my_events = EventList.read(...)
my_gti = GTI.read(...)

my_obs = ObservationIACT(events=my_events, gti=my_gti, psf=psf, aeff=aeff, edisp=edisp, bkg=bkg)

# now we can modify the objects in myObs or replace them:
my_obs.events.select_time(time_intervall)
my_obs.gti = my_modified_gti

# Thoughts: In a next step i would like to factor out the IRF to its own namespace,
# so each Observation object would hold an IRFS object:
my_obs = Observation(events=my_events, irfs=my_irfs)
my_obs.irfs.aeff
# Looks somehow cleaner to me and maybe in the future we could support multiple IRF in one observation.
# Each IRF container would carry a GTI object to state its valid time intervals. Something like:
my_obs.get_irfs(time).aeff


""" There is new namespace for metadata with the ObservationMeta class.
-----------------------------------------------------------------------
"""

my_obs = ObservationIACT(events=my_events, my_metadata='Best Obs ever')
my_obs.meta.my_metadata
# Not sure if this is 100% necessary, it just helps a bit to keep the namespace cleaner ...


""" Creating observations from a DataStore object
-------------------------------------------------
"""

from gammapy.data import DataStore
from gammapy.data import ObservationIACTMaker

data_store = DataStore.from_dir('my_dir')
my_obs = ObservationIACTMaker.from_data_store(data_store, obs_id=1)
# or
my_obs = ObservationIACT.from_data_store(data_store, obs_id=1)  # just a convenient method that calls the ObservationIACTMaker ...
type(my_obs) -> gammapy.data.ObservationIACT

# If we want to choose not to load the event list into memory we can do so.
# Less memory usage, but like this we cannot modify or replace the event list.
my_obs = ObservationIACTMaker.from_data_store(data_store, obs_id=1, link_data_store=True)
type(my_obs) -> gammapy.data.ObservationIACTLinked  # this is a child class of ObservationIACT
my_obs.events.select_time(time_intervall)  # this has no effect
my_obs.events = my_other_events  # no effect either


""" A new generic Checker class, i propose to adapt the other checkers to follow its scheme
-------------------------------------------------------------------------------------------
"""

# This checker is a mix of implementations we have in Gammapy right now.
# I think it is a neat implementation since it allows to do checks recursively:
# the ObservationChecker calls the IRFChekcer calls the AEffChecker ...
# Maybe the other checkers we have right now could follow its scheme :)
# I would be willing to do the code changes of course!
from gammapy.data import ObservationChecker

my_obs_checker = ObservationChecker(my_obs)
my_obs_checker.available_checks() # returns a list of strings
result = my_obs_checker.run(checks=['event_list', 'aeff'])  # returns a dict with a summarizing `result['status']`
# or
result = my_obs_checker.run(checks='all')
# or
my_obs.check_observation()  # convenient method





