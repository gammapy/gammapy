import numpy as np
from gammapy.utils.time import time_ref_from_dict
from astropy.units import Quantity
import astropy.units as u
from astropy.table import Column
from gammapy.time import LightCurve, LightCurveEstimator
from gammapy.spectrum import (
    models, SpectrumExtraction, SpectrumObservationList, SpectrumFit, SpectrumResult,
)
from gammapy.background import RingBackgroundEstimator, ReflectedRegionsBackgroundEstimator

__all__ = [
    'PhaseCurve',
]

#log = logging.getLogger(__name__)


class PhaseCurve(object):

      def __init__(self,obs_list,period,T0,model,energy_range):
          self.obs_list = obs_list
          self.period = period
          self.T0  = T0
          self.model = model
          self.energy_range = energy_range 
          
      def phase_to_obs(self,table):
          met_ref = time_ref_from_dict(table.meta)
          met = Quantity(table['TSTART'].astype('float64'), 'second')
          tstart = met_ref + met
          met = Quantity(table['TSTOP'].astype('float64'), 'second')
          tstop = met_ref + met
          time_mjd = (tstart.value + tstop.value)/2
          Tp = self.period
          T0 = self.T0
          phase_mid = (time_mjd - T0)/Tp - np.int_((time_mjd - T0)/Tp)
          aa = Column(phase_mid, name='PHASE_MID')
          table.add_column(aa, index=0)
          return table
      
      def phasewise_obs_list(self,table):
          obs_ids=[]
          phase_bin = np.arange(11)*0.1
          for i in range(10):
              obs_ids.append([row['OBS_ID'] for row in table if (row['PHASE_MID'] < phase_bin[i+1] and row['PHASE_MID'] > phase_bin[i])]) 
          return obs_ids

      def intervals(self,obs_list):
            intervals = []
            for obs in obs_list:
                intervals.append((obs.events.time[0], obs.events.time[-1]))
            
            f_intervals = [[intervals[0][0],intervals[-1][-1]]]
            return f_intervals
 
      def evaluate_flux(self,obs_list,on_region,exclusion_mask):

          bkg_estimator = ReflectedRegionsBackgroundEstimator(
              obs_list=obs_list,
              on_region=on_region,
              exclusion_mask=exclusion_mask,
              )
          bkg_estimator.run()
          bkg_estimate = bkg_estimator.result
    
          extraction = SpectrumExtraction(obs_list = obs_list,
                                    bkg_estimate = bkg_estimator.result,
                                    containment_correction=False
                                    )

          extraction.run()  

          fit = SpectrumFit(extraction.observations, model = self.model)
          
          phase_interval = self.intervals(obs_list)
          lc_estimator = LightCurveEstimator(extraction)
          lc = lc_estimator.light_curve(
                time_intervals=phase_interval,
                spectral_model=self.model,
                energy_range=self.energy_range,
                 ) 
          return lc 
