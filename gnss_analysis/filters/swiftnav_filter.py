"""
swiftnav_filter.py

This Filter is designed with the intent of reproducing the same estimates
that would have been made on the chip.  It uses as much logic from libswiftnav
as possible.
"""
import logging
import numpy as np
import pandas as pd

from swiftnav import dgnss_management

from gnss_analysis import dgnss, ephemeris
from gnss_analysis.filters.common import TimeMatchingDGNSSFilter


class SwiftNavDGNSSFilter(TimeMatchingDGNSSFilter):
  """
  This version of the filter simply uses the wrapped libswiftnav code,
  essentially reproducing what is happening directly on the chip.
  """
  def __init__(self, disable_raim=False, *args, **kwdargs):
    super(SwiftNavDGNSSFilter, self).__init__(*args, **kwdargs)
    self.disable_raim = disable_raim
    # our initial guess for the rover position is to assume it's simply
    # in the same location as the base receiver.
    self.rover_pos = np.zeros(3)
    # Initialized the ambiguity state
    self.amb_state = dgnss_management.AmbiguityState()
    self.initialized = False

  def update_matched_obs(self, rover_obs, base_obs):
    """
    Updates the swiftnav kalman filter with a new pair of rover and base
    station observations.
    
    TimeMatchingDGNSSFilter.update takes care of finding pairs of time
    of arrival aligned rover and base observations, this function
    is then responsible for actually updating the filter.
    
    See also: piksi_firmware/solution.c:process_matched_obs
    """
    # creates a DataFrame of single differences
    rover_obs = rover_obs.copy()
    base_obs = base_obs.copy()
    rover_obs['carrier_phase'] = -rover_obs['carrier_phase']
    base_obs['carrier_phase'] = -base_obs['carrier_phase']
    sdiffs = self.get_single_diffs(rover_obs, base_obs, propagate_base=False)
    # converts the DataFrame to c objects
    sdiff_t = list(dgnss.create_single_difference_objects(sdiffs))
    # update the filter
    dgnss_management.dgnss_update_(sdiff_t, self.base_pos,
                                   disable_raim=self.disable_raim)
    # dgnss_update_ takes care of the initialization logic
    self.initialized = True
    # update the ambiguities
    dgnss_management.dgnss_update_ambiguity_state_(self.amb_state)

  def get_baseline(self, obs_set):
    """
    Uses the current filter state to estimate the baseline for
    a set of (possibly non-aligned) rover and base observations.
    
    Base observations are propagated to the rover time of arrival
    (if required) before being used to compute the baseline.
    """
    # Add the satellite state information (position etc ..)
    # to the rover observations.
    obs_set['base'] = ephemeris.add_satellite_state(obs_set['base'],
                                                    obs_set['ephemeris'])
    # creates a DataFrame of single differences
    sdiffs = self.get_single_diffs(obs_set['rover'], obs_set['base'],
                                   propagate_base=True)
    # converts the DataFrame to c objects
    sdiff_t = list(dgnss.create_single_difference_objects(sdiffs))
    # use the current filter state to estimate the baseline
    flag, n_used, baseline = dgnss_management.dgnss_baseline_(sdiff_t,
                                                  self.base_pos,
                                                  self.amb_state,
                                                  disable_raim=self.disable_raim)
    # check the flag to see if baseline computations succeeded.  If not
    # we return None instead of a possibly bogus baseline.
    if flag < 0:
      logging.warn("Baseline computation failed with %d" % flag)
      return None
    return pd.Series(baseline, index=['x', 'y', 'z'])
