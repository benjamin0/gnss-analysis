import logging
import numpy as np
import pandas as pd

from gnss_analysis import dgnss, ephemeris, propagate, solution


def get_unique_value(x):
  uniq = np.unique(x)
  assert uniq.size == 1
  return uniq[0]


class DGNSSFilter(object):
  """
  DGNSSFilter
  
  An abstract class that contains some core shared logic required for
  a DGNSS filter used to estimate baselines.
  """

  def __init__(self, base_pos=None):
    if base_pos is None:
      self.base_known = False
      self.base_pos = None
    else:
      self.base_known = True
      self.base_pos = base_pos
    self.prev_sdiffs = None

  def maybe_update_base_position(self, base_obs):
    """
    If base_known is True this function does nothing, otherwise it
    will update the base position by performing PVT solve.
    """
    if not self.base_known:
      # TODO: on the piksi this position has a low pass filter applied.
      self.base_pos = solution.single_point_position(base_obs)['pos_ecef']

  def get_single_diffs(self, rover_obs, base_obs, propagate_base=False):
    """
    Computes the single differences between a set of rover and base observations.
    """
    rover_toa = get_unique_value(rover_obs['time'])
    if propagate_base:
      # Move the base observations forward in time to math the rover obs.
      base_obs = propagate.delta_tof_propagate(self.base_pos,
                                               base_obs,
                                               rover_toa)
    else:

      # if we aren't propagating make sure they are on the same time.
      assert get_unique_value(base_obs['time']) == rover_toa

    # Compute single differences.
    sdiffs = dgnss.make_propagated_single_differences(rover_obs, base_obs,
                                                      self.base_pos)

    if self.prev_sdiffs is not None:
      # check to make sure the lock hasn't changed since the previous
      # update.
      locks = sdiffs['lock'].align(self.prev_sdiffs['lock'], 'left')
      # we still use a single difference if the previous was nan
      # under the assumption that the filter logic will add it
      # to the state appropriately
      use = np.logical_or(locks[0] == locks[1],
                          np.isnan(locks[1]))
      if not np.all(use):
        logging.warn("Lock counters slipped, dropping %d diff(s)"
                     % np.sum(np.logical_not(use)))
      good_sdiffs = sdiffs[use]
    else:
      good_sdiffs = sdiffs
    # store the current sdiffs for the next iteration
    self.prev_sdiffs = good_sdiffs
    return good_sdiffs

  def update(self, state):
    raise NotImplementedError

  def get_baseline(self, state):
    raise NotImplementedError


class TimeMatchingDGNSSFilter(DGNSSFilter):
  """
  TimeMatchingDGNSSFilter
  
  This is an abstract filter which takes care of the logic required
  to match base observations with corresponding rover observations.
  
  Each time this filter is updated the rover observations are pushed
  into a buffer, then when a new base observation is encountered it
  is matched with the corresponding rover observations from the buffer.
  """

  def __init__(self, *args, **kwdargs):
    super(TimeMatchingDGNSSFilter, self).__init__(*args, **kwdargs)
    # TODO: this is certainly not the most efficient buffer
    self.rover_buffer = pd.DataFrame()
    self.prev_base = None

  def update_matched_obs(self, rover_obs, base_obs):
    raise NotImplementedError("TimeMatchingDGNSSFilter requires implementing"
                              " update_matched_obs.")

  def _pop_from_buffer(self, t):
    """
    Removes the rover observations matching tow from the buffer.  The buffer
    is then reduced to contain only observations after the requested tow.
    """
    use = self.rover_buffer['time'] == t
    # the buffer hasn't built up long enough of a history yet, return none
    if not np.any(use):
      return None
    matched_obs = self.rover_buffer[use]
    # we should no longer need any rover observations from at or before
    # the currently requested tow
    self.rover_buffer = self.rover_buffer[self.rover_buffer.time > t]
    return matched_obs

  def _append_to_buffer(self, rover_state):
    """
    Adds a set of rover state observations to the buffer.
    """
    rover_toa = get_unique_value(rover_state['time'])
    # make sure the current tow doesn't exist in the buffer yet,
    # tow needs to be unique since it's used to index.
    assert (self.rover_buffer.empty or
            not rover_toa in self.rover_buffer['time'])
    # update the DataFrame buffer
    self.rover_buffer = pd.concat([self.rover_buffer,
                                   rover_state])

  def _validate_base_observation(self, state):
    # Check to see if we have a set of base observations.  Without
    # them we can't proceed.
    if state.get('base', None) is None:
      logging.warn("No base observations found, not updating filter.")
      return False

    # get the time of arrival for the base observations
    toa = get_unique_value(state['base']['time'])

    # make sure the the current base observations are newer than
    # anything we've seen before.  If not we skip this update.
    if (self.prev_base is not None and
        toa <= get_unique_value(self.prev_base['time'])):
      logging.warn("Base observations are old, not updating filter.")
      return False
    return True

  def update(self, state):
    """
    Updates the filter given the new state.
    """
    # Always push the rover state to the buffer
    self._append_to_buffer(state['rover'])

    # Make sure the state contains a new and valid base observation
    if not self._validate_base_observation(state):
      return

    # Store the current base as a reference for the next iteration
    self.prev_base = state['base']
    # Possibly update the base position using it's SPP.
    self.maybe_update_base_position(state['base'])

    # Infer any required variables and add satellite state.
    state['rover'] = ephemeris.add_satellite_state(state['rover'],
                                                   state['ephemeris'],
                                                   account_for_sat_error=True)

    # get the time of arrival for the base observations
    toa = get_unique_value(state['base']['time'])

    # Attempt to find a rover observation that corresponds to the base
    # observation.
    rover_obs = self._pop_from_buffer(toa)
    # If no observations were found, issue a warning and return
    if rover_obs is None:
      logging.warn("No rover observations available for %f" % toa)
      return

    # this should be checked before calling this function, make sure
    assert not rover_obs.empty
    assert not state['base'].empty
    assert rover_obs.shape[0] >= 4
    assert state['base'].shape[0] >= 4

    # Add the satellite state information (position etc ..)
    # to the rover observations.
    base_obs = ephemeris.add_satellite_state(state['base'],
                                             state['ephemeris'],
                                             account_for_sat_error=True)

    # We now have a pair of rover and base observations that correspond
    # to the same time of arrival, we pass them on to the implementing
    # filter.
    self.update_matched_obs(rover_obs, base_obs)
