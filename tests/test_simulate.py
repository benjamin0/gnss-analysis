
import copy
import numpy as np

from gnss_analysis import simulate


def test_simulate_from_log(jsonlog):
  """
  A quick and dirty test that just makes sure
  log processing runs without failing and that
  each iteration is a copy, but doesn't check
  any of the actual content.
  """
  states = simulate.simulate_from_log(jsonlog)
  prev_state = states.next()
  states = [x for _, x in zip(range(10), states)]
  for state in states:
    state_copy = copy.deepcopy(state)
    keys = ['rover', 'base', 'ephemeris']
    for k in keys:
      prev_v_copy = copy.deepcopy(prev_state[k])
      # fill with nans
      state[k].ix[:] = np.nan
      # make sure modifying the current state won't
      # alter the previous state
      np.testing.assert_array_equal(prev_state[k].values,
                                    prev_v_copy.values)
    prev_state = state_copy
