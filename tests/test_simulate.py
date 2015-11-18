
import pytest
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
  for state in states:
    prev_state_copy = prev_state.copy()
    state_copy = state.copy()
    # fill with nans
    state.ix[:] = np.nan
    # make sure modifying the current state won't
    # alter the previous state
    np.testing.assert_array_equal(prev_state.values,
                                  prev_state_copy.values)
    prev_state = state_copy
