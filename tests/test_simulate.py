import copy
import pytest
import numpy as np

from gnss_analysis.io import simulate


def test_simulators(observation_sets):
  """
  A quick and dirty test that just makes sure a simulator runs
  without failing and that each iteration is a copy, but doesn't
  check any of the actual content.
  """
  prev_obs_set = observation_sets.next()
  obs_sets = [x for _, x in zip(range(10), observation_sets)]
  # note that sometimes we don't have base observations, so it isn't
  # a required key
  expected_keys = set(['rover', 'ephemeris', 'epoch',
                       'rover_info', 'base_info', 'ephemeris_info'])
  for obs_set in obs_sets:
    # make sure all the expected keys are keys in the obs_set
    assert not len(expected_keys.difference(obs_set.keys()))

    # now make sure we can modify fields in the obs_set without impacting
    # others.
    obs_set_copy = copy.deepcopy(obs_set)
    for k in [k for k in obs_set.keys() if not k.endswith('_info')]:
      # scalars can't be updated in place, so no need to worry about them.
      # also skip this if an observation type wasn't seen before
      if np.isscalar(obs_set[k]) or k not in prev_obs_set:
        continue
      prev_v_copy = copy.deepcopy(prev_obs_set[k])
      # fill with nans
      obs_set[k].ix[:] = np.nan
      # make sure modifying the current state won't
      # alter the previous state.  We have to compare
      # by iterating over each column since each column
      # may be a different data type.
      for (_, x), (_, y) in zip(prev_obs_set[k].iteritems(),
                                prev_v_copy.iteritems()):
        np.testing.assert_array_equal(x.values, y.values)
    prev_obs_set = obs_set_copy
