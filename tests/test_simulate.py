import copy
import pytest
import numpy as np

from gnss_analysis.io import simulate

@pytest.fixture
def sbp_log_simulator(jsonlog):
  return simulate.simulate_from_log(jsonlog)

@pytest.fixture
def rinex_simulator(rinex_observation, rinex_navigation, rinex_base):
  return simulate.simulate_from_rinex(rinex_observation,
                                      rinex_navigation,
                                      rinex_base)

# TODO: There has to be a way to parameterize this with py.test,
#   but I can't figure it out.
def test_simulators(sbp_log_simulator,
                    rinex_simulator):
  """
  A quick and dirty test that just makes sure a simulator runs
  without failing and that each iteration is a copy, but doesn't
  check any of the actual content.
  """
  for obs_sets in [sbp_log_simulator, rinex_simulator]:
    prev_obs_set = obs_sets.next()
    obs_sets = [x for _, x in zip(range(10), obs_sets)]
    expected_keys = set(['rover', 'base', 'ephemeris', 'epoch',
                         'rover_info', 'base_info', 'ephemeris_info'])
    for obs_set in obs_sets:
      # make sure all the expected keys are keys in the obs_set
      assert not len(expected_keys.difference(obs_set.keys()))

      # now make sure we can modify fields in the obs_set without impacting
      # others.
      obs_set_copy = copy.deepcopy(obs_set)
      keys = ['rover', 'base', 'ephemeris']
      for k in keys:
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
