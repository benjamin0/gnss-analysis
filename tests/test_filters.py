import pytest
import numpy as np

from gnss_analysis import solution
from gnss_analysis import locations


@pytest.mark.slow
@pytest.mark.regression
def test_reasonable_and_doesnt_diverge(synthetic_stationary_observations,
                                       dgnss_filter_class):
  """
  Uses noise free synthetic data ane makes sure the filter is
  converging and ends up with reasonably small error after a
  small number of iterations.
  """
  obs_sets = [y for _, y in zip(range(40), synthetic_stationary_observations)]
  base_pos = obs_sets[0]['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
  dgnss_filter = dgnss_filter_class(base_pos=base_pos)
  # Here we iterate over some number of baseline solutions and
  # report the error relative to the known baseline
  def iter_errors():
    for obs_set in obs_sets:
      base_ecef = obs_set['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
      rover_ecef = obs_set['rover'][['ref_x', 'ref_y', 'ref_z']].values[0]
      expected_baseline = rover_ecef - base_ecef

      # update the filter and compute the baseline
      dgnss_filter.update(obs_set)
      bl = dgnss_filter.get_baseline(obs_set)
      yield bl - expected_baseline

  # Then we solve for the slope of the error magnitude and make
  # sure it isn't obviously blowing up.
  errors = np.array([np.linalg.norm(x) for x in iter_errors()])
  A = np.vstack([np.ones(errors.size), np.arange(errors.size)]).T
  slope = np.linalg.lstsq(A, errors)[0][1]
  # make sure the errors aren't increasing by more than a centimeter a sec
  assert slope <= 0.01
  # make sure the worst of the last five iterations gets within
  # a mm.
  assert np.max(errors[-5:]) <= 1e-3


@pytest.mark.slow
@pytest.mark.regression
def test_eventually_gets_synthetic_baseline(synthetic_stationary_observations,
                                            dgnss_filter_class):

  def assert_matches_at_some_point():
    for i, obs_set in enumerate(synthetic_stationary_observations):

      base_ecef = obs_set['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
      rover_ecef = obs_set['rover'][['ref_x', 'ref_y', 'ref_z']].values[0]
      expected_baseline = rover_ecef - base_ecef

      if i == 0:
        dgnss_filter = dgnss_filter_class(base_pos=base_ecef)

      # update the filter and compute the baseline
      dgnss_filter.update(obs_set)
      bl = dgnss_filter.get_baseline(obs_set)
      if not bl is None and i > 1:
        # check to see if the estimated baseline is close to the expected one
        # if it is close we break out of the for loop and consider the test
        # a success, if not we ignore the assertion and continue
        try:
          np.testing.assert_allclose(bl, expected_baseline, atol=0.1)
          return
        except:
          pass
    # if the baseline matched at some point it would hit the return
    # statement above and break out of this function.  If that didn't happen
    # then we raise an assertion error
    raise AssertionError("filter didn't match baseline before end of sequence")

  assert_matches_at_some_point()


@pytest.mark.slow
@pytest.mark.regression
def test_agrees_with_piksi_logs(piksi_roof, dgnss_filter_class):
  """
  Tests for reasonable agreement with the piksi logs.  This is done
  by running dgnss_filter_class through the piksi_roof log and
  making sure the error at the end is less than twice the error
  of the piksi reported baselines.
  """
  dgnss_filter = dgnss_filter_class(base_pos=locations.LEICA_ABSOLUTE)

  for obs_set in solution.solution(piksi_roof, dgnss_filter):
    # check that the single point positions match.
    np.testing.assert_allclose(obs_set['rover_spp_ecef'][['x', 'y', 'z']].values[0],
                               obs_set['rover_pos'][['x', 'y', 'z']].values[0],
                               atol=0.2, rtol=1e-1)
    # if the python solver returned a non none baseline and
    # the piksi logged an rtk solution we check to see if the
    # two agree.
    if ('baseline_x' in obs_set['rover_pos'] and
        'rover_rtk_ned' in obs_set):
      baseline = obs_set['rover_pos'][['baseline_x',
                                       'baseline_y',
                                       'baseline_z']].values[0]
      piksi_baseline = obs_set['rover_rtk_ecef'][['x', 'y', 'z']].values[0]

  piksi_error = np.linalg.norm(piksi_baseline - locations.NOVATEL_BASELINE)
  error = np.linalg.norm(baseline - locations.NOVATEL_BASELINE)

  # After a decent number of iterations, make sure the filter's error
  # is less than or equal to the piksi.
  assert 2 * piksi_error >= error


@pytest.mark.slow
@pytest.mark.regression
def test_cors_baseline(cors_observation_sets, dgnss_filter_class, request):
  """
  Tests that the filter is capable of estimating the baseline for the
  cors short baseline to less than 1m accuracy within a reasonable
  amount of time.
  """
  # Extract the known positions from the info attributes
  first = cors_observation_sets.next()
  rover_pos = np.array([first['rover_info']['x'],
                        first['rover_info']['y'],
                        first['rover_info']['z']])
  base_pos = np.array([first['base_info']['x'],
                       first['base_info']['y'],
                       first['base_info']['z']])
  expected_baseline = rover_pos - base_pos
  dgnss_filter = dgnss_filter_class(single_band=True)

  # This iterates over solutions until the baseline gets within
  # 1 meter of the known solution at which point it will return True.
  # If that never happens it returns False.  We could clamp down the
  # accuracy further, but some of the test datasets don't contain
  # sufficient number of observations.
  def eventually_close():
    for _, soln in zip(range(100), solution.solution(cors_observation_sets,
                                                     dgnss_filter)):
      bl = soln['rover_pos'][['baseline_x',
                              'baseline_y',
                              'baseline_z']].values
      if np.linalg.norm(bl - expected_baseline) <= 1.:
        return True
    return False

  assert eventually_close()


@pytest.mark.slow
@pytest.mark.regression
def test_multiband_cors_baseline(multignss_cors_35km_baseline, dgnss_filter_class):
  """
  Tests that the filter is capable of estimating the baseline for the
  cors short baseline to less than 1m accuracy within a reasonable
  amount of time.
  """
  # Extract the known positions from the info attributes
  first = multignss_cors_35km_baseline.next()
  rover_pos = np.array([first['rover_info']['x'],
                        first['rover_info']['y'],
                        first['rover_info']['z']])
  base_pos = np.array([first['base_info']['x'],
                       first['base_info']['y'],
                       first['base_info']['z']])
  expected_baseline = rover_pos - base_pos
  dgnss_filter = dgnss_filter_class(single_band=False)

  # This iterates over solutions until the baseline gets within
  # 0.5 meters of the known solution at which point it will return True.
  # If that never happens it returns False.
  def eventually_close():
    for _, soln in zip(range(100), solution.solution(multignss_cors_35km_baseline,
                                                     dgnss_filter)):
      bl = soln['rover_pos'][['baseline_x',
                              'baseline_y',
                              'baseline_z']].values
      if np.linalg.norm(bl - expected_baseline) <= 0.5:
        return True
    return False

  assert eventually_close()
