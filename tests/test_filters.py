import pytest
import numpy as np

from gnss_analysis.io import simulate, rinex
from gnss_analysis import locations, solution
from gnss_analysis.filters import kalman_filter, swiftnav_filter


@pytest.fixture()
def dgnss_filter():
  # TODO: Eventually use py.test parameterize to iterate over all filters
  return kalman_filter.StaticKalmanFilter(base_pos=locations.LEICA_ABSOLUTE)


@pytest.mark.slow
@pytest.mark.regression
def test_reasonable_and_doesnt_diverge(synthetic_stationary_states,
                                       dgnss_filter):
  """
  Uses noise free synthetic data ane makes sure the filter is
  converging and ends up with reasonably small error after a
  small number of iterations.
  """
  # Here we iterate over some number of baseline solutions and
  # report the error relative to the known baseline
  def iter_errors():
    for _, state in zip(range(40), synthetic_stationary_states):
      base_ecef = state['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
      rover_ecef = state['rover'][['ref_x', 'ref_y', 'ref_z']].values[0]
      expected_baseline = rover_ecef - base_ecef

      # update the filter and compute the baseline
      dgnss_filter.update(state)
      bl = dgnss_filter.get_baseline(state)
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
def test_eventually_gets_synthetic_baseline(synthetic_stationary_states,
                                            dgnss_filter):

  def assert_matches_at_some_point():
    for i, state in enumerate(synthetic_stationary_states):
      base_ecef = state['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
      rover_ecef = state['rover'][['ref_x', 'ref_y', 'ref_z']].values[0]
      expected_baseline = rover_ecef - base_ecef

      # update the filter and compute the baseline
      dgnss_filter.update(state)
      bl = dgnss_filter.get_baseline(state)
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


@pytest.mark.skipif(True, reason="The swiftnav baseline quickly diverges.")
def test_matches_libswiftnav(synthetic_stationary_states, dgnss_filter):

  swift_filter = swiftnav_filter.SwiftNavDGNSSFilter(disable_raim=True,
                                             base_pos=locations.LEICA_ABSOLUTE)

  for _, state in zip(range(10), synthetic_stationary_states):
    # This test fails because this baseline rapidly diverges.  It'll pass
    # for the first iteration, but then error out
    swift_filter.update(state)
    swift_bl = swift_filter.get_baseline(state)

    # update the filter and compute the baseline
    dgnss_filter.update(state)
    bl = dgnss_filter.get_baseline(state)

    # for now we'll behappy if the tow agree within a meter.
    # np.testing.assert_allclose(bl, swift_bl, atol=1)


@pytest.mark.skipif(True, reason="Can't match the piksi yet!")
def test_matches_piksi_logs(jsonlog, dgnss_filter):

  for state in solution.solution(simulate.simulate_from_log(jsonlog),
                                 dgnss_filter):
    # check that the single point positions match.
    np.testing.assert_allclose(state['rover_spp_ecef'][['x', 'y', 'z']].values[0],
                               state['rover_pos']['pos_ecef'],
                               atol=0.001)
    # if the python solver returned a non none baseline and
    # the piksi logged an rtk solution we check to see if the
    # two agree.
    if (state['rover_pos'].get('baseline', None) is not None and
        'rover_rtk_ned' in state):
      baseline = state['rover_pos']['baseline']
      piksi_baseline = state['rover_rtk_ned'][['n', 'e', 'd']]

      print "piksi", piksi_baseline.values
      print "python", baseline
      print "actual", locations.NOVATEL_BASELINE

@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize('filter_class', [kalman_filter.StaticKalmanFilter,
                                          kalman_filter.DynamicKalmanFilter])
def test_cors_baseline(datadir, filter_class):
  """
  Tests that the filter is capable of estimating the baseline for the
  cors short baseline to less than 1m accuracy within a reasonable
  amount of time.
  """

  rov = datadir.join('short_baseline_cors/seat032/seat0320.16o').strpath
  nav = datadir.join('short_baseline_cors/seat032/seat0320.16n').strpath
  base = datadir.join('short_baseline_cors/ssho032/ssho0320.16o').strpath

  states = simulate.simulate_from_rinex(rov, nav, base)
  rover_lines = rinex.iter_padded_lines(rov)
  rover_header = rinex.parse_header(rover_lines)
  base_lines = rinex.iter_padded_lines(base)
  base_header = rinex.parse_header(base_lines)

  rover_pos = np.array([rover_header['x'],
                        rover_header['y'],
                        rover_header['z']])
  base_pos = np.array([base_header['x'],
                       base_header['y'],
                       base_header['z']])
  expected_baseline = rover_pos - base_pos

  dgnss_filter = filter_class(base_pos=base_pos)

  # This iterates over solutions until the baseline gets within
  # one meter of the known solution at which point it will return True.
  # If that never happens it returns False
  def eventually_close():
    for _, soln in zip(range(100), solution.solution(states, dgnss_filter)):
      bl = soln['rover_pos']['baseline']
      if np.linalg.norm(bl - expected_baseline) <= 1.:
        return True
    return False

  assert eventually_close()


@pytest.mark.slow
@pytest.mark.regression
def test_cors_drops_reference(datadir, dgnss_filter):
  """
  Tests that the filter is capable of estimating the baseline for the
  cors short baseline to less than 1m accuracy within a reasonable
  amount of time.
  """

  rov = datadir.join('cors_drops_reference/seat032/partial_seat0320.16o').strpath
  nav = datadir.join('cors_drops_reference/seat032/seat0320.16n').strpath
  base = datadir.join('cors_drops_reference/ssho032/partial_ssho0320.16o').strpath

  states = simulate.simulate_from_rinex(rov, nav, base)
  rover_lines = rinex.iter_padded_lines(rov)
  rover_header = rinex.parse_header(rover_lines)
  base_lines = rinex.iter_padded_lines(base)
  base_header = rinex.parse_header(base_lines)

  rover_pos = np.array([rover_header['x'],
                        rover_header['y'],
                        rover_header['z']])
  base_pos = np.array([base_header['x'],
                       base_header['y'],
                       base_header['z']])
  expected_baseline = rover_pos - base_pos

  dgnss_filter = kalman_filter.StaticKalmanFilter(base_pos=base_pos)

  solns = list(solution.solution(states, dgnss_filter))
  err = np.linalg.norm(solns[-1]['rover_pos']['baseline'] - expected_baseline)
  # this dataset doesn't have many observations so we just make sure the
  # baseline is reasonable.
  assert err <= 2.
