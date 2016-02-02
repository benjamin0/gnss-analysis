import pytest
import random
import numpy as np

from pandas.util.testing import assert_frame_equal

from swiftnav import dgnss_management

from gnss_analysis import constants as c
from gnss_analysis import dgnss, ephemeris, synthetic, locations


def test_single_difference(synthetic_state):
  """
  This assumes that single_difference returns accurate differences
  (which should be checked by test_make_propagated_single_differences)
  and instead focuses on testing edge cases (dropped satellites etc.)
  """
  base_obs = ephemeris.add_satellite_state(synthetic_state['base'],
                                            account_for_sat_error=False)
  rover_obs = ephemeris.add_satellite_state(synthetic_state['rover'],
                                            account_for_sat_error=False)
  # Compute the full set of single differences.
  expected_diffs = dgnss.single_difference(rover_obs, base_obs)

  # now shuffle the observations and make sure satellites don't get confused.
  random.seed(1982)
  inds = rover_obs.index.values.copy()
  random.shuffle(inds)
  shuffled_rover = rover_obs.ix[inds]
  actual = dgnss.single_difference(shuffled_rover, base_obs)
  assert_frame_equal(actual, expected_diffs)

  # drop a row from rover and make sure it's handled properly
  subset_rover = shuffled_rover.iloc[1:]
  actual = dgnss.single_difference(subset_rover, base_obs)
  assert_frame_equal(actual, expected_diffs.ix[subset_rover.index].sort())

  # drop a row from rover and make sure it's handled properly
  subset_base = base_obs.iloc[1:]
  actual = dgnss.single_difference(rover_obs, subset_base)
  assert_frame_equal(actual, expected_diffs.ix[subset_base.index].sort())


def test_make_propagated_single_differences(ephemerides):
  """
  Create single differences for pairs of rover and base stations
  that correspond to different times of arrival, then confirm that the
  resulting single differences agree with the geometric interpretation
  of single differences.
  """
  # Compute the expected baseline from the base / rover reference locations
  base_ecef = np.array(locations.LEICA_ABSOLUTE)
  rover_ecef = np.array(locations.NOVATEL_ABSOLUTE)
  expected_baseline = rover_ecef - base_ecef

  # create a set of base observations
  base_toa = ephemerides['toe'].values[0] + np.timedelta64(100, 's')
  base_obs = synthetic.observations_from_toa(ephemerides,
                                             base_ecef, base_toa)

  omega_unit_vect = dgnss.omega_dot_unit_vector(base_ecef, base_obs,
                                                expected_baseline)
  # the single differences should equal the baseline dotted with the unit vectors
  expected_sdiffs = np.dot(omega_unit_vect, expected_baseline)
  # now compute the single difference between base and rover observations when
  # the base observation needs to be propagated forward in time (up to 10
  # seconds).
  for i in range(10):
    toa = base_toa + np.timedelta64(i, 's')
    rover_obs = synthetic.observations_from_toa(ephemerides,
                                                rover_ecef, toa)

    sdiffs = dgnss.make_propagated_single_differences(rover_obs,
                                                      base_obs,
                                                      base_pos_ecef=base_ecef)
    # TODO: Using datetime64 objects with nanosecond resolution means
    # that we can only get +/- 0.3m accuracy on ranges!  Oops!  We'll need to
    # change the way times are represented, perhaps rolling back to using
    # wn, tow
    np.testing.assert_allclose(sdiffs['pseudorange'].values,
                               expected_sdiffs,
                               atol=6e-1)


def test_matches_make_measurements(synthetic_state):
  """
  Compute double differences and make sure they match the differences
  from swiftnav.dgnss_management.make_measurements_
  """
  # make a set of single differences
  base_obs = ephemeris.add_satellite_state(synthetic_state['base'],
                                           account_for_sat_error=False)
  rover_obs = ephemeris.add_satellite_state(synthetic_state['rover'],
                                           account_for_sat_error=False)
  base_ecef = synthetic_state['base'][['ref_x', 'ref_y', 'ref_z']].values[0]
  sdiffs = dgnss.make_propagated_single_differences(rover_obs,
                                                    base_obs,
                                                    base_pos_ecef=base_ecef)
  sdiffs_t = list(dgnss.create_single_difference_objects(sdiffs))
  # pass them through swiftnav's code
  expected_ddiffs = dgnss_management.make_measurements_(sdiffs_t)
  # then similar python code
  actual_ddiffs = dgnss.double_difference(sdiffs)
  # and make sure the outputs agree.
  nddiffs = len(sdiffs_t) - 1
  np.testing.assert_array_equal(expected_ddiffs[:nddiffs],
                                actual_ddiffs['carrier_phase'].values)
  np.testing.assert_array_equal(expected_ddiffs[nddiffs:],
                                actual_ddiffs['pseudorange'].values)


def test_double_differences(ephemerides):
  """
  Compute double differences and make sure they align with the geometric
  interpretation.
  """
  # Compute the expected baseline from the base / rover reference locations
  base_ecef = np.array(locations.LEICA_ABSOLUTE)
  rover_ecef = np.array(locations.NOVATEL_ABSOLUTE)
  expected_baseline = rover_ecef - base_ecef

  # create a set of base observations
  toa = ephemerides['toe'].values[0] + np.timedelta64(100, 's')
  base_obs = synthetic.observations_from_toa(ephemerides,
                                             base_ecef, toa)
  rover_obs = synthetic.observations_from_toa(ephemerides,
                                              rover_ecef, toa)
  omega_unit_vect = dgnss.omega_dot_unit_vector(base_ecef, base_obs,
                                                expected_baseline)
  # compute a matrix of unit vector differences between the reference
  # satellite and all others, the result will have shape (k - 1, 3).
  unit_diffs = omega_unit_vect[1:] - omega_unit_vect[0]
  # the double differences should equal the baseline dotted with the difference
  # in unit vectors
  expected_ddiffs = np.dot(unit_diffs, expected_baseline)

  sdiffs = dgnss.make_propagated_single_differences(rover_obs,
                                                    base_obs,
                                                    base_pos_ecef=base_ecef)
  ddiffs = dgnss.double_difference(sdiffs)
  # Because synthetic observations are made with time units that only have
  # nanosecond resolution we can't get better than +/-0.3 m resolution
  # on ranges, in turn these are only accurate to 6e-1.
  np.testing.assert_allclose(ddiffs['pseudorange'].values,
                             expected_ddiffs, atol=6e-1)
  np.testing.assert_allclose(ddiffs['carrier_phase'].values * c.GPS_L1_LAMBDA,
                             expected_ddiffs, atol=6e-1)