import pytest
import numpy as np
import pandas as pd

from swiftnav import time as gpstime

import gnss_analysis.constants as c

from gnss_analysis import ephemeris, observations, locations, time_utils


def test_calc_sat_state(ephemerides):
  """
  This confirms that the calculations made by the gnss_analysis
  version of calc_sat_state matches the c implementation.
  """
  n = ephemerides.shape[0]
  time = ephemerides['time'].copy()
  noise = 1e-3 * np.random.normal(size=n)
  time += time_utils.timedelta_from_seconds(noise)

  # make sure we've added a fit interval
  assert 'fit_interval' in ephemerides

  actual = ephemeris.calc_sat_state(ephemerides, time)

  for i in range(n):
    eph = ephemerides.iloc[[i]]
    eph_obj = observations.mk_ephemeris(eph.reset_index())
    # compute the distance to satellite at the time of observation
    wn_tow = time_utils.datetime_to_tow(time.values[i])
    gpst = gpstime.GpsTime(**wn_tow)
    expected = eph_obj.calc_sat_state(gpst)
    another = ephemeris.calc_sat_state(eph, time.iloc[[i]])
    act = actual.iloc[[i], :]

    # make sure bulk and individual sat state error computations
    # are identical
    assert np.all(act == another)
    pos, vel, clock_error, clock_error_rate = expected
    # make sure positions agree within a mm
    assert np.all(np.abs(act[['sat_x', 'sat_y', 'sat_z']] - pos) < 1e-3)
    # make sure velocities agree within a mm/second
    assert np.all(np.abs(act[['sat_v_x', 'sat_v_y', 'sat_v_z']] - vel) < 1e-3)
    # make sure clock error agree within a femtosecond
    assert np.all(np.abs(clock_error - act['sat_clock_error']) < 1e-12)
    # make sure clock error rate agrees within a femtosecond / second
    assert np.all(np.abs(clock_error_rate - act['sat_clock_error_rate']) < 1e-12)

  # Test with the time empbedded in the ephemerides object
  ephemerides['time'] = time
  another = ephemeris.calc_sat_state(ephemerides)
  np.testing.assert_array_equal(actual.values, another.values)


def test_sagnac_rotation(ephemerides):
  time = ephemerides['time'].copy()
  n = ephemerides.shape[0]
  noise = 40 + 1e-3 * np.random.normal(size=n)
  time += time_utils.timedelta_from_seconds(noise)

  sat_state = ephemeris.calc_sat_state(ephemerides, time)

  time_of_flight = np.random.normal(0.07, 0.01, size=n)
  time_of_flight = time_utils.timedelta_from_seconds(time_of_flight)
  rotated = ephemeris.sagnac_rotation(sat_state[['sat_x', 'sat_y', 'sat_z']],
                                      time_of_flight=time_of_flight)

  for i in range(ephemerides.shape[0]):
    one_rot = ephemeris.sagnac_rotation(sat_state[['sat_x', 'sat_y', 'sat_z']].values[i],
                                        time_of_flight=time_of_flight[i])
    np.testing.assert_array_equal(rotated[i, :], one_rot[0])

    # only difference here is that we take a two-dim array instead of a 1-d one.
    one_rot = ephemeris.sagnac_rotation(sat_state[['sat_x', 'sat_y', 'sat_z']].values[[i]],
                                        time_of_flight=time_of_flight[i])
    np.testing.assert_array_equal(rotated[i, :], one_rot[0])

  # rotate backwards and compare to the original
  maybe_orig = ephemeris.sagnac_rotation(rotated, time_of_flight=-time_of_flight)
  np.testing.assert_array_almost_equal(sat_state[['sat_x', 'sat_y', 'sat_z']],
                                       maybe_orig, decimal=6)


def test_update_columns():
  one = pd.DataFrame({'one': np.arange(5.),
                      'shared': np.arange(5.) + 2},
                      index=np.arange(5))
  two = pd.DataFrame({'shared': np.arange(1, 6.) + 2,
                      'two': np.arange(1, 6.) + 3},
                     index=np.arange(1, 6))

  expected = pd.DataFrame({'one': np.arange(1, 5.),
                           'shared': np.arange(1, 5.) + 2,
                           'two': np.arange(1, 5.) + 3},
                      index=np.arange(1, 5))

  actual = ephemeris._update_columns(one, two)
  assert np.all(actual == expected)

  # try again with a nan in the mix
  one['one'].values[2] = np.nan
  expected['one'].values[1] = np.nan
  actual = ephemeris._update_columns(one, two)
  np.testing.assert_array_equal(actual.values, expected.values)

  # make sure that if one is a subset of two all the values of
  # are updated.
  one = two.copy()
  one.values[:] += np.random.normal(size=one.shape)
  should_be_two = ephemeris._update_columns(one, two)
  np.testing.assert_array_equal(should_be_two.values, two.values)

  # make sure even when none of the columns for one are used, the
  # inner set of indices still is
  one = two.copy()
  one = one.iloc[1:]
  one.values[:] += np.random.normal(size=one.shape)
  missing_row = ephemeris._update_columns(one, two)
  np.testing.assert_array_equal(missing_row.values, two.iloc[1:].values)


def test_add_satellite_state(ephemerides):
  """
  Makes sure that add_satellite_state performs as expected
  """
  ref_time = ephemerides.iloc[0]['time']
  ref_time = ref_time.to_datetime64() + time_utils.timedelta_from_seconds(40)

  # make some random transmission times
  np.random.seed(1982)
  noise = np.random.normal(0.08, 0.01, size=ephemerides.shape[0])
  tot = ref_time - time_utils.timedelta_from_seconds(noise)
  # compute the corresponding raw_pseudoranges according to the method used in
  # track.c:calc_navigation_measurements
  expected = ephemeris.calc_sat_state(ephemerides, tot)
  tof = time_utils.seconds_from_timedelta(ref_time - tot)
  expected['raw_pseudorange'] = tof * c.GPS_C
  expected['raw_doppler'] = np.random.normal(100., 100., size=expected.shape[0])
  expected['tot'] = tot
  expected['pseudorange'] = expected['raw_pseudorange'] + expected.sat_clock_error * c.GPS_C
  expected['doppler'] = expected['raw_doppler'] + expected.sat_clock_error_rate * c.GPS_L1_HZ

  def equals_expected(to_test):
    _, to_test = expected.align(to_test, 'left')
    for (k, x), (_, y) in zip(expected.iteritems(),
                              to_test.iteritems()):
      if not np.all(x == y):
        return False
    return True

  orig = ephemerides.copy()
  orig['raw_pseudorange'] = expected['raw_pseudorange']
  orig['raw_doppler'] = expected['raw_doppler']
  orig['time'] = ref_time

  actual = ephemeris.add_satellite_state(orig)
  assert equals_expected(actual)

  # try with ephemerides passed in seperate
  actual = ephemeris.add_satellite_state(orig, ephemerides)
  equals_expected(actual)

  bad_ephemerides = orig.copy()
  bad_ephemerides['af0'] += 10
  bad = ephemeris.add_satellite_state(bad_ephemerides)
  assert not equals_expected(bad)
  # make sure the explicit ephemerides overrides the bad ones
  good = ephemeris.add_satellite_state(bad_ephemerides, ephemerides)
  assert equals_expected(good)

  # make sure it's idempotent
  another = ephemeris.add_satellite_state(actual)
  assert equals_expected(another)

  # make sure we can modify an object it it won't change the original
  another['time'] += time_utils.timedelta_from_seconds(5)
  assert equals_expected(actual)


def test_time_of_transmission_vectorization(ephemerides):
  """
  Tests to make sure that the vectorized version matches
  the output if you pipe individual satellites through to
  ensure there aren't strange vectorization issues.
  """
  ref_time = ephemerides.iloc[0]['time'].to_datetime64()
  ref_time += time_utils.timedelta_from_seconds(40)
  ref_loc = locations.NOVATEL_ABSOLUTE

  # a pretty straight forward consistency check that makes sure
  # vectorized solves are the same as individual solves.
  full = ephemeris.time_of_transmission(ephemerides, ref_time, ref_loc)
  for i in range(full.shape[0]):
    single = ephemeris.time_of_transmission(ephemerides.iloc[[i]],
                                            ref_time, ref_loc)
    assert full[i] == single[0]

  # make sure the default iterations is basically
  # the same as after 10
  ten = ephemeris.time_of_transmission(ephemerides,
                                       ref_time,
                                       ref_loc,
                                       max_iterations=10,
                                       tol=1e-12)
  t_diff = time_utils.seconds_from_timedelta(ten - full)
  assert np.all(np.abs(t_diff) <= 1e-9)

  # make sure the algorithm converged and is less than
  # the convergence tolerance
  nine = ephemeris.time_of_transmission(ephemerides,
                                        ref_time,
                                        ref_loc,
                                        max_iterations=9,
                                        tol=0)
  t_diff = time_utils.seconds_from_timedelta(nine - ten)
  np.all(np.abs(t_diff) < 1e-12)


def test_time_of_roundtrip(ephemerides):
  """
  Starts with some satellite pos and random times of transmission,
  calculates the times of arrival at a reference location then
  backs out the time of transmission to make sure it matches.
  """
  n = ephemerides.shape[0]
  tot = ephemerides['time'].copy()
  noise = np.random.normal(0.1, 0.1, size=n)
  tot += 40 + time_utils.timedelta_from_seconds(noise)
  ref_loc = locations.NOVATEL_ABSOLUTE

  # compute the time of arrival for a signal leaving at transmission time from
  # the satellites and arriving at ref_loc.
  toa = ephemeris.time_of_arrival(ephemerides, tot, ref_loc)
  # now run time_of_transmission and make sure we get the original back.
  actual_tot = ephemeris.time_of_transmission(ephemerides,
                                              time_of_arrival=toa,
                                              ref_loc=ref_loc)
  tdiffs = time_utils.seconds_from_timedelta(tot.values - actual_tot.values)
  assert np.all(np.abs(tdiffs) < 1e-12)


def test_sat_velocity(ephemerides):
  # computes a set of satellite states, then uses the state
  # velocity to predict the state at a nearby time and compares
  # it to the actual sat state at that time.
  ref_time = ephemerides.iloc[0]['time']
  ref_time += time_utils.timedelta_from_seconds(40)

  sat_state = ephemeris.calc_sat_state(ephemerides, ref_time)
  vel = sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']].values
  pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values

  dt = 1e-6
  new_time = ref_time + time_utils.timedelta_from_seconds(dt)
  new_sat_state = ephemeris.calc_sat_state(ephemerides, new_time)

  expected = new_sat_state[['sat_x', 'sat_y', 'sat_z']].values
  actual = pos + dt * vel
  np.testing.assert_array_almost_equal(expected, actual)
