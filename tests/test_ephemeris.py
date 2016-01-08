import pytest
import numpy as np
import pandas as pd

from swiftnav import time as gpstime
from swiftnav import ephemeris as swiftnav_ephemeris

import gnss_analysis.constants as c

from gnss_analysis import ephemeris, observations, locations


def test_calc_sat_state(ephemerides):
  """
  This confirms that the calculations made by the gnss_analysis
  version of calc_sat_state matches the c implementation.
  """
  n = ephemerides.shape[0]
  time = ephemerides[['wn', 'tow']].copy()
  time['tow'] += 1e-3 * np.random.normal(size=n)

  ephemerides['toc_wn'] = ephemerides['toe_wn']

  # make sure we've added a fit interval
  assert 'fit_interval' in ephemerides

  actual = ephemeris.calc_sat_state(ephemerides, time)

  for i in range(n):
    eph = ephemerides.iloc[[i]]
    eph_obj = observations.mk_ephemeris(eph.reset_index())
    # compute the distance to satellite at the time of observation
    gpst = gpstime.GpsTime(wn=time.iloc[i]['wn'], tow=time.iloc[i]['tow'])
    expected = eph_obj.calc_sat_state(gpst)
    another = ephemeris.calc_sat_state(eph, time.iloc[i])
    act = actual.iloc[i, :]

    # make sure bulk and individual sat state error computations
    # are identical
    assert np.all(act == another)
    pos, vel, clock_error, clock_error_rate = expected
    # make sure positions agree within a micrometer
    assert np.all(np.abs(act[['sat_x', 'sat_y', 'sat_z']] - pos) < 1e-6)
    # make sure velocities agree within a micrometer/second
    assert np.all(np.abs(act[['sat_v_x', 'sat_v_y', 'sat_v_z']] - vel) < 1e-6)
    # make sure clock error agree within a femtosecond
    assert np.abs(clock_error - act['sat_clock_error']) < 1e-12
    # make sure clock error rate agrees within a femtosecond / second
    assert np.abs(clock_error_rate - act['sat_clock_error_rate']) < 1e-12

  # Test with the time empbedded in the ephemerides object
  ephemerides['wn'] = time['wn']
  ephemerides['tow'] = time['tow']
  another = ephemeris.calc_sat_state(ephemerides)
  np.testing.assert_array_equal(actual.values, another.values)

  # try with a time that is equivalent, but definied with negative
  # time of week and make sure that doesn't break anything
  neg_tow = time.copy()
  neg_tow['wn'] += 1
  neg_tow['tow'] -= c.WEEK_SECS
  assert np.all(0 == ephemeris.gpsdifftime(neg_tow['wn'], neg_tow['tow'],
                                           time['wn'], time['tow']))
  another = ephemeris.calc_sat_state(ephemerides, neg_tow)
  # make sure nothing (except the wn and tow) are different.
  np.testing.assert_array_equal(another.drop(['sat_tow', 'sat_wn'], axis=1),
                                actual.drop(['sat_tow', 'sat_wn'], axis=1))



def test_sagnac_rotation(ephemerides):
  time = ephemerides[['wn', 'tow']].copy()
  n = ephemerides.shape[0]
  time['tow'] += 40 + 1e-3 * np.random.normal(size=n)

  sat_state = ephemeris.calc_sat_state(ephemerides, time)

  time_of_flight = np.random.normal(0.07, 0.01, size=n)
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
  ref_time = ephemerides.iloc[0][['wn', 'tow']].copy()
  ref_time['tow'] += 40

  expected = ephemeris.calc_sat_state(ephemerides, ref_time)

  # make some random transmission times and compute the
  # corresponding raw_pseudoranges according to the method used in
  # track.c:calc_navigation_measurements
  np.random.seed(1982)
  tot = ref_time['tow'] - np.random.normal(0.08, 0.01, size=ephemerides.shape[0])
  expected['raw_pseudorange'] = (ref_time['tow'] - tot) * c.GPS_C
  expected['tot'] = tot
  expected['pseudorange'] = expected['raw_pseudorange'] + expected.sat_clock_error * c.GPS_C
  def equals_expected(to_test):
    _, to_test = expected.align(to_test, 'left')
    return np.all(expected == to_test)

  orig = ephemerides.copy()
  orig['raw_pseudorange'] = expected['raw_pseudorange']
  orig['tow'] = ref_time['tow']

  actual = ephemeris.add_satellite_state(orig)
  assert equals_expected(actual)

  # try with ephemerides passed in seperate
  actual = ephemeris.add_satellite_state(orig, ephemerides)
  assert equals_expected(actual)

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
  another['tow'] += 5
  assert equals_expected(actual)


def test_time_of_transmission_vectorization(ephemerides):
  """
  Tests to make sure that the vectorized version matches
  the output if you pipe individual satellites through to
  ensure there aren't strange vectorization issues.
  """
  ref_time = ephemerides.iloc[0][['wn', 'tow']].copy()
  ref_time['tow'] += 40
  ref_loc = locations.NOVATEL_ABSOLUTE

  # a pretty straight forward consistency check that makes sure
  # vectorized solves are the same as individual solves.
  full, sat_states = ephemeris.time_of_transmission(ephemerides,
                                                   ref_time,
                                                   ref_loc)
  for i in range(full.shape[0]):
    single, sat_state = ephemeris.time_of_transmission(ephemerides.iloc[[i]],
                                                       ref_time, ref_loc)
    assert np.all(full.iloc[[i]] == single)
    assert np.all(sat_states.iloc[[i]] == sat_state)

  # make sure the default iterations is basically
  # the same as after 10
  ten, _ = ephemeris.time_of_transmission(ephemerides,
                                       ref_time,
                                       ref_loc,
                                       max_iterations=10,
                                       tol=1e-12)
  assert np.all(np.abs(ten - full) <= 1e-9)

  # make sure the algorithm converged and is less than
  # the convergence tolerance
  nine, _ = ephemeris.time_of_transmission(ephemerides,
                                       ref_time,
                                       ref_loc,
                                       max_iterations=9,
                                       tol=0)
  np.all(np.abs(nine['tow'] - ten['tow']) < 1e-12)


def test_time_of_transmission_regression(ephemerides):
  """
  Starts with some satellite pos and random times of transmission,
  calculates the times of arrival at a reference location then
  backs out the time of transmission to make sure it matches.
  """
  n = ephemerides.shape[0]
  tot = ephemerides[['wn', 'tow']].copy()
  tot['tow'] += 40 + np.random.normal(0.1, 0.1, size=n)
  ref_loc = locations.NOVATEL_ABSOLUTE
  sat_state = ephemeris.calc_sat_state(ephemerides, tot)
  sat_pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values.copy()

  # compute the line of sight position, we'll use the time of
  # flight to then get the time of arrival.
  los_pos = sat_pos
  for i in range(3):
    tof = np.linalg.norm(los_pos - ref_loc, axis=1) / c.GPS_C
    los_pos = ephemeris.sagnac_rotation(sat_pos, time_of_flight=tof)

  # now run time_of_transmission and make sure we get the original back.
  toa = tot.copy()
  toa['tow'] += tof
  actual_tot, actual_state = ephemeris.time_of_transmission(ephemerides,
                                          time_of_arrival=toa,
                                          ref_loc=ref_loc)

  np.testing.assert_array_almost_equal(tot['tow'].values,
                                       actual_tot['tow'].values)
  np.testing.assert_array_almost_equal(sat_state.values,
                                       actual_state.values, 3)


def test_gpsdifftime():
  tests = [({'end_wn': 0, 'end_tow': 1, 'start_wn': 0, 'start_tow': 0}, 1),
           # trivial case.
           ({'end_wn': 0, 'end_tow': 1, 'start_wn': 0, 'start_tow': 1}, 0),
           # These next two make sure that the difference between two times, one of which
           # has a negative time of week is equivalent to the difference between two
           # properly defined gps times.
           ({'end_wn': 0, 'end_tow': 1, 'start_wn': 0, 'start_tow':-1}, 2),
           ({'end_wn': 0, 'end_tow': 1, 'start_wn':-1, 'start_tow': c.WEEK_SECS - 1}, 2),
           # Make sure the same time of week but incremented wn is exactly one week apart.
           ({'end_wn': 1, 'end_tow': 0, 'start_wn': 0, 'start_tow': 0}, c.WEEK_SECS),
           ({'end_wn': 1, 'end_tow': 1, 'start_wn': 0, 'start_tow': 0}, c.WEEK_SECS + 1),
           # Make sure negative differences are handled.
           ({'end_wn': 0, 'end_tow': 0, 'start_wn': 0, 'start_tow': 1}, -1),
           ]

  for params, expected in tests:
    assert expected == ephemeris.gpsdifftime(**params)


def test_sat_velocity(ephemerides):
  # computes a set of satellite states, then uses the state
  # velocity to predict the state at a nearby time and compares
  # it to the actual sat state at that time.
  ref_time = ephemerides.iloc[0][['wn', 'tow']].copy()
  ref_time['tow'] += 40

  sat_state = ephemeris.calc_sat_state(ephemerides, ref_time)
  vel = sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']].values
  pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values

  dt = 1e-6
  new_time = ref_time.copy()
  new_time['tow'] += dt
  new_sat_state = ephemeris.calc_sat_state(ephemerides, new_time)

  expected = new_sat_state[['sat_x', 'sat_y', 'sat_z']].values
  actual = pos + dt * vel
  np.testing.assert_array_almost_equal(expected, actual)
