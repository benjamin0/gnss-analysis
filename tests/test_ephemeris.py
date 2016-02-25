import pytest
import logging
import numpy as np
import pandas as pd

from swiftnav import time as gpstime
from swiftnav import ephemeris as swiftnav_ephemeris

import gnss_analysis.constants as c

from gnss_analysis import ephemeris, observations, locations, time_utils
from gnss_analysis.io import common


def test_calc_sat_state(ephemerides):
  """
  This confirms that the calculations made by the gnss_analysis
  version of calc_sat_state matches the c implementation.
  """
  n = ephemerides.shape[0]
  time = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
  noise = 1e-3 * np.random.normal(size=n)
  time += time_utils.timedelta_from_seconds(noise)

  # make sure we've added a fit interval
  assert 'fit_interval' in ephemerides

  actual = ephemeris.calc_sat_state(ephemerides, time)

  for i in range(n):
    eph = ephemerides.iloc[[i]]
    eph_obj = observations.mk_ephemeris(eph.reset_index())
    # compute the distance to satellite at the time of observation
    wn_tow = time_utils.datetime_to_tow(time[i])
    gpst = gpstime.GpsTime(**wn_tow)
    expected = eph_obj.calc_sat_state(gpst)
    another = ephemeris.calc_sat_state(eph, time[i])
    act = actual.iloc[[i], :]

    # make sure bulk and individual sat state error computations
    # are identical
    assert np.all(act == another)
    pos, vel, clock_error, clock_error_rate = expected
    # make sure positions agree within a mm
    assert np.all(np.abs(act[['sat_x', 'sat_y', 'sat_z']] - pos) < 1e-3)
    # make sure velocities agree within a micrometer/second
    assert np.all(np.abs(act[['sat_v_x', 'sat_v_y', 'sat_v_z']] - vel) < 1e-6)
    # make sure clock error agree within a femtosecond
    assert np.all(np.abs(clock_error - act['sat_clock_error']) < 1e-12)
    # make sure clock error rate agrees within a femtosecond / second
    assert np.all(np.abs(clock_error_rate - act['sat_clock_error_rate']) < 1e-12)

  # Test with the time empbedded in the ephemerides object
  ephemerides['time'] = time
  another = ephemeris.calc_sat_state(ephemerides)
  np.testing.assert_array_equal(actual.values, another.values)


def test_sagnac_rotation(ephemerides):
  time = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
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


def test_join_common_sats():
  one = pd.DataFrame({'one': np.arange(5.),
                      'shared': np.arange(5.) + 2},
                      index=pd.Index(np.arange(5), name='sid'))
  one['constellation'] = 'GPS'
  one['band'] = 1

  two = pd.DataFrame({'shared': np.arange(1, 6.) + 2,
                      'two': np.arange(1, 6.) + 3},
                     index=pd.Index(np.arange(1, 6), name='sid'))
  two['constellation'] = 'GPS'
  two['band'] = 1

  expected = pd.DataFrame({'one': np.arange(1, 5.),
                           'shared': np.arange(1, 5.) + 2,
                           'two': np.arange(1, 5.) + 3},
                      index=pd.Index(np.arange(1, 5), name='sid'))
  expected['constellation'] = 'GPS'
  expected['band'] = 1

  actual = ephemeris._join_common_sats(one, two)
  assert np.all(actual == expected)

  # try again with a nan in the mix
  one['one'].values[2] = np.nan
  expected['one'].values[1] = np.nan
  actual = ephemeris._join_common_sats(one, two)
  np.testing.assert_array_equal(actual.drop('constellation', axis=1).values,
                                expected.drop('constellation', axis=1).values)

  # make sure that if one is a subset of two all the values of
  # are updated.
  one = two.copy()
  is_float =  (one.dtypes == 'float64').values
  one.values[:, is_float] += np.random.normal()
  should_be_two = ephemeris._join_common_sats(one, two)
  np.testing.assert_array_equal(should_be_two.drop('constellation', axis=1).values,
                                two.drop('constellation', axis=1).values)

  # make sure even when none of the columns for one are used, the
  # inner set of indices still is
  one = two.copy()
  one = one.iloc[1:]
  one.values[:, is_float] += np.random.normal()
  missing_row = ephemeris._join_common_sats(one, two)
  np.testing.assert_array_equal(missing_row.drop('constellation', axis=1),
                                two.iloc[1:].drop('constellation', axis=1))


def test_add_satellite_state(ephemerides):
  """
  Makes sure that add_satellite_state performs as expected
  """
  ref_time = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
  ref_time = ref_time + time_utils.timedelta_from_seconds(40)

  # make some random transmission times
  np.random.seed(1982)
  noise = np.random.normal(0.08, 0.01, size=ephemerides.shape[0])
  tot = ref_time - time_utils.timedelta_from_seconds(noise)
  # compute the corresponding raw_pseudoranges according to the method used in
  # track.c:calc_navigation_measurements
  expected = ephemeris.calc_sat_state(ephemerides, tot)

  tof = time_utils.seconds_from_timedelta(ref_time - tot)
  expected['raw_pseudorange'] = (tof - expected.sat_clock_error) * c.GPS_C
  expected['raw_doppler'] = np.random.normal(100., 100., size=expected.shape[0])
  expected['tot'] = tot
  expected['pseudorange'] = expected['raw_pseudorange'].copy()
  expected['pseudorange'] += expected.sat_clock_error * c.GPS_C
  expected['doppler'] = expected['raw_doppler'].copy()
  expected['doppler'] += expected.sat_clock_error_rate * c.GPS_L1_HZ
  expected.reset_index()
  expected['constellation'] = 'GPS'
  expected['band'] = 1
  expected = common.normalize(expected)


  def equals_expected(to_test):
    _, to_test = expected.align(to_test, 'left')

    for (k, x), (_, y) in zip(expected.iteritems(),
                              to_test.iteritems()):
      # diff is a time delta
      if x.dtype.kind == 'M':
        diff = x - y
        if np.any(np.abs(diff) > np.timedelta64(1, 'ns')):
          return False
      elif x.dtype.kind == 'O':
        # string comparisons must be equal
        np.testing.assert_array_equal(x.values, y.values)
      else:
        diff = x - y
        # otherwise for float value we look for values
        # that agree up to 5 decimals.
        if np.any(np.abs(diff) > 1e-5):
          return False
    return True

  orig = ephemerides.copy()
  orig['raw_pseudorange'] = expected['raw_pseudorange'].values.copy()
  orig['raw_doppler'] = expected['raw_doppler'].values.copy()
  orig['time'] = ref_time.copy()
  orig.reset_index()
  orig['constellation'] = 'GPS'
  orig['band'] = 1
  orig = common.normalize(orig)

  actual = ephemeris.add_satellite_state(orig)
  assert equals_expected(actual)

  # try with ephemerides passed in seperate
  actual = ephemeris.add_satellite_state(orig, ephemerides)
  equals_expected(actual)

  # make sure it's idempotent
  once = ephemeris.add_satellite_state(orig, ephemerides)
  twice = ephemeris.add_satellite_state(once)
  assert equals_expected(twice)

  bad_ephemerides = orig.copy()
  bad_ephemerides['af0'] += 10
  bad = ephemeris.add_satellite_state(bad_ephemerides)
  assert not equals_expected(bad)
  # make sure the explicit ephemerides overrides the bad ones
  good = ephemeris.add_satellite_state(bad_ephemerides, ephemerides)
  assert equals_expected(good)



def test_time_of_flight(ephemerides):
  """
  Tests to make sure that the time of flight functions work
  including testing the vectorization.
  """

  for tof_func in [ephemeris.time_of_flight_from_tot,
                   ephemeris.time_of_flight_from_toa]:

    ref_time = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
    ref_time += time_utils.timedelta_from_seconds(40)
    ref_loc = locations.NOVATEL_ABSOLUTE

    # a pretty straight forward consistency check that makes sure
    # vectorized solves are the same as individual solves.
    full = tof_func(ephemerides, ref_time, ref_loc)
    for i in range(full.shape[0]):
      single = tof_func(ephemerides.iloc[[i]],
                        ref_time, ref_loc)
      assert full[i] == single[0]

    # make sure the default iterations is basically
    # the same as after 10
    ten = tof_func(ephemerides, ref_time, ref_loc,
                   max_iterations=10, tol=1e-12)
    np.testing.assert_allclose(ten, full, atol=1e-9)

    # make sure the algorithm converged and is less than
    # the convergence tolerance
    nine = tof_func(ephemerides, ref_time, ref_loc,
                    max_iterations=9, tol=0)
    np.testing.assert_allclose(ten, nine, atol=1e-12)


def test_time_of_roundtrip(ephemerides):
  """
  Starts with some satellite pos and random times of transmission,
  calculates the times of arrival at a reference location then
  backs out the time of transmission to make sure it matches.
  """
  n = ephemerides.shape[0]
  tot = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
  noise = np.random.normal(0.1, 0.1, size=n)
  tot += 40 + time_utils.timedelta_from_seconds(noise)
  ref_loc = locations.NOVATEL_ABSOLUTE

  # compute the time of arrival for a signal leaving at transmission time from
  # the satellites and arriving at ref_loc.
  tot_tof = ephemeris.time_of_flight_from_tot(ephemerides, tot, ref_loc)
  toa = tot + time_utils.timedelta_from_seconds(tot_tof)
  # now run time_of_transmission and make sure we get the original back.
  toa_tof = ephemeris.time_of_flight_from_toa(ephemerides,
                                               toa=toa,
                                               ref_loc=ref_loc)
  roundtrip_tot = toa - time_utils.timedelta_from_seconds(toa_tof)
  error_seconds = time_utils.seconds_from_timedelta(roundtrip_tot - tot)
  assert np.all(error_seconds == 0.)


def test_sat_velocity(ephemerides):
  # computes a set of satellite states, then uses the state
  # velocity to predict the state at a nearby time and compares
  # it to the actual sat state at that time.
  ref_time = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')
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
