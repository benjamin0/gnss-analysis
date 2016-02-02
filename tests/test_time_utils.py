import datetime
import numpy as np
import pandas as pd

from gnss_analysis import time_utils
from gnss_analysis import constants as c


def test_diff_time():
  tests = [({'end': {'wn': 0, 'tow': 1}, 'start': {'wn': 0, 'tow': 0}}, 1),
           # trivial case.
           ({'end': {'wn': 0, 'tow': 1}, 'start': {'wn': 0, 'tow': 1}}, 0),
           # These next two make sure that the difference between two times, one of which
           # has a negative time of week is equivalent to the difference between two
           # properly defined gps times.
           ({'end': {'wn': 0, 'tow': 1}, 'start': {'wn': 0, 'tow':-1}}, 2),
           ({'end': {'wn': 0, 'tow': 1}, 'start': {'wn':-1, 'tow': c.WEEK_SECS - 1}}, 2),
           # Make sure the same time of week but incremented wn is exactly one week apart.
           ({'end': {'wn': 1, 'tow': 0}, 'start': {'wn': 0, 'tow': 0}}, c.WEEK_SECS),
           ({'end': {'wn': 1, 'tow': 1}, 'start': {'wn': 0, 'tow': 0}}, c.WEEK_SECS + 1),
           # Make sure negative differences are handled.
           ({'end': {'wn': 0, 'tow': 0}, 'start': {'wn': 0, 'tow': 1}}, -1),
           ]

  for params, expected in tests:
    assert expected == time_utils.diff_time(**params)


def assert_time_equal(x, y):
  x = pd.to_datetime(x)
  if hasattr(x, 'to_datetime64'):
    x = x.to_datetime64()
  np.testing.assert_array_equal(x, y)


def assert_time_not_equal(x, y):
  try:
    assert_time_equal(x, y)
    assert False
  except AssertionError:
    pass


# def test_gps_to_utc_roundtrip():
#
#   test_cases = [(datetime.datetime(2000, 1, 1), 14),
#                 (datetime.datetime(2016, 1, 20), 17),
#                 (c.GPS_WEEK_0, 0),
#                 (pd.date_range(start=datetime.datetime(2016, 1, 1),
#                                end=datetime.datetime(2016, 1, 20)), 17),
#                 ]
#
#   for d, delta_utc in test_cases:
#     gpst = time_utils.utc_to_gpst(d, delta_utc)
#     actual = time_utils.gpst_to_utc(gpst, delta_utc)
#     # check that they are the same, the result will always be a np.datetime64
#     # object, so we first need to convert the original value.
#     assert_time_equal(d, actual)
#     bad = time_utils.gpst_to_utc(gpst, delta_utc + 1)
#     assert_time_not_equal(d, bad)


def test_tow_datetime_roundtrip():

  test_cases = [datetime.datetime(2000, 1, 1),
                datetime.datetime(2016, 1, 20),
                c.GPS_WEEK_0,
                pd.date_range(start=datetime.datetime(2016, 1, 1),
                              end=datetime.datetime(2016, 1, 20)),
                ]

  for d in test_cases:
    wn_tow = time_utils.datetime_to_tow(d)
    actual = time_utils.tow_to_datetime(**wn_tow)
    # check that they are the same, the result will always be a np.datetime64
    # object, so we first need to convert the original value.
    assert_time_equal(d, actual)
    and_back = time_utils.datetime_to_tow(actual)
    np.testing.assert_array_equal(and_back['wn'], wn_tow['wn'])
    np.testing.assert_array_equal(and_back['tow'], wn_tow['tow'])


def test_timedelta_to_seconds():
  test_cases = [(np.timedelta64(1, 'ns'), 1.e-9),
                (np.array(1e9).astype('timedelta64[ns]'), 1.),
                (np.array(0).astype('timedelta64[ns]'), 0.),
                (np.array([np.timedelta64(1, 'ns'), np.timedelta64('NaT', 'ns')]),
                 (1e-9, np.nan))]

  for td, fl in test_cases:
    actual_float = time_utils.seconds_from_timedelta(td)
    np.testing.assert_array_equal(actual_float, fl)

    actual_td = time_utils.timedelta_from_seconds(fl)
    np.testing.assert_array_equal(actual_td, td)

    roundtrip = time_utils.seconds_from_timedelta(actual_td)
    np.testing.assert_array_equal(roundtrip, fl)
