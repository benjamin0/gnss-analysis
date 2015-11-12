#!/usr/bin/env python
# Copyright (C) 2015 Swift Navigation Inc.
# Contact: Bhaskar Mookerji <mookerji@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""Basic integration tests for pandas GPS time interpolation
utilities.

"""

import os
import pytest
import numpy as np
import pandas as pd

from pandas.tslib import Timestamp, Timedelta

import gnss_analysis.hitl_table_utils as t

def test_interpolate_gps_time(hdf5log):
  """
  Loads a known hdf5 log from the data directory and
  compares it to known values.
  """
  idx = hdf5log.rover_spp.T.host_offset.reset_index()
  model = t.interpolate_gpst_model(idx)
  assert isinstance(model, pd.stats.ols.OLS)
  assert np.allclose([model.beta.x, model.beta.intercept],
                     [1.00000368376, -64.2579561376])
  init_offset = hdf5log.rover_spp.T.host_offset[0]
  init_date = hdf5log.rover_spp.T.index[0]
  f = lambda t1: t.apply_gps_time(t1*t.MSEC_TO_SEC, init_date, model)
  dates = hdf5log.rover_logs.T.host_offset.apply(f)
  l = dates.tolist()
  start, end = l[0], l[-1]
  assert start == Timestamp("2015-04-29 23:32:55.272075")
  assert end == Timestamp("2015-04-29 23:57:46.457568")
  init_secs_offset \
    = hdf5log.rover_spp.T.host_offset[0] - hdf5log.rover_logs.T.index[0]
  assert np.allclose([init_secs_offset*t.MSEC_TO_SEC], [55.859])
  assert (init_date - start) == Timedelta('0 days 00:00:55.848925')
  assert (end - init_date) == Timedelta('0 days 00:23:55.336568')
  assert pd.DatetimeIndex(dates).is_monotonic_increasing
  assert dates.shape == (2457,)


@pytest.mark.slow
def test_gps_time_col(hdf5log):
  tables = ['rover_iar_state', 'rover_logs', 'rover_tracking']
  t.get_gps_time_col(hdf5log, tables)
  gpst = hdf5log.rover_iar_state.T.approx_gps_time
  assert gpst.shape == (1487,)
  assert pd.DatetimeIndex(gpst).is_monotonic_increasing
  gpst = hdf5log.rover_logs.T.approx_gps_time
  assert gpst.shape == (2457,)
  assert pd.DatetimeIndex(gpst).is_monotonic_increasing
  gpst = hdf5log.rover_tracking[:, 'approx_gps_time', :]
  assert gpst.shape == (32, 7248)


def test_gaps():
  td = pd.DatetimeIndex(['2015-05-21 21:24:52.200000',
                         '2015-05-21 21:24:52.400000',
                         '2015-05-21 21:24:52.600000',
                         '2015-05-21 21:25:52.800000',
                         '2015-05-21 21:27:53'],
                        dtype='datetime64[ns]',
                        freq=None, tz=None)
  assert np.allclose(t.find_largest_gaps(td, 10).values,
                     [120.2, 60.2, 0.2, 0.2])
  assert np.allclose(t.find_largest_gaps(td, 1).values, [120.2])
  assert np.allclose(t.find_largest_gaps(td[0:2], 10).values, [0.2])
  assert np.allclose(t.find_largest_gaps(td[0:1], 10).values, [0])
