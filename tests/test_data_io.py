# Copyright (C) 2015 Swift Navigation Inc.
# Contact: Bhaskar Mookerji <mookerji@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

import pytest
import numpy as np
import pandas as pd

from gnss_analysis import data_io


def get_first_ephemeris(ephs):
  """
  Get a DataFrame containing the first non-NaN ephemeris for each satellite.

  Parameters
  ----------
  ephs : Panel
    An ephemeris Panel potentially containing NaN ephemerides.

  Returns
  -------
  DataFrame
    A DataFrame whose columns are satellites and rows are ephemeris fields.
    The fields are those of the first ephemeris in the input for which
    af0 is not NaN.
  """
  #TODO respect invalid/unhealthy ephemerides.
  sats_to_be_found = set(ephs.minor_axis)
  first_ephs = dict()
  for t in ephs.items:
    df = ephs.ix[t]
    for sat in sats_to_be_found:
      col = df[sat]
      if not np.isnan(col['af0']):
        first_ephs[sat] = col
  return pd.DataFrame(first_ephs)


def fill_in_ephs(ephs, first_ephs):
  """Fills in an ephemeris Panel so that there are no missing
  ephemerises.

  Parameters
  ----------
  ephs : Panel
    A Panel of ephemerises with potentially missing colums
  first_ephs : DataFrame
    A DataFrame of the first non-missing ephemerises for each satellite.

  Returns
  -------
  Panel
    The same panel as input, except the missing ephemerises are filled in with
    the most recent ephemeris if there is one, otherwise the first ephemeris.

  """
  #TODO respect invalid/unhealthy ephemerises
  new_ephs = ephs
  prev_eph = first_ephs
  for itm in ephs.iteritems():
    t = itm[0]
    df = itm[1]
    for sat in df.axes[1]:
      if np.isnan(df[sat]['af0']):
        df[sat] = prev_eph[sat]
    prev_eph = df
    new_ephs[t] = df
  return new_ephs


def test_fill_ephemeris(hdf5log):
  """
  Test the get_fst_ephs method, make sure it agrees with
  an alternative approach.
  """
  # load ephemerides from the store
  ephs = hdf5log.ephemerides
  actual = data_io.fill_in_ephemerides(ephs)
  # fill the ephemerides using legacy code
  first_ephs = get_first_ephemeris(ephs)
  expected = fill_in_ephs(ephs, first_ephs)
  assert np.all(actual == expected)
