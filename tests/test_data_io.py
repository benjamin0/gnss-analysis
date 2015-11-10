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


def alternative_get_first_ephermeris(ephs):
  """
  A method which computes the first valid
  ephermeris values in a Panel of ephermerides.
  """
  iter_ephs = ephs.iteritems()
  # grab the first item and copy it
  first = iter_ephs.next()[1].copy()
  for _, eph in iter_ephs:
    # if we've found valid values for all satellites, break
    if np.all(np.isfinite(first)):
      break
    # fill in any nan values with values from the next eph
    first.fillna(eph, inplace=True)
  # make sure we didn't miss anything
  if np.any(np.isnan(first)):
    raise ValueError("Incomplete set of ephermerides")
  return first


def test_get_first_ephemeris(hdf5log):
  """
  Test the get_fst_ephs method, make sure it agrees with
  an alternative approach.
  """
  # load ephemerides from the store
  ephs = hdf5log.ephemerides
  actual = data_io.get_first_ephemeris(ephs)
  # the columns should be integers which correspond to
  # satellite prn numbers.
  assert isinstance(actual.columns, pd.core.index.Int64Index)
  # try an alternative method and make sure they are
  # the same.
  expected = alternative_get_first_ephermeris(ephs)
  assert np.all(actual == expected)
