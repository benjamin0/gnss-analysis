#!/usr/bin/env python
# Copyright (C) 2015 Swift Navigation Inc.
# Contact: Ian Horn <ian@swiftnav.com>
#          Bhaskar Mookerji <mookerji@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.


import gnss_analysis.dgnss as d
import os
import pandas as pd
import pytest


@pytest.mark.skipif(True, reason="Fill in later!")
def test_dgnss_sanity():
  log_datafile \
    = "./data/serial_link_log_20150314-190228_dl_sat_fail_test1.log.json.dat"
  filename = log_datafile + ".hdf5"
  assert os.path.isfile(filename)
  with pd.HDFStore(filename) as store:
    assert store
    assert isinstance(store.base_obs, pd.Panel)
    assert store.keys()
    assert d.process_dgnss(store)
    assert store.keys()
