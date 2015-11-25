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


"""Runs the DGNSS filters, given post-processed observations.

"""

from gnss_analysis.constants import MIN_SATS
from swiftnav.observation import SingleDiff
import numpy as np
import pandas as pd
import swiftnav.dgnss_management as mgmt


def mk_sdiff(x):
  """Make a libswiftnav sdiff_t from an object with the same elements,
  if possible, otherwise returning numpy.nan.  We assume here that if
  C1 is not nan, then nothing else is nan, except potentially D1.

  Parameters
  ----------
  x : Series
    A series with all of the fields needed for a libswiftnav sdiff_t.

  Returns
  -------
  SingleDiff or numpy.nan
    If C1 is nan, we return nan, otherwise return a SingleDiff from
    the Series.

  """
  if np.isnan(x.pseudorange):
    return np.nan
  return SingleDiff(x.pseudorange,
                    x.carrier_phase,
                    x.doppler,
                    np.array([x.sat_pos_x, x.sat_pos_y, x.sat_pos_z]),
                    np.array([x.sat_vel_x, x.sat_vel_y, x.sat_vel_z]),
                    x.snr,
                    x.prn)

def get_filter_state():
  initial_sats = mgmt.get_sats_management()[1]
  kf_means = mgmt.get_amb_kf_mean()
  kf_covs = mgmt.get_amb_kf_cov2()
  iar_ambs = mgmt.dgnss_iar_MLE_ambs()
  print iar_ambs

def process_dgnss(table):
  """Software simulate the RTK filter given a GPS timestamped
  observations from a HITL test. See:
  piksi_firmware/src/solution.c:process_matched_obs.

  For the time being, I've left out things like reset logic and
  initialization, as is used in the firmware, and some vector/pandas
  optimizations. In the future, we may want to (re)introduce those
  things.

  Parameters
  ----------
  table : pandas.io.pytables.HDFStore
    Pandas table

  """
  assert isinstance(table, pd.HDFStore), "Not a Pandas table!"
  init_done = False
  results = {}
  for timestamp, sdiff_t in table.sdiffs.iteritems():
    rover_spp_sim_t = table.rover_spp_sim.get(timestamp, None)
    base_spp_sim_t = table.base_spp_sim.get(timestamp, None)
    if not (rover_spp_sim_t is None or base_spp_sim_t is None):
      n_sds = len(sdiff_t)
      sdiff_t = pd.Series(dict((prn, mk_sdiff(s)) for prn, s in sdiff_t.iteritems()))
      if not init_done and n_sds >= MIN_SATS:
        print "Initializing DGNSS!"
        mgmt.dgnss_init(sdiff_t, base_spp_sim_t.values)
        init_done = True
      elif n_sds >= MIN_SATS:
        get_filter_state()
        mgmt.dgnss_update(sdiff_t, base_spp_sim_t.values)
        b = None
        num_used, b = mgmt.dgnss_fixed_baseline(sdiff_t, base_spp_sim_t.values)
        #print b
        results[timestamp] = b
  return results
