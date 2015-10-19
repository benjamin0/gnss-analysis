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

"""This tool runs a software-in-the-loop (SITL) simulation of the
libswiftnav filters.

<class 'pandas.io.pytables.HDFStore'>
File path: data/serial-link-20150506-175750.log.json.new_fields.hdf5
/ephemerides               wide         (shape->[2,26,8])
/base_obs                  wide         (shape->[7272,4,8])
/rover_obs                 wide         (shape->[7241,4,8])
/rover_rtk_ecef            frame        (shape->[7,14545])
/rover_rtk_ned             frame        (shape->[8,14545])
/rover_spp_sim             frame        (shape->[7237,3])
/base_spp_sim              frame        (shape->[7237,3])
/rover_sdiff               frame        (shape->[7237,3])
/base_ddiff                frame        (shape->[7237,3])
/rover_ddiff               frame        (shape->[7237,3])

"""

from gnss_analysis.dgnss import process_dgnss
from gnss_analysis.utils import validate_table_schema
import pandas as pd
import sys


# Expected tables for the HDF5 store used for the dgnss simulation
_EXPECTED_KEYS = ['ephemerides',
                  'base_obs',
                  'rover_obs',
                  'rover_rtk_ecef',
                  'rover_rtk_ned',
                  'rover_spp_sim',
                  'base_spp_sim',
                  'rover_sdiff',
                  'base_sdiff',
                  'rover_ddiff']


def main():
  import argparse
  parser = argparse.ArgumentParser(description='Swift Nav DGNSS sim tool.')
  parser.add_argument('file',
                      help='Specify the log file to use.')
  args = parser.parse_args()
  log_datafile = args.file
  with pd.HDFStore(log_datafile) as store:
    # assert validate_table_schema(store, _EXPECTED_KEYS), \
    #     "Invalid schema! %s" % store.keys()
    try:
      baselines = process_dgnss(store)
      store.put('rover_rtk_ecef_sim', pd.DataFrame(baselines))
    except (KeyboardInterrupt, SystemExit):
      print "Exiting!"
      sys.exit()

if __name__ == "__main__":
  main()
