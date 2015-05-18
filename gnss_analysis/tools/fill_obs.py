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

"""The fill observations tool takes the generated HDF5 output from
records2table and fills in derived observation quantities, such as
single-difference (sdiff) and double-difference (ddiff)
observations. It will also selectively backfill in missing data, such
as missing ephemeris data, at least when valid data is available from
a previous timestep. Hopefully you should only need to do this once.

For example, passing an HDF5 table that looks like
<class 'pandas.io.pytables.HDFStore'>
File path: data/serial-link-20150506-175750.log.json.new_fields.hdf5
/base_obs                  wide         (shape->[7272,4,8])
/ephemerides               wide         (shape->[2,26,8])
/rover_obs                 wide         (shape->[7241,4,8])
/rover_rtk_ecef            frame        (shape->[7,14545])
/rover_rtk_ned             frame        (shape->[8,14545])
/rover_spp                 frame        (shape->[7237,3])

... will *add in* derived quantities such as the single difference and
double-difference observables:
<class 'pandas.io.pytables.HDFStore'>
File path: data/serial-link-20150506-175750.log.json.new_fields.hdf5
/base_obs                  wide         (shape->[7272,4,8])
/ephemerides               wide         (shape->[2,26,8])
/rover_obs                 wide         (shape->[7241,4,8])
/rover_rtk_ecef            frame        (shape->[7,14545])
/rover_rtk_ned             frame        (shape->[8,14545])
/rover_spp                 frame        (shape->[7237,3])
/sdiff                     frame        (shape->[7237,3])
/ddiff                     frame        (shape->[7237,3])

If derived observables already exist, it won't overwrite those unless
you specify the overwite flag.

"""

from gnss_analysis.utils import validate_table_schema
from gnss_analysis.observations import ffill_panel, fill_observations
import pandas as pd


_EXPECTED_KEYS = ['/base_obs',
                  '/ephemerides',
                  '/rover_obs',
                  '/rover_rtk_ecef',
                  '/rover_rtk_ned',
                  '/rover_spp']


def main():
  import argparse
  import sys
  parser = argparse.ArgumentParser(description='Swift Nav fill in derived obs.')
  parser.add_argument('file',
                      help='Specify the log file to use.')
  parser.add_argument('-n', '--num_records',
                      nargs=1,
                      default=[None],
                      help='Number of GPS observation records to process.')
  parser.add_argument('-v', '--verbose',
                      action='store_true',
                      help='Verbose output.')
  # TODO (Buro): Add in handling for explicit overwrites. Currently,
  # this will fill in and overwrite (specifically sdiffs, etc.) that
  # you might have.
  #
  # parser.add_argument('-o', '--output',
  #                     nargs=1,
  #                     default=[None],
  #                     help='Test results output filename')
  # parser.add_argument('-w', '--overwrite',
  #                     action='store_true'
  #                     help='Overwrite .')
  args = parser.parse_args()
  log_datafile = args.file
  num_records = args.num_records[0]
  verbose = args.verbose
  with pd.HDFStore(log_datafile) as store:
    try:
      validate_table_schema(store, _EXPECTED_KEYS)
      if verbose:
        print "Verbose output specified..."
        print "Loading table %s, " % str(store)
      store.put('ephemerides_filled', ffill_panel(store.ephemerides))
      records = fill_observations(store, num_records, verbose)
      for k, new_item in records.iteritems():
        store.put(k, new_item)
    except (KeyboardInterrupt, SystemExit):
      print "Exiting!"
      sys.exit()
    finally:
      if verbose:
        print "\nWriting out table: %s" % str(store)
      store.close()

if __name__ == "__main__":
  main()
