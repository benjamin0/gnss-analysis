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

"""Utilities for GNSS observations: interop with libswiftnav
constructions and creating derived quantities, such as single
differenced (sdiff) and double-differenced (ddiff) observations.

TODO (BURO): Currently, this file manually iterates through an
existing collection, storing all its intermediate results in a
dictionary and *then* writing to a Pandas table. This can obviously be
vector refactorized later.

"""

import time
import datetime
import numpy as np
import pandas as pd
import swiftnav.time as gpstime

from swiftnav.pvt import calc_PVT_ as calc_PVT
from swiftnav.track import NavigationMeasurement
from swiftnav.time import GpsTime
from swiftnav.ephemeris import Ephemeris
from swiftnav.observation import SingleDiff

from gnss_analysis.constants import *
from gnss_analysis import time_utils

###############################################################################
# Misc. libswiftnav object constructors


def match_obs(timestamp, obs):
  """Finds a matching base station observation.

  """
  return obs.get(timestamp, None)

# TODO (Buro): Does this have to be prn+1?????

def mk_ephemeris(eph):
  """Construct a libswiftnav ephemeris. Intended for use with a Pandas
  Series of an ephemeris.

  Parameters
  ----------
  eph : object
    Any object with structured ephemeris data (i.e., either a
    MsgEphemeris from libsbp or a Series object.)
  """
  kepler_vars = {'tgd': 'tgd',
                 'crs': 'c_rs',
                 'crc': 'c_rc',
                 'cuc': 'c_uc',
                 'cus': 'c_us',
                 'cic': 'c_ic',
                 'cis': 'c_is',
                 'dn': 'dn',
                 'm0': 'm0',
                 'ecc': 'ecc',
                 'sqrta': 'sqrta',
                 'omega0': 'omega0',
                 'omegadot': 'omegadot',
                 'w': 'w',
                 'inc': 'inc',
                 'inc_dot': 'inc_dot',
                 'af0': 'af0',
                 'af1': 'af1',
                 'af2': 'af2',
                 'iode': 'iode',
                 'iodc': 'iodc'}
  kepler = {k: eph.get(v) for k, v in kepler_vars.iteritems()}

  kepler['toc'] = time_utils.datetime_to_tow(eph['toc'])
  # we assume L1 signals for the moment
  band = eph.get('band', 1)
  if np.isnan(band):
    band = 1
  assert band == 1
  return Ephemeris(toe=time_utils.datetime_to_tow(eph['toe']),
                   valid=eph['valid'],
                   healthy=eph['healthy'],
                   kepler=kepler,
                   ura=eph['ura'],
                   fit_interval=eph['fit_interval'],
                   sid={'sat': eph.sat,
                        'code': band - 1})


def ffill_panel(panel, axis=1):
  """Returns a new panel, forward filling a minor axis column in a
  Pandas panel with the most recent valid value.

  """
  new_panel = panel.copy()
  # Tried and failed to make this more concise. Apparently, you can't
  # fillna across an entire Panel:
  # https://github.com/pydata/pandas/issues/8251
  for i in new_panel.minor_axis:
    new_panel[:, :, i].ffill(axis=axis, inplace=True)
  return new_panel


def match_ephemeris(timestamp, eph, interval=EPHEMERIS_TOL):
  """Returns the most recent ephemeris within some time interval in the
  past, or None, if unavailable. Should handle satellite health.

  """
  return eph.ix[-1]


###############################################################################
# Differenced observations

# TODO (Buro): Replace with function exposed/called from
# libswiftnav/libswiftnav-python.

def mk_single_diff(nav_meas_rover, nav_meas_base):
  """Constructs single diffs from the rover and base station navigation
  measurements.

  Parameters
  ----------
  nav_meas_rover : object

  nav_meas_base : object


  Returns
  ----------

  """
  sdiffs = {}
  for prn, rover_meas in nav_meas_rover.iteritems():
    if nav_meas_base.get(prn, None):
      b = nav_meas_base[prn]
      sdiffs[prn] = SingleDiff(rover_meas.pseudorange - b.pseudorange,
                               rover_meas.carrier_phase - b.carrier_phase,
                               rover_meas.raw_doppler - b.raw_doppler,
                               np.array(rover_meas.sat_pos),
                               np.array(rover_meas.sat_vel),
                               min(rover_meas.snr, b.snr),
                               rover_meas.prn)
  return sdiffs


def mk_double_diff():
  """ Constructs double-differenced observations.

  Parameters
  ----------

  Returns
  ----------

  """
  raise NoImplementedError("Double Differencing function not yet implemented!")


###############################################################################
# Operations on navigation measurements


def mk_nav_measurement(gpst, prn, obs_t, eph_t):
  """Create a navigation measurement using an observation and a
  satellite ephemeris.

  Parameters
  ----------
  gpst : object

  prn : object

  obs : object

  eph : object

  """
  # Check if we have an ephemeris for this satellite, we will
  # need this to fill in satellite position etc. parameters.
  sat_state = eph_t.calc_sat_state(gpst)
  sat_pos, sat_vel, clock_err, clock_rate_err = sat_state
  # Apply corrections to the raw pseudorange
  pseudorange = obs_t.P + clock_err * GPS_C
  dop = np.nan
  lock_time = np.nan
  nm = NavigationMeasurement(obs_t.P,# raw_pseudorange
                             pseudorange,# pseudorange
                             obs_t.L,# carrier_phase
                             dop,# raw_doppler
                             dop,# doppler
                             sat_pos,# sat_pos
                             sat_vel,# sat_vel
                             obs_t.cn0,# snr
                             lock_time,# lock_time
                             # TODO: gpst is NOT tot
                             gpst,# tot
                             prn,# prn
                             obs_t.lock)# lock_counter
  return nm


def obs_table_to_nav_measurement(timestamp, obs_t, eph_t):
  """At a particular GPS timestep, produces a set of navigation
  measurements from an observation table and the ephemeris. Retr

  see: piksi_firmware/src/base_obs.c

  Parameters
  ----------
  timestamp : object

  obs_t : object

  eph_t : object


  Returns
  ----------

  """
  gpst = gpstime.datetime2gpst(timestamp)
  nms = {}
  # Iterate through each of the satellites in the observation,
  for prn, obs in obs_t.iteritems():
    if eph_t.get(prn) is None or eph_t.get(prn).empty:
      continue
    eph = mk_ephemeris(eph_t[prn])
    nms[prn] = mk_nav_measurement(gpst, prn, obs, eph)
  return nms


# TODO (Buro) double check to see if it matches version in libswiftnav
# with Dopper correction. In fact, just replace with function
# exposed/called from libswiftnav/libswiftnav-python.
def tdcp_doppler(nav_meas_old, nav_meas_new):
  """Returns measurement precise Doppler using time difference of
  carrier phase.

  see: libswiftnav/src/track.c:tdcp_dopper

  Parameters
  ----------
  nav_meas_rover : object

  nav_meas_base : object

  """
  t_old = gpstime.gpst_components2datetime(nav_meas_old.tot.wn,
                                           nav_meas_old.tot.tow)
  t_new = gpstime.gpst_components2datetime(nav_meas_new.tot.wn,
                                           nav_meas_new.tot.tow)
  dt = (t_new - t_old).total_seconds()
  doppler = (nav_meas_new.carrier_phase - nav_meas_old.carrier_phase) / dt
  return doppler


def update_nav_meas(nav_meas_old, nav_meas_new):
  """Update observations based on prior time step. Does two things:

  0. Drop observations based on lock counter changes from the previous
  navigation measurement.
  1. Add doppler frequency.

  Parameters
  ----------
  nav_meas_rover : object

  nav_meas_base : object


  Returns
  ----------

  """
  nav_meas_updated = {}
  for prn, nav_meas in nav_meas_new.iteritems():
    if nav_meas_old.get(prn, None):
      if nav_meas_old[prn].lock_counter == nav_meas_new[prn].lock_counter:
        nav_meas_updated[prn] = nav_meas
        doppler = tdcp_doppler(nav_meas_old[prn], nav_meas)
        nav_meas_updated[prn].doppler = doppler
        nav_meas_updated[prn].raw_doppler = doppler
  return nav_meas_updated


def derive_gpst_meas(timestamp, obs_t, eph_t, nav_meas_tmp):
  """Process a single rover observation at a specific timestep,
  producing sdiffs and

  See: piksi_firmware/src/solution.c:time_matched_obs_thread.

  Parameters
  ----------
  timestamp : object

  rover_obs_t : object

  eph_t : object

  nav_meas_tmp : object


  Returns
  ----------

  """
  nav_meas = obs_table_to_nav_measurement(timestamp, obs_t, eph_t)
  if nav_meas_tmp is None:
    return (nav_meas, None)
  else:
    # Update the observations (i.e., doppler and stuff and PVT).
    nav_meas = update_nav_meas(nav_meas_tmp, nav_meas)
    spp = calc_PVT(nav_meas.values())
    return (nav_meas, spp)


###############################################################################
# Operations on Observation tables


def filter_col(x):
  if x is not None:
    return x.dropna(axis=1, how='all')
  else:
    return None


def fill_observations(table, cutoff=None, verbose=False):
  """Given a table of observations from a HITL test, fills in some
  derived observations using raw base/rover observations and ephemeris
  data. Also reconstructs single point positions using these
  observations. Returns dicts of each of these quantities, keyed by
  GPS timestamp.

  Parameters
  ----------
  table : Pandas table

  Returns
  ----------
  dict
    Derived observations, keyed by the name used for the Panel in the
    HDF5 store: {<panel_name> : {<timestamp>: <value>}}

  """
  # Holds the navigation measurement from the previous timestep
  base_nav_tmp = None
  rover_nav_tmp = None
  sdiffs = {}
  base_spp_sim = {}
  rover_spp_sim = {}
  i = 0
  logging_interval = 1000
  start = time.time()
  # Iterate through each of the timestamped observations from the rover.
  for timestamp, rover_obs_t in table.rover_obs.iteritems():
    i += 1
    if verbose:
      if i % logging_interval == 0:
        print "Processed %d records! @ %s sec." % (i, time.time() - start)
    if cutoff is not None and i >= int(cutoff):
      if verbose:
        print "Exiting at %d records! @ %s sec!" % (i, time.time() - start)
      break
    # Extract obs_base, obs_rover and spp_rover entries for this
    # timestep.  Filter out invalid ephemerides and observations
    eph_t = filter_col(match_ephemeris(timestamp,
                                       table.ephemerides_filled))
    base_obs_t = filter_col(match_obs(timestamp, table.base_obs))
    if base_obs_t is None or eph_t is None:
      continue
    # Transform obs_base and obs_rover entries into
    # navigation_measurements, which we turn into a set of sdiffs.
    base_nav_tmp, base_spp_sim_t = derive_gpst_meas(timestamp, base_obs_t,
                                                    eph_t, base_nav_tmp)
    base_spp_sim_t = calc_PVT(base_nav_tmp.values())
    rover_nav_tmp, rover_spp_sim_t = derive_gpst_meas(timestamp,
                                                      filter_col(rover_obs_t),
                                                      eph_t, rover_nav_tmp)
    rover_spp_sim_t = calc_PVT(rover_nav_tmp.values())
    # Differenced observations
    sdiff_ts = mk_single_diff(rover_nav_tmp, base_nav_tmp)
    if sdiff_ts:
      sdiffs[timestamp] = {t: s.__dict__() for (t, s) in sdiff_ts.iteritems()}
    if base_spp_sim_t:
      base_spp_sim[timestamp] = dict(zip(['x', 'y', 'z'],
                                         base_spp_sim_t.pos_ecef))
    if rover_spp_sim_t:
      rover_spp_sim[timestamp] = dict(zip(['x', 'y', 'z'],
                                          rover_spp_sim_t.pos_ecef))
  return {'rover_sdiffs': pd.Panel(sdiffs),
          'rover_ddiffs': pd.DataFrame({}),
          'base_spp_sim': pd.DataFrame(base_spp_sim),
          'rover_spp_sim': pd.DataFrame(rover_spp_sim)}
