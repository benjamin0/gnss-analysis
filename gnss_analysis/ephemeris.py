# Copyright (C) 2015 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
Code to compute ephemeris information in python.  This duplicates some of the
functionality in swiftnav.ephemeris, but avoids the (costly) conversion
to and from c objects.
"""

import logging
import numpy as np
import pandas as pd

from scipy.optimize import newton

import gnss_analysis.constants as c

from gnss_analysis import time_utils


def sagnac_rotation(sat_pos, time_of_flight):
  """
  This function corrects for the sagnac effect caused when a
  satellite with known position at the time of transmission broadcasts
  a signal that then travels to earth.  During that time the
  earth rotates under the satellites so (in an earth centric earth fixed
  reference frame) it will appear the signal had been sent from
  different positions.
  
  Parameters
  ----------
  sat_pos : array-like
    An n by 3 array consisting of the x, y, z coordinates (columns) for
    n satellites (rows).  A single dimensional length 3 array works as
    well.
  time_of_flight : scalar or array-like (np.timedelta64)
    The time of flight of the signal, used to determine how much the
    earth has rotated.
    
  Returns
  -------
  sat_pos_rot : np.ndarray
    An n by 3 array with the apparent positions to a fixed observer
    on earth.
  
  https://en.wikipedia.org/wiki/Sagnac_effect#Reference_frames
  """
  # convert sat_pos to an array
  sat_pos = np.asarray(sat_pos)
  # make sure sat_pos is two dimensional (we assume this below)
  if not sat_pos.ndim == 2:
    sat_pos = sat_pos[None, :]
  # make sure there are three and only three columns (x, y, z)
  assert sat_pos.shape[1] == 3
  assert time_of_flight.dtype.kind == 'm'
  # on the way the earth rotates omega degress, so we adjust the
  # satellites positions.  The resulting position (sat_pos_rot) is the ecef
  # coordinate of where it would have appeared the signal had been
  # sent from if it took delta_t seconds to reach the observer.
  tof_seconds = time_utils.seconds_from_timedelta(time_of_flight)
  omega = c.GPS_OMEGAE_DOT * tof_seconds
  sat_pos_rot = sat_pos.copy()
  cos = np.cos(omega)
  sin = np.sin(omega)
  sat_pos_rot[:, 0] = cos * sat_pos[:, 0] + sin * sat_pos[:, 1]
  sat_pos_rot[:, 1] = -sin * sat_pos[:, 0] + cos * sat_pos[:, 1]
  return sat_pos_rot


def time_of_transmission(eph, time_of_arrival, ref_loc,
                         max_iterations=2, tol=1e-12):
  """
  Computes the time of transmission given a set of ephemeris data, a reference
  location and a time.  This requires an iterative process in which
  an approximate distance is computed from a first satellite position at
  arrival time.  An approximate time of transmission is then computed
  which gives a more accurate satellite position that can be used to
  improve the time of transmission estimate.
  
  Parameters
  ----------
  eph : pd.DataFrame
    A DataFrame holding ephemeris parameters.
  time_of_arrival : datetime64
    Represents the time when a transmission was received at ref_loc.
  ref_loc : array-like
    An array like holding the ECEF coordinates of a reference location
  max_iterations : int (optional)
    The maximum number of iterations to perform, defaults to two.
    Typically this only takes a couple iterations to converge to sub
    mm resolutions.
  tol : float (optional)
    The tolerance used to determine convergence.  If the change in time
    between two iterations is less than tol, convergence is assumed.

  Returns
  -------
  tot : pd.DataFrame
    A data frame holding the week number and time of week that represent
    the time of transmission.
  """
  # start with the time of transmission the same as the time of arrival
  tot = time_of_arrival
  # Note that just one iteration is typically enough to get within mm
  # precision which should be sufficient.
  old_tot = tot
  for i in range(max_iterations):
    # compute the time of flight using the current best guess at
    # time of transmission
    sat_state = calc_sat_state(eph, tot)
    sat_pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values
    dists = np.linalg.norm(sat_pos - ref_loc, axis=1)
    tof = time_utils.timedelta_from_seconds(dists / c.GPS_C)
    # rotate the earth according to the time of flight
    sat_pos = sagnac_rotation(sat_pos, tof)
    dists = np.linalg.norm(sat_pos - ref_loc, axis=1)
    # use the resulting distance to update our guess of the
    # time of transmission.
    correction = time_utils.timedelta_from_seconds(dists / c.GPS_C)
    tot = time_of_arrival - correction
    if np.all(np.abs(time_utils.seconds_from_timedelta(tot - old_tot))) < tol:
      break
    old_tot = tot
  return tot


def time_of_arrival(eph, tot, ref_loc,
                    max_iterations=1, tol=1e-12):
  """
  Computes the time of arrival of a signal sent at some transmission
  time given a set of ephemeris data, a reference
  location.  This requires an iterative process in which
  an approximate distance is computed from a first guess at arrival time.
  The earth is the rotated according to corresponding time of flight
  and the time of arrival is updated.
  
  Parameters
  ----------
  eph : pd.DataFrame
    A DataFrame holding ephemeris parameters.
  tot : datetime64
    Represents the GPS sytem time when a transmission was sent.
  ref_loc : array-like
    An array like holding the ECEF coordinates of a reference location
  max_iterations : int (optional)
    The maximum number of iterations to perform, defaults to two.
    Typically this only takes a couple iterations to converge to sub
    mm resolutions.
  tol : float (optional)
    The tolerance used to determine convergence.  If the change in time
    between two iterations is less than tol, convergence is assumed.

  Returns
  -------
  toa : pd.DataFrame
    A data frame holding the week number and time of week that represent
    the time of arrival.
  """
  old_toa = tot
  # compute the satellite state at time of transmission
  sat_state = calc_sat_state(eph, tot)
  sat_pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values
  dists = np.linalg.norm(sat_pos - ref_loc, axis=1)
  # Note that just one iteration is typically enough to get within mm
  # precision which should be sufficient.  In fact, this should converge
  # significantly faster than time_of_transmission which requires
  # iteratively updating the satellite_state.
  for i in range(max_iterations):
    # current guess at time of flight
    tof = time_utils.timedelta_from_seconds(dists / c.GPS_C)
    # corresponding earth rotation
    los_pos = sagnac_rotation(sat_pos, tof)
    # new time of arrival
    dists = np.linalg.norm(los_pos - ref_loc, axis=1)
    correction = time_utils.timedelta_from_seconds(dists / c.GPS_C)
    toa = tot + correction
    # check for convergence
    if np.all(np.abs(time_utils.seconds_from_timedelta(toa - old_toa))) < tol:
      break
    old_toa = toa
  return toa


def _eccentric_anomaly_from_mean_anomaly(ma, ecc):
  """
  Solve for the eccentric anomaly using newton's method
  to zero solve.

  https://en.wikipedia.org/wiki/Eccentric_anomaly#From_the_mean_anomaly
  """
  def func(ea):
    return ea - ecc * np.sin(ea) - ma

  def gfunc(ea):
    return 1 - ecc * np.cos(ea)

  ret = newton(func, x0=ma, fprime=gfunc)
  return ret


def calc_sat_state(eph, t=None):
  """
  Calculates the satellite state (position, velocity, clock error
  and clock error rate) from set of ephemeris parameters.  This
  was ported from libwsiftnav/ephemeris.c to avoid the costly
  translation to and from swiftnav c objects and to allow for
  vectorized operations.

  Parameters
  ----------
  eph : pd.DataFrame or pd.Series
    A Series or DataFrame that holds ephemeris parameters and optionally
    the time ('time') of the desired satellite state.
  t : datetime64
    The GPS system time for which the satellite state is desired.
    NOTE! system time is different than satellite time.  In otherwords,
    the satellite clock will be reporting t_sat = t_system + clock_err.
    All positions and velocities from this algorithm are valid at t_system.

  Returns
  -------
  out : type(ephemerides)
    A Series or DataFrame holding the satellite position, velocity
    and clock errors.  All columns are prefixed with 'sat_', so
    the resulting data can be merged with corresponding observations.

  See Also:
    - swiftnav.ephemeris.calc_sat_state_
    - libswiftnav/python/swiftnav/ephemeris.pyx
    - IS-GPS-200D, Section 20.3.3.3.3.1 and Table 20-IV
  """
  # TODO: Once all these tests are settled it'd be nice to break this appart
  #       into smaller pieces, one that computes position one for clock errors
  #       etc.
  # TODO: If the c version of calc_sat_state took float values for all the
  #       ephemeris parameters (instead of ephemeris objects) we could implement
  #       a cython vectorized call that would likely be as fast (or faster) than
  #       this but avoid all the duplicate code.

  # NOTE: Notice that belowe we always extract the values rather than working
  # with pandas Series/DataFrames.  Then only at the end we reassemble into a
  # DataFrame.  This significantly speeds up the function.

  if t is None:
    t = eph['time']

  # In order to support the use of pd.Timestamp objects for `t` we do
  # some duck typing here to optionally convert to datetime64
  if hasattr(t, 'to_datetime64'):
    t = t.to_datetime64()

  # time difference between the query time (t) and the ephemeris reference
  dt_toc = time_utils.seconds_from_timedelta(t - eph['toc'])
  # Seconds from clock data reference time (toc)
  clock_err = (eph.af0.values +
               dt_toc * (eph.af1.values + dt_toc * eph.af2.values) -
               eph.tgd.values)
  # this is the derivative of the taylor approximation above.
  clock_rate_err = eph.af1.values + 2.0 * dt_toc * eph.af2.values

  # Seconds from the time from ephemerides reference epoch (toe)
  dt_toe = time_utils.seconds_from_timedelta(t - eph['toe'])

  # If dt is greater than 4 hours our ephemerides isn't valid.
  if np.any(np.abs(dt_toe) > eph['fit_interval'] * 60 * 60):
    logging.warn("Using an ephemerides outside validity period, dt = %+.0f")

  # Calculate position per IS-GPS-200D p 97 Table 20-IV

  # Semi-major axis in meters.
  a = np.square(eph.sqrta.values)
  # Corrected mean motion in radians/sec.
  ma_dot = np.sqrt(c.GPS_GM / (a * a * a)) + eph.dn.values
  # Corrected mean anomaly in radians.
  ma = eph.m0.values + ma_dot * dt_toe

  # Iteratively solve for the Eccentric Anomaly
  # (from Keith Alter and David Johnston)
  ecc = eph.ecc.values
  ea_from_ma = np.vectorize(_eccentric_anomaly_from_mean_anomaly)
  ea = ea_from_ma(ma, ecc)
  ea_dot_denom = 1.0 - ecc * np.cos(ea)
  ea_dot = ma_dot / ea_dot_denom

  # Relativistic correction term.
  einstein = c.GPS_F * ecc * eph.sqrta.values * np.sin(ea)
  clock_err += einstein

  # Begin calc for True Anomaly and Argument of Latitude
  minor_over_major = np.sqrt(1.0 - ecc * ecc)
  # Argument of Latitude = True Anomaly + Argument of Perigee.
  al = np.arctan2(minor_over_major * np.sin(ea),
                  np.cos(ea) - ecc) + eph.w.values
  al_dot = minor_over_major * ea_dot / ea_dot_denom

  # Calculate corrected argument of latitude based on position.
  cal = (al + eph.c_us.values * np.sin(2.0 * al) +
              eph.c_uc.values * np.cos(2.0 * al))
  cal_dot = (al_dot * (1.0 + 2.0 * (eph.c_us.values * np.cos(2.0 * al) -
                                    eph.c_uc.values * np.sin(2.0 * al))))

  # Calculate corrected radius based on argument of latitude.
  r = (a * ea_dot_denom + eph.c_rc.values * np.cos(2.0 * al) +
                          eph.c_rs.values * np.sin(2.0 * al))
  r_dot = (a * ecc * np.sin(ea) * ea_dot
                 + 2.0 * al_dot * (eph.c_rs.values * np.cos(2.0 * al) -
                                   eph.c_rc.values * np.sin(2.0 * al)))

  # Calculate inclination based on argument of latitude.
  inc = (eph.inc.values + eph.inc_dot.values * dt_toe +
         eph.c_ic.values * np.cos(2.0 * al) +
         eph.c_is.values * np.sin(2.0 * al))
  inc_dot = (eph.inc_dot.values +
             2.0 * al_dot * (eph.c_is.values * np.cos(2.0 * al) -
                             eph.c_ic.values * np.sin(2.0 * al)))

  # Calculate position and velocity in orbital plane.
  x = r * np.cos(cal)
  y = r * np.sin(cal)
  x_dot = r_dot * np.cos(cal) - y * cal_dot
  y_dot = r_dot * np.sin(cal) + x * cal_dot

  # Corrected longitude of ascenting node.
  om_dot = eph.omegadot.values - c.GPS_OMEGAE_DOT
  om = (eph.omega0.values + dt_toe * om_dot +
        - c.GPS_OMEGAE_DOT * eph.toe_tow.values)

  # Compute the satellite's position in Earth-Centered Earth-Fixed
  # coordiates.
  out = {'sat_x': x * np.cos(om) - y * np.cos(inc) * np.sin(om),
         'sat_y': x * np.sin(om) + y * np.cos(inc) * np.cos(om),
         'sat_z': y * np.sin(inc)}

  # Compute the satellite's velocity in Earth-Centered Earth-Fixed
  # coordiates.
  temp = y_dot * np.cos(inc) - y * np.sin(inc) * inc_dot
  out['sat_v_x'] = -om_dot * out['sat_y'] + x_dot * np.cos(om) - temp * np.sin(om)
  out['sat_v_y'] = om_dot * out['sat_x'] + x_dot * np.sin(om) + temp * np.cos(om)
  out['sat_v_z'] = y * np.cos(inc) * inc_dot + y_dot * np.sin(inc)

  out['sat_clock_error'] = clock_err
  out['sat_clock_error_rate'] = clock_rate_err
  out['sat_time'] = t

  return pd.DataFrame(out, index=eph.index)


def _update_columns(left, right):
  """
  Performs a join on two DataFrames in which the intersection of the
  indices and the union of the columns are used, giving prefernence
  to columns in 'right'.
  """
  # only use columns in left if the don't exist in right
  cols_to_use = left.columns.difference(right.columns)
  if len(cols_to_use):
    # if we are using any of the columns in left we concatenate across
    # columns
    return pd.concat([left[cols_to_use], right], axis=1, join='inner')
  else:
    # otherwise we only use the left's index.
    return right.ix[left.index]


def has_sat_state(obs):
  """
  A simple convenience function which checks if a satellite state field
  exists in obs.
  """
  return 'sat_x' in obs


def maybe_convert_time(obs):
  if 'time' not in obs:
    obs.ix[:, 'time'] = time_utils.tow_to_datetime(obs[['wn', 'tow']])
    obs.drop(['wn', 'tow'], inplace=True)
  return obs


def add_satellite_state(obs, ephemerides=None, account_for_sat_error=False):
  """
  Compute the satellite state for a set of observations and
  add any corrections to the observation data.  This function
  also serves the goal of inferring variables that were not
  explicitly sent in sbp.MsgObs messages, including:
    - A satellite clock error corrected pseudorange which
      is added as a new column
    - The time of transmission which is inferred from the
      raw pseudorange
    - Computing doppler from raw_doppler

  Parameters
  ----------
  obs : pd.DataFrame
    A DataFrame holding observations (such as raw_pseudorange,
    carrier phase etc ...) which are corrected according to the
    computed satellite states.  If ephemerides is None this
    DataFrame is also expected to contain the ephemeris data
    or precomputed satellite state.
  ephemerides : (optional) pd.DataFrame
    A DataFrame holding either ephemeris parameters, or
    precomputed satellite state (from calc_sat_state) or
    None (in which case such information is expected to
    be held in obs).
    
  Returns
  -------
  obs : pd.DataFrame
    A DataFrame with the same index as obs, but with satellite
    state and corrected observations.
    
  See Also: libswiftnav/track.c::calc_nav_measurements
  """
  assert obs.index.name == 'sid'
  assert 'raw_pseudorange' in obs

  obs = maybe_convert_time(obs)

  if ephemerides is None:
    # if ephemerides weren't supplied make sure obs contains
    # either ephemerides or satellite state
    assert 'af0' in obs or has_sat_state(obs)
  else:
    assert ephemerides.index.name == 'sid'
    # combine the base observations with available ephemeris
    if 'time' in ephemerides and 'time' in obs:
      ephemerides.drop('time', axis=1, inplace=True)
    # join the two dataframes together, adding columns, but
    # prefering columns from ephemerides
    obs = _update_columns(obs, ephemerides)

  # this is the apparent time of flight, not the actual (physical) one
  tof = time_utils.timedelta_from_seconds(obs['raw_pseudorange'] / c.GPS_C)
  # time of transmission is the epoch time minus the time of flight.
  obs['tot'] = obs['time'].values - tof

  # it is sometimes helpful to precompute satellite state all
  # at once, then pass it through the some other set of functions
  # that can't assume the state has been precomputed.
  # Here we check if the state was already added, and if so
  # skip the expensive calc_sat_state computation.
  if not has_sat_state(obs):
    sat_state = calc_sat_state(obs, obs['tot'])
    # Here we optionally adjust the satellite state to account for
    # satellite clock error.  In otherwords, the satellites all transmit
    # at what they think is a synchronized tot, but each is actually off
    # by 'sat_clock_error'.  This error can be on the order of tenths of
    # a second, during which time the satellite position can change
    # significantly.
    if account_for_sat_error:
      sat_error = time_utils.timedelta_from_seconds(sat_state['sat_clock_error'])
      obs['tot'] -= sat_error
      sat_state = calc_sat_state(obs, obs['tot'])
    obs = _update_columns(obs, sat_state)

  # compute the satellite position at the observation time
  # add the clock error to form the corrected pseudorange
  obs['pseudorange'] = obs['raw_pseudorange'] + obs['sat_clock_error'] * c.GPS_C
  obs['doppler'] = obs.raw_doppler + obs['sat_clock_error_rate'] * c.GPS_L1_HZ
  return obs