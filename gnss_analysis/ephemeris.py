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
  time_of_flight : scalar or array-like
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
  # on the way the earth rotates omega degress, so we adjust the
  # satellites positions.  The resulting position (sat_pos_rot) is the ecef
  # coordinate of where it would have appeared the signal had been
  # sent from if it took delta_t seconds to reach the observer.
  omega = c.GPS_OMEGAE_DOT * time_of_flight
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
  time_of_arrival : dict of pd.DataFrame
    Should contain week number ('wn') and time of week ('tow') fields
    and represents the time when a transmission was received at ref_loc.
  ref_loc : array-like
    An array like holding
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
  sat_state : pd.DataFrame
    Returns the satellite state at the time of transmission.
  """
  tot = eph[['wn', 'tow']].copy()
  tot['wn'] = time_of_arrival['wn']
  tot['tow'] = time_of_arrival['tow']
  # Note that just one iteration is typically enough to get within mm
  # precision which should be sufficient.
  old_tow = np.nan
  for i in range(max_iterations):
    sat_state = calc_sat_state(eph, tot)
    sat_pos = sat_state[['sat_x', 'sat_y', 'sat_z']].values
    dists = np.linalg.norm(sat_pos - ref_loc, axis=1)
    tot['tow'] = time_of_arrival['tow'] - dists / c.GPS_C
    if np.allclose(tot['tow'], old_tow, atol=tol):
      break
    old_tow = tot['tow']

  return tot, sat_state


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


def gpsdifftime(end_wn, end_tow, start_wn, start_tow):
  """
  Returns the time difference in seconds between to times
  stored in week number and time of week representations.
  """
  # TODO: should we make a gpstime module ?
  return end_tow - start_tow + (end_wn - start_wn) * c.WEEK_SECS


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
    the time of week (tow) and week number (wn) of the desired
    satellite state.
  t : dict-like
    An object that contains items t['tow'] and t['wn'] which indicate
    the time of the desired satellite state.

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

  if t is None and 'wn' in eph and 'tow' in eph:
    t = eph[['wn', 'tow']]

  dt = gpsdifftime(t['wn'], t['tow'],
                   eph['toc_wn'].values,
                   eph['toc_tow'].values)
  # Seconds from clock data reference time (toc)
  clock_err = (eph.af0.values +
               dt * (eph.af1.values + dt * eph.af2.values) -
               eph.tgd.values)
  clock_rate_err = eph.af1.values + 2.0 * dt * eph.af2.values

  # Seconds from the time from ephemerides reference epoch (toe)
  dt = gpsdifftime(t['wn'], t['tow'],
                   eph['toe_wn'].values, eph['toe_tow'].values)

  # If dt is greater than 4 hours our ephemerides isn't valid.

  if np.any(np.abs(dt) > eph['fit_interval'] * 60 * 60):
    logging.warn("Using an ephemerides outside validity period, dt = %+.0f")

  # Calculate position per IS-GPS-200D p 97 Table 20-IV

  # Semi-major axis in meters.
  a = np.square(eph.sqrta.values)
  # Corrected mean motion in radians/sec.
  ma_dot = np.sqrt(c.GPS_GM / (a * a * a)) + eph.dn.values
  # Corrected mean anomaly in radians.
  ma = eph.m0.values + ma_dot * dt

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
  inc = (eph.inc.values + eph.inc_dot.values * dt +
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
  om = (eph.omega0.values + dt * om_dot +
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
  out['sat_wn'] = t['wn']
  out['sat_tow'] = t['tow']

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


def add_satellite_state(obs, ephemerides=None):
  """
  Compute the satellite state for a set of observations and
  add any corrections to the observation data.  This
  includes:
    - A satellite clock error corrected pseudorange which
      is added as a new column
    - The time of transmission which is inferred from the
      raw pseudorange

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

  if ephemerides is None:
    # if ephemerides weren't supplied make sure obs contains
    # either ephemerides or satellite state
    assert 'af0' in obs or has_sat_state(obs)
  else:
    assert ephemerides.index.name == 'sid'
    # combine the base observations with available ephemeris
    if 'tow' in ephemerides and 'tow' in obs:
      ephemerides = ephemerides.drop(['wn', 'tow'], axis=1)
    # join the two dataframes together, adding columns, but
    # prefering columns from ephemerides
    obs = _update_columns(obs, ephemerides)

  # ephemerides can be either the ephemeris parameters or
  # a pre-computed satellite state.  Here we check if the
  # state has been pre-computed and if not add it to obs.
  if not has_sat_state(obs):
    sat_state = calc_sat_state(obs, obs[['wn', 'tow']])
    obs = _update_columns(obs, sat_state)

  # infer the time of transmission from the raw_pseudorange.
  obs['tot'] = obs['tow'] - obs.raw_pseudorange / c.GPS_C
  # TODO: tot might end up on a different week number.
  assert np.all(obs['tot'] > 0.)
  # compute the satellite position at the observation time
  # add the clock error to form the corrected pseudorange
  obs['pseudorange'] = obs.raw_pseudorange + obs.sat_clock_error * c.GPS_C
  return obs
