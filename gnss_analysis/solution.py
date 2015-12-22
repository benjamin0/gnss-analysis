import logging
import numpy as np
import pandas as pd

from swiftnav import gpstime
from swiftnav.pvt import calc_PVT_ as calc_PVT
from swiftnav.track import NavigationMeasurement
from swiftnav.signal import GNSSSignal

import gnss_analysis.constants as c

from gnss_analysis import ephemeris


def _create_navigation_measurement(obs):
  """
  A private method which computes a single libsswiftnav navigation
  measurement object used to compute single point solutions
  from libswiftnav.
  """
  assert 'sat_x' in obs
  assert 'pseudorange' in obs
  assert 'raw_pseudorange' in obs
  assert isinstance(obs, pd.Series)

  lock_time = np.nan
  # NOTE: this is using the time of transmission NOT the gpstime
  tot = gpstime.GpsTime(wn=obs.wn, tow=obs.tot)
  sid = GNSSSignal(sat=obs.name, band=0, constellation=0)
  # stuff all our known observations into a NavigationMeasurement object.
  nm = NavigationMeasurement(raw_pseudorange=obs.raw_pseudorange,
                             pseudorange=obs.pseudorange,
                             carrier_phase=obs.carrier_phase,
                             raw_doppler=obs.get('doppler', np.nan),
                             doppler=obs.get('doppler', np.nan),
                             sat_pos=obs[['sat_x', 'sat_y', 'sat_z']],
                             sat_vel=obs[['sat_v_x', 'sat_v_y', 'sat_v_z']],
                             snr=obs.cn0,
                             lock_time=lock_time,
                             tot=tot,
                             sid=sid,
                             lock_counter=obs.lock)
  return nm


def create_navigation_measurements(satellites, sat_states=None):
  """
  Creates a list of navigations measurements from a set of satellite
  observations.  The observations are expected to be DataFrames with
  index of satellite id (sid) and columns holding ephemeris parameters
  and actual pseudorange / carrier phase observations.

  Parameters
  ----------
  satellites : pd.DataFrame
    A DataFrame with rows corresponding to satellites and columns to
    observed parameters.  The columns should consist of both ephemeris
    and signal observations.
  sat_states : pd.DataFrame
    A DataFrame containing the location of satellites computed from
    ephemeris information, if None this is computed from the
    satellite data.

  Returns
  -------
  nav_measurements : list(NavigationMeasurement)
    A list of NavigationMeasurement c objects that contain the information
    required by libswiftnav to make position estimates.
  """
  assert satellites.index.name == 'sid'
  # Handle the case where there appears to be a single satellite.
  if isinstance(satellites, pd.Series):
    return _create_navigation_measurement(satellites)
  # create a nav meas for each satellite observation
  nav_measurements = [_create_navigation_measurement(sat)
                      for (_, sat) in satellites.iterrows()]
  return nav_measurements


def libswiftnav_calc_PVT(observations):
  """
  Takes a set of satellite observations and returns the single
  point position computed using swiftnav's calc_PVT function.
  """
  # make sure we have enough observations to estimate a position.
  if (observations is None or
      observations.empty or
      observations.shape[0] < 4):
    logging.debug("Require at least 3 observations to get a fix.")
    return None
  assert 'sat_x' in observations
  assert 'pseudorange' in observations
  assert 'raw_pseudorange' in observations
  # observations should always be indexed by sid
  assert observations.index.name == 'sid'
  # make sure all the observations are from a single time
  assert np.unique(observations['tow'].values).size == 1
  # make sure there aren't any duplicate satellite observations
  assert np.unique(observations.index).size == observations.shape[0]
  # create the navigation measurement objects
  nav_measurements = create_navigation_measurements(observations)
  # and run the observations through calc_PVT
  flag, spp, dops = calc_PVT(filter(None, nav_measurements))
  if flag < 0:
    logging.warn("Single Point Position failed with flag %d" % flag)
  spp = {'pos_llh': spp.pos_llh,
         'pos_ecef': spp.pos_ecef,
         'time': spp.time,
         'clock_offset': spp.clock_offset}
  # TODO: why are the lon, lat in radians?
  spp['pos_llh'] = (spp['pos_llh'][0] * 180. / np.pi,
                    spp['pos_llh'][1] * 180. / np.pi,
                    spp['pos_llh'][2])
  return spp


def can_compute_position(obs):
  """
  Convenience function that returns true if the state contains
  enough information to compute a position.
  """
  # make sure we have enough rover observations
  if obs.empty or obs.shape[0] < 4:
    logging.info("Waiting for 4 satellites, only %d found so far."
                 % obs.shape[0])
    return False
  return True


def single_point_position(obs, max_iterations=15, tol=1e-4):
  """
  Computes the single point position by iteratively solving linearized
  least squares solutions
  
  Parameters
  ----------
  obs : pd.DataFrame
      A DataFrame that holds `pseudorange` observations (corrected for satellite
    clock errors), satellite position (`sat_x`, `sat_y`, `sat_z`) all in ECEF
    coordinates at the time of transmission and `tow` variable which corresponds
    to the time of arrival.
      All observations are assumed to have been propagated to a common time of
    arrival.
  max_iterations : int (optional)
      The maximum number of iterations, defaults to 15.
  tol : float (optional)
      Tolerance for convergence (same is used for both position and time),
      defaults to 1e-4.

  Returns
  --------
  spp : dict
    A dictionary holding the single point position.  Keys included:
      pos_ecef : a length three array representing the position in ECEF coords.
      time : the gps system time at the solution
      clock_offset : the receiver clock error
      converged : a boolean indicating convergence.
    
  Reference:
    Kaplan, E.. Understanding GPS - Principles and applications. 2nd edition
    Section 2.4.2
  """
  # extract the satellite position from the observations
  sat_pos = obs[['sat_x', 'sat_y', 'sat_z']].values
  # we assume that all observations were made (or were propagated to)
  # a common time of arrival.
  assert np.unique(obs['tow'].values).size == 1
  toa = obs['tow'].values[0]

  # we are solving for the ECEF position, cur_x[:3], and clock error cur_x[3].
  cur_x = np.zeros(4)

  converged = False
  for i in range(max_iterations):
    # Range converted into an approximate time of flight in secs.
    dist = np.linalg.norm(sat_pos - cur_x[:3], axis=1)
    tof = dist / c.GPS_C
    # Rotate the satellites relative to the ECEF to compensate for
    # earth's rotation.  We call this the line of sight position.
    los_pos = ephemeris.sagnac_rotation(sat_pos, time_of_flight=tof)
    cur_range = np.linalg.norm(los_pos - cur_x[:3], axis=1) + cur_x[3] * c.GPS_C
    # Create the design matrix for an update to the position and clock error.
    # The general idea is that the (non-linear) theoretical ranges are linearized
    # around the current estimate.
    #
    # linearize the theoretical pseudorange around our current estiamte of x
    #   p(x) ~ p(cur_x) + sum_i{dx_i * dp/dx_i(cur_x)}
    # reformulate into matrix form, where p_obs - p_cur is the observed minus
    # the predicted ranges and A is the jacobian of the predicted range with
    # respect to the position and times.
    #   (p_obs - p_cur) = A * delta_x
    # delta x is the update to our current estimates and is found by performing
    # the least squares solve:
    #   min_{delta_x} = |A * delta_x - (p_obs - p_cur)|_2
    A = (cur_x[:3] - sat_pos) / cur_range[:, None]
    A = np.hstack([A, c.GPS_C * np.ones((A.shape[0], 1))])
    delta_x = np.linalg.lstsq(A, obs.pseudorange - cur_range)
    cur_x += delta_x[0]
    converged = np.linalg.norm(delta_x[0]) < tol
    if converged:
      break

  if not converged:
    cur_x.fill(np.nan)
    logging.warn("single_point_position failed to converge")

  pos_ecef = cur_x[:3]
  time = toa + cur_x[3]

  return {'pos_ecef': pos_ecef,
          'time': {'wn': obs['wn'].values[0], 'tow': time},
          'clock_offset': cur_x[3],
          'converged': converged}


def solution(states, dgnss_filter=None):
  """
  Mimics the solution_thread in piksi_firmware/solution.c, though the
  order is somewhat reversed since on the piksi the solve is made,
  then (modified) observations are sent.  So this back first infers
  missing variables, then solves.
  """

  for state in states:
    # This is similar to creating naviation measurement objects,
    # here the actual units satellite clock error is
    # taken into account to convert raw_pseudorange to pseudorange.
    state['rover'] = ephemeris.add_satellite_state(state['rover'],
                                                   state['ephemeris'])

    # make sure we have enough satellites to compute a position
    if not can_compute_position(state['rover']):
      continue

    # The next step in piksi_firmware computes the doppler, but that has
    # already been done in the simulate module.

    # compute the single point position
    rover_pos = single_point_position(state['rover'])

    # TODO: WIP, plug in DGNSS filters here.
    # if a filter is present and we have enough base observations
    # to compute a position from them.
    if (dgnss_filter is not None and
        can_compute_position(state['base'])):
      state['base'] = ephemeris.add_satellite_state(state['base'],
                                                    state['ephemeris'])
      # Update the filter with the new state
      dgnss_filter.update(state)
      # TODO: if low-latency make propagated sdiffs
      # NOTE: at this point on the piksi baseline messages are output.
      bl = dgnss_filter.get_baseline(state)
    # NOTE: only now are observations sent from the piksi

    yield rover_pos

