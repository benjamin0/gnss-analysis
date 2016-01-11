import numpy as np

import ephemeris
import constants as c


def observations_from_toa(ephemerides, location_ecef, toa,
                          *args, **kwdargs):
  """
  A covenience wrapper around observations_from_tot that
  creates synthetic observations from a time of arrival (toa).
  
  See observations_from_tot for details.
  """
  # compute the time of transmission had all the signals arrived
  # simultanousely at time of arrival
  tot = ephemeris.time_of_transmission(ephemerides, toa,
                                       location_ecef)
  # the times will all be different since each satellite is
  # a different distance.  Set the synchronized time of transmission
  # to be the 10th second interval before all the transmission times,
  # this ensures that all signals would have arrived before toa.
  tot = {'wn': toa['wn'],
         'tow': np.round(np.min(tot['tow']), 1)}
  # then return the synthetic observations based off our new tot.
  return observations_from_tot(ephemerides, location_ecef, tot,
                               *args, **kwdargs)


def observations_from_tot(ephemerides, location_ecef, tot,
                          rover_clock_error=0.,
                          account_for_sat_error=True):
  """
  Creates a set of observations (obs time and pseudoranges)
  based off a set of ephemeris information, a known location
  and some time of transmission.
  
  Parameters
  ----------
  ephemerides : pd.DataFrame
    A DataFrame holding ephemeris parameters for some set of
    satellites.
  location_ecef : array-like
    A location given in earth center earth fixed coordinates.
  tot : dict-like
    The time of transmission.  This is assumed to be a dictionary
    like object with fields 'wn' and 'tow' holding a unique
    week number and time of week respectively.
  
  Returns
  -------
  observations : pd.DataFrame
    A DataFrame holding a set of synthetic observations similar
    to what would be returned by simulate.simulate_from_log.
  """
  assert np.unique(tot['wn']).size == 1
  assert np.unique(tot['tow']).size == 1
  # The satellites all transmit at what they think is the same
  # time, but are actually off by 'clock_error' seconds
  sat_state = ephemeris.calc_sat_state(ephemerides, tot)
  obs_tot = ephemerides[['wn', 'tow']].copy()
  obs_tot['wn'] = tot['wn']
  obs_tot['tow'] = tot['tow']
  actual_tot = obs_tot.copy()
  actual_tot['tow'] -= sat_state['sat_clock_error']
  # compute the time when the signal would have arrived at the receiver
  obs_toa = ephemeris.time_of_arrival(ephemerides,
                                      actual_tot,
                                      location_ecef)
  # the actual time of flight is the time between when the
  # signal arrived and when it was actually transmitted
  actual_time_of_flight = obs_toa['tow'] - actual_tot['tow']
  # Set the nav time (the time for which we're trying to solve)
  # to be the next nearest tenth of a second after all the signals
  # arrived at the rover
  nav_time = tot.copy()
  nav_time['tow'] = np.round(obs_toa['tow'].max(), 1) + 0.1

  # In these next few steps we take a bunch of signals that arrived
  # asynchronously and nudge them forward so they look as they would
  # have if they had all arrived exactly at nav_time.  This is similar to
  # (BUT NOT EXACTLY) what happens on the piksi, which uses chip rate
  # to adjust the time of transmission.
  # TODO: make sure this is equivalent.

  # The first order approximation of propagated time of transmission
  # is to simply push transmission times forward by the difference
  # between desired arrival time and actual.
  propagated_tot = obs_tot.copy()
  propagated_tot['tow'] = obs_tot['tow']
  dt = (nav_time['tow'] - obs_toa['tow'])
  propagated_tot['tow'] += dt

  # During that time the satellites would have moved slightly, changing
  # the time of flight (and in turn the time of transmission).  To
  # compensate for this we compute the actual line of sight vectors,
  # then propagate the satellite positions forward in time (by dt)
  # and subtract out the difference between the propagated time of flight
  # and the actual.
  # TODO: Here we are using the satellite state at the *actual* time of
  #   transmission, not the expected GPS system time.
  #   in libswiftnav/track.c:calc_navigation_measurements the
  #   satellite positions are computed at the expected GPS system time.
  #   verify which of these conventions is correct.
  if account_for_sat_error:
    actual_sat_state = ephemeris.calc_sat_state(ephemerides, actual_tot)
  else:
    actual_sat_state = sat_state
  sat_vel = actual_sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']].values
  sat_pos = actual_sat_state[['sat_x', 'sat_y', 'sat_z']].values
  los_sat_pos = ephemeris.sagnac_rotation(sat_pos, actual_time_of_flight)
  # Then account for the rate of change in tot due to satellite velocity
  # ASSUMPTION: earth's rotation doesn't matter much here.  The difference
  # in los_pos would be O(dt * sat_vel / c) which should be extremelly small.
  new_pos = los_sat_pos + dt.values[:, None] * sat_vel
  new_tof = np.linalg.norm(new_pos - location_ecef, axis=1) / c.GPS_C
  # if the new time of flight is longer we need to subract from the
  # transmission time in order to get signals which would have arrived
  # simultaneously.
  propagated_tot['tow'] -= (new_tof - actual_time_of_flight)

  # ASSUMPTION: no noise!
  raw_pseudorange = (nav_time['tow'] - propagated_tot['tow']) * c.GPS_C
  raw_pseudorange += rover_clock_error * c.GPS_C

  wave_length = c.GPS_C / c.GPS_L1_HZ
  carrier_phase = actual_time_of_flight.values * c.GPS_C / wave_length

  obs = ephemerides.copy()
  obs['raw_pseudorange'] = raw_pseudorange
  obs['tow'] = nav_time['tow']
  obs['wn'] = nav_time['wn']
  # carrier_pahse and doppler aren't created yet.
  obs['raw_doppler'] = np.nan
  obs['carrier_phase'] = carrier_phase
  obs['cn0'] = 30.
  obs['lock'] = 0.
  obs['ref_x'] = location_ecef[0]
  obs['ref_y'] = location_ecef[1]
  obs['ref_z'] = location_ecef[2]
  obs['ref_t'] = nav_time['tow']
  obs['ref_rover_clock_error'] = rover_clock_error

  return obs
