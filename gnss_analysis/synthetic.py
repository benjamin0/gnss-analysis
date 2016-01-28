import numpy as np

from gnss_analysis import ephemeris
from gnss_analysis import time_utils
from gnss_analysis import constants as c
from gnss_analysis.io import sbp_utils


def observations_from_toa(ephemerides, location_ecef, toa,
                          rover_clock_error=0.,
                          nav_time=None,
                          include_sat_error=True):
  """
  Creates synthetic observations from a time of arrival (toa).
  This is done by backsolving for the time of transmission that
  at each satellite that would have resulted in simultaneous
  signal arrival at toa (time of arrival).
  
  This is a lot simpler than observations_from_tot, but is
  less closesly aligned with the way observations are created
  on the piksi.
  
  See also: observations_from_tot
  """
  # this method isn't capable of removing satellite error yet.
  assert include_sat_error
  # compute the time of transmission had all the signals arrived
  # simultanousely at time of arrival
  tot = ephemeris.time_of_transmission(ephemerides, toa,
                                       location_ecef)
  # ASSUMPTION: no noise!
  rover_clock_error = 0.
  actual_time_of_flight = time_utils.seconds_from_timedelta(toa - tot)
  # TODO,
  raw_pseudorange = (actual_time_of_flight + rover_clock_error) * c.GPS_C
  carrier_phase = actual_time_of_flight * c.GPS_L1_HZ

#   sat_state = ephemeris.calc_sat_state(ephemerides, tot)
#   sat_velocity = sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']]
#
#   los_pos = ephemeris.sagnac_rotation(sat_state[['sat_x', 'sat_y', 'sat_z']],
#                                       toa - tot)
#   los_vect = los_pos - location_ecef
#   unit_vect = los_vect / np.linalg.norm(los_vect, axis=1)[:, None]
#   doppler = np.dot(sat_velocity.values.T, unit_vect) / c.GPS_L1_LAMBDA

  obs = ephemerides.copy()
  obs['raw_pseudorange'] = raw_pseudorange
  obs['time'] = toa
  # Carrier phase is currently set negative to emulate what happens on the
  # piksi, but is later corrected by sbp_utils.normalize()
  obs['carrier_phase'] = -carrier_phase
  # doppler, cn0 and lock aren't simulated yet.
  obs['raw_doppler'] = np.nan
  obs['cn0'] = 30.
  obs['lock'] = 0.
  obs['ref_x'] = location_ecef[0]
  obs['ref_y'] = location_ecef[1]
  obs['ref_z'] = location_ecef[2]
  obs['ref_t'] = toa
  obs['ref_rover_clock_error'] = rover_clock_error

  # note that we don't need to account for satellite clock error since
  # it was not built into the observations.
  obs = ephemeris.add_satellite_state(obs, ephemerides,
                                      account_for_sat_error=False)
  # then return the synthetic observations based off our new tot.
  return sbp_utils.normalize(obs)


def observations_from_tot(ephemerides, location_ecef, tot,
                          rover_clock_error=0.,
                          toa=None,
                          include_sat_error=True):
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
  tot : datetime64
    The time of transmission in GPS time.
  rover_clock_error : float (optional)
    The amount of time (in seconds) that the rover clock is
    in advance of GPS system time.
  toa : ddatetime64
    The GPS system time that the observations are valid for.
  include_sat_error : boolean
    All satellites will transmit at a regular interval according to the
    satellite clock (which is imperfect).  This means they will all
    think they are transmitting simultaneously, but will actually
    be asynchronous.  If this flag is True the satellite positions
    will be taken to be their actual locations when the signal was
    transmitted.  If False, the satellite positions are where the
    satellite thinks it is when the signal is transmitted.
    In otherwords setting include_sat_error to False is the equivalent
    of assuming the satellite clocks are perfect.
    
  Returns
  -------
  observations : pd.DataFrame
    A DataFrame holding a set of synthetic observations similar
    to what would be returned by simulate.simulate_from_log.
  """
  assert tot.dtype.kind == 'M'
  # The satellites all transmit at what they think is the same
  # time, but are actually off by 'clock_error' seconds.  Here
  # we create tot_sat, or the time of transmission according to
  # the satellites
  sat_state = ephemeris.calc_sat_state(ephemerides, tot)
  tot_sat = tot
  tot_sys = tot_sat - time_utils.timedelta_from_seconds(sat_state['sat_clock_error'])
  # compute the time when the signal would have arrived at the receiver
  toa_sys = ephemeris.time_of_arrival(ephemerides,
                                      tot_sys,
                                      location_ecef)
  # the actual time of flight is the time between when the
  # signal arrived and when it was actually transmitted
  time_of_flight_sys = toa_sys - tot_sys
  if toa is None:
    # Set the time of arrival (the time for which we're trying to solve)
    # to be the next nearest tenth of a second after all the signals
    # arrived at the rover
    assert tot.dtype == '<M8[ns]'
    max_as_tenths = np.ceil(np.max(tot).astype('int64') * 1e-8)
    toa = (max_as_tenths * 1e8).astype('datetime64[ns]')

  # In these next few steps we take a bunch of signals that arrived
  # asynchronously and nudge them forward so they look as they would
  # have if they had all arrived exactly at toa.  This is similar to
  # (BUT NOT EXACTLY) what happens on the piksi, which uses chip rate
  # to adjust the time of transmission.
  # TODO: make sure this is equivalent.

  # The first order approximation of propagated time of transmission
  # is to simply push transmission times forward by the difference
  # between desired arrival time and actual.
  dt = (toa - toa_sys)
  propagated_tot = tot_sat + dt

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
  if include_sat_error:
    actual_sat_state = ephemeris.calc_sat_state(ephemerides, tot_sys)
  else:
    actual_sat_state = sat_state
  sat_vel = actual_sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']].values
  sat_pos = actual_sat_state[['sat_x', 'sat_y', 'sat_z']].values
  los_sat_pos = ephemeris.sagnac_rotation(sat_pos, time_of_flight_sys)
  # Then account for the rate of change in tot due to satellite velocity
  # ASSUMPTION: earth's rotation doesn't matter much here.  The difference
  # in los_pos would be O(dt * sat_vel / c) which should be extremelly small.
  dt_seconds = time_utils.seconds_from_timedelta(dt)
  new_pos = los_sat_pos + dt_seconds[:, None] * sat_vel
  new_tof = np.linalg.norm(new_pos - location_ecef, axis=1) / c.GPS_C
  # if the new time of flight is longer we need to subract from the
  # transmission time in order to get signals which would have arrived
  # simultaneously.
  tof_sys_sec = time_utils.seconds_from_timedelta(time_of_flight_sys)
  propagated_tot -= time_utils.timedelta_from_seconds(new_tof - tof_sys_sec)

  # ASSUMPTION: we've assumed there is no noise!
  obs_tof_sec = time_utils.seconds_from_timedelta(toa - propagated_tot)
  raw_pseudorange = obs_tof_sec * c.GPS_C
  raw_pseudorange += rover_clock_error * c.GPS_C

  wave_length = c.GPS_L1_LAMBDA
  tof_sec = time_utils.seconds_from_timedelta(time_of_flight_sys)
  carrier_phase = tof_sec * c.GPS_C / wave_length

  obs = ephemerides.copy()
  obs['raw_pseudorange'] = raw_pseudorange
  obs['time'] = toa
  # Carrier phase is currently set negative to emulate what happens on the
  # piksi, but is later corrected by sbp_utils.normalize()
  obs['carrier_phase'] = -carrier_phase
  # doppler, lock and cn0 aren't actually created yet.
  obs['raw_doppler'] = np.nan
  obs['cn0'] = 30.
  obs['lock'] = 0.
  obs['ref_x'] = location_ecef[0]
  obs['ref_y'] = location_ecef[1]
  obs['ref_z'] = location_ecef[2]
  obs['ref_t'] = toa
  obs['ref_rover_clock_error'] = rover_clock_error
  obs['ref_tot'] = propagated_tot

  return sbp_utils.normalize(obs)


def synthetic_state(ephemerides, rover_ecef, base_ecef, toa):
  state = {'rover': observations_from_toa(ephemerides,
                                          location_ecef=rover_ecef,
                                          toa=toa),
           'base': observations_from_toa(ephemerides,
                                         location_ecef=base_ecef,
                                         toa=toa),
           'ephemeris': ephemerides}
  return state
