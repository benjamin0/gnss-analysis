"""
synthetic.py

This module provides functions for generating synthetic observations
which (for the most part) is used for testing.
"""

import numpy as np

from gnss_analysis import dgnss
from gnss_analysis import ephemeris
from gnss_analysis import time_utils
from gnss_analysis import constants as c
from gnss_analysis.io import common


def observation(ephemerides, location_ecef, toa,
                rover_clock_error=0.,
                include_sat_error=True):
  """
  Creates synthetic observations from a time of arrival (toa).
  This is done by backsolving for the time of transmission that
  at each satellite that would have resulted in simultaneous
  signal arrival at toa (time of arrival).

  The resulting observations conform to the definition of
  observables in the RINEX v2.11 specifications:
  
    https://igscb.jpl.nasa.gov/igscb/data/format/rinex211.txt
  
  Parameters
  ----------
  ephemerides : pd.DataFrame
    A DataFrame holding ephemeris parameters for some set of
    satellites.
  location_ecef : array-like
    A location given in earth center earth fixed coordinates.
  toa : datetime64
    The GPS system time that the observations are valid for.
  rover_clock_error : float (optional)
    The amount of time (in seconds) that the rover clock is
    in advance of GPS system time.
  include_sat_error : boolean (optional)
    When a receiver is computing the raw_pseudorange it does so by
    inferring the time of transmission encoded in the satellite
    signal.  This time of transmission will be relative to the
    satellite clock, which means it is subject to satellte clock
    errors.  Setting this to True indicates that these satellite errors
    should be included in raw_pseudorange computations.  The
    default is True since this aligns with RINEX specifications
    of measurements.

  Returns
  -------
  observations : pd.DataFrame
    A DataFrame holding a set of synthetic observations similar to
    what would be returned by rinex.iter_observations().

  See also: piksi_like_observation
  """
  # compute the time of transmission had all the signals arrived
  # simultanousely at time of arrival.
  tof = ephemeris.time_of_flight_from_toa(ephemerides, toa, location_ecef)
  # Here we compute the time of transmission in system time.  This is
  # the actual time the satellites would have needed to transmit for the
  # signals to arrive at toa.  Note however that this differs from the
  # time of transmission according to the satellite (which has clock bias)
  # and that the raw pseudorange is expected to contain all clock errors.
  tot_sys = toa - time_utils.timedelta_from_seconds(tof)
  if include_sat_error:
    # In order to compute the time of transmission according to the satellite
    # we need to determine the satellite clock bias from ephemeris
    sat_error = ephemeris.calc_sat_state(ephemerides, tot_sys)['sat_clock_error']
  else:
    sat_error = 0.
  # then add error to the time of flight.  Since the time of flight is inferred
  # from the satellite code and since the code will be using the satellite
  # referrence frame, observations will be made with an incorrect time
  # of flight.  Satellite clock error is reported as an advance from system
  # time, and an advance will lead to an apparent decrease in the time of
  # flight.
  tof_sat = tof - sat_error
  # the raw_pseudorange incorporates satellite clock error
  raw_pseudorange = (tof_sat + rover_clock_error) * c.GPS_C
  # but the carrier phase does not.
  # TODO: add ambiguity to the carrier phase.
  carrier_phase = tof * c.GPS_L1_HZ

  # Here we compute the doppler from the satellite velocity dotted
  # with the unit vector between satellite and receiver.
  # This is oriented such that positive doppler corresponds to
  # an approaching satellite.
  sat_state = ephemeris.calc_sat_state(ephemerides, tot_sys)
  unit_vect = dgnss.omega_dot_unit_vector(location_ecef, sat_state,
                                          np.zeros(3))
  sat_velocity = sat_state[['sat_v_x', 'sat_v_y', 'sat_v_z']]
  doppler = -np.sum(sat_velocity * unit_vect, axis=1) / c.GPS_L1_LAMBDA

  # TODO: There is some disagreement between columns names in data returned
  # by the rinex parsers and by sbp log parsers.  We should reconsile those
  # differences and fix them here.
  obs = ephemerides.copy()
  obs.ix[:, 'raw_pseudorange'] = raw_pseudorange.values
  obs['time'] = toa
  obs['carrier_phase'] = carrier_phase
  # doppler, cn0 and lock aren't simulated yet.
  obs['raw_doppler'] = doppler.values
  obs['signal_noise_ratio'] = 30.
  obs['lock'] = 0.
  obs['ref_x'] = location_ecef[0]
  obs['ref_y'] = location_ecef[1]
  obs['ref_z'] = location_ecef[2]
  obs['ref_t'] = toa
  obs['ref_rover_clock_error'] = rover_clock_error
  obs['band'] = 1
  return common.normalize(obs)


def piksi_like_observation(ephemerides, location_ecef, toa,
                           *args, **kwdargs):
  """
  Creates a set of observations (obs time and pseudoranges)
  based off a set of ephemeris information, a known location
  and some time of transmission.  This is done in a way that
  tries to emulate the process on the piksi so that any implicit
  errors introduced by assumptions are represented.
  
  Parameters
  ----------
  ephemerides : pd.DataFrame
    A DataFrame holding ephemeris parameters for some set of
    satellites.
  location_ecef : array-like
    A location given in earth center earth fixed coordinates.
  toa : datetime64
    The GPS system time that the observations are valid for.
  *args, **kwdargs : See synthetic.observation()
    
  Returns
  -------
  observations : pd.DataFrame
    A DataFrame holding a set of synthetic observations similar
    to what would be returned by simulate.simulate_from_log.
  """
  assert toa.dtype.kind == 'M'

  # tof (time of flight in system time) is the time between when the
  # signal arrived and when it was actually transmitted, which is
  # different from when the satellite claims it transmitted.
  tof = ephemeris.time_of_flight_from_toa(ephemerides, toa, location_ecef)

  # Here we figure out a common time of transmission that would have led to
  # all of the signals arriving before toa
  # TODO: The time of transmission (relative to satellite clocks) may not
  # actually be the same for all transmissions?  Particularly with some
  # of the rather large clock errors?  In which case we may need to
  # change the logic below.
  clock_error = ephemeris.calc_sat_state(ephemerides, toa)['sat_clock_error']
  # we add a small buffer so that all the observations end up with
  # some propagation (this is intended to be something like hardware delay).
  time_buffer = 1e-4
  offset = tof + time_buffer - clock_error
  # this is the time of transmission relative to each satellite's clock.
  tot_sat = toa - time_utils.timedelta_from_seconds(offset)
  # pick the earliest time of transmission.
  tot_sat = np.min(tot_sat)
  # tot_sys is the actual system time at the time of transmission
  tot_sys = tot_sat - time_utils.timedelta_from_seconds(clock_error)
  # Compute the time when the signal would have arrived at the receiver
  # In order to emulate the propagation that happens on the piksi,
  # we solve for the toa_sys, or when the signals would have
  # actually arrived at the reciever.  The measurements will later
  # be propagated to a common time of arrival (toa).
  toa_sys = tot_sys + time_utils.timedelta_from_seconds(tof)

  # In these next few steps we take a bunch of signals that arrived
  # asynchronously and nudge them forward so they look as they would
  # have if they had all arrived exactly at toa.  This is similar to
  # (BUT NOT EXACTLY) what happens on the piksi, which uses chip rate
  # to adjust the time of transmission.
  # TODO: make sure this is equivalent.

  # dt is the change in time of arrival.
  # In track.c the satellite coded time of transmission is shifted forward
  # in time by dt (the difference between actual time of arrival and the
  # desired epoch).  This assumes the time of flight doesn't change.
  # This method will include any resulting error.
  dt = time_utils.seconds_from_timedelta(toa - toa_sys)

  # Compute the observations as they would have arrived at toa_sys
  # we will subsequently push the observations forward to a common
  # time of arrival.
  obs = observation(ephemerides, location_ecef, toa_sys, *args, **kwdargs)
  obs['time'] = toa
  # TODO: this is only an approximation of what happens in track.c
  obs['carrier_phase'] -= dt * obs['raw_doppler']
  obs['raw_pseudorange'] -= dt * obs['raw_doppler'] * c.GPS_L1_LAMBDA
  return obs


def synthetic_observation_set(ephemerides, rover_ecef, base_ecef, toa):
  """
  Creates a set of observations for both a rover and a base
  station and stuffs it in a observation set dictionary similar to the
  ones produced by simulate.
  """
  obs_set = {'rover': observation(ephemerides,
                                  location_ecef=rover_ecef,
                                  toa=toa),
             'base': observation(ephemerides,
                                 location_ecef=base_ecef,
                                 toa=toa),
             'ephemeris': ephemerides,
             'epoch': toa}
  return obs_set
