# Copyright (C) 2016 Swift Navigation Inc.
# Contact: Alex Kleeman <alex@swift-nav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
propagate.py

Contains a few (simple) algorithms used to take a set of observations
made by some receiver and use them to infer what the same receiver would
have observed at some future time.  This process is referred to as
propagation.
"""

import numpy as np

from gnss_analysis import ephemeris
from gnss_analysis import constants as c
from gnss_analysis import time_utils


def line_of_sight_distance(sat, position_ecef, tof):
  """
  A convenience function which applies sagnac correction, then
  computes the range (in meters) from position_ecef 
  to each satellite in `sat`.
  """
  los_pos = ephemeris.sagnac_rotation(sat[['sat_x', 'sat_y', 'sat_z']],
                                      time_of_flight=tof)
  return np.linalg.norm(los_pos - position_ecef, axis=1)


def resolve_toa_tot(sat, new_toa, new_tot, position_ecef):
  if new_tot is None and new_toa is None:
    raise ValueError("Expected either new_toa or new_tot")
  if new_tot is not None and new_toa is not None:
    raise ValueError("Expected either new_toa or new_tot (but got both)")

  if new_tot is None and new_toa is not None:
    # if new_toa was provided, we back solve for the new time of transmit
    new_tot = ephemeris.time_of_transmission(sat, new_toa, position_ecef)
  elif new_toa is None and new_tot is not None:
    # if new_tot was provicded we forward solve for the new time of arrival
    new_toa = ephemeris.time_of_arrival(sat, new_tot, position_ecef)
  else:
    raise ValueError("Expected either new_toa or new_tot but not both")

  return new_toa, new_tot


def doppler_propagate(position_ecef, satellite,
                      new_toa=None, new_tot=None,
                      clock_error_rate=0.):
  """
  This propagation algorithm uses the doppler, or time derivative of the
  carrier phase, (in units of cycles/sec) to compute a first order
  approximation of the change in range to each satellite.
  
  L' = L + dt * doppler
  P' = P + dt * doppler * wavelength
  
  """
  # copy so we don't overwrite
  sat = satellite.copy()
  # determine the new time of arrival and transmission.
  new_toa, new_tot = resolve_toa_tot(sat, new_toa, new_tot, position_ecef)
  # How
  delta_t = time_utils.seconds_from_timedelta(new_toa - sat['time'])
  # the change in carrier phase is simply the time difference times doppler
  sat['carrier_phase'] += delta_t * sat['doppler']
  # the change in range is the same as change in carrier_phase times wavelength
  delta_dist = delta_t * sat['doppler'] * c.GPS_L1_LAMBDA
  # the pseudorange gets pushed forward by delta distance
  sat['raw_pseudorange'] += delta_dist
  sat['pseudorange'] += delta_dist
  sat['time'] = new_toa
  sat['tot'] = new_tot
  return sat


def delta_tof_propagate(position_ecef, satellite, new_toa=None, new_tot=None):
  """
  Takes a single satellite observation taken from a known position and
  propagates it to the target time of arrival or time of transmission
  by looking at the difference in time of flight.

  Parameters
  ----------
  position_ecef : tuple of floats
    The position of the receiver in earth centric coordinates.
    The tuple is expected to contain x, y, z values in meters.
  satellite : pd.DataFrame
    A set of satellite observations taken at some time.
  new_toa : datetime64
    The desired time of arrival
  new_tot : datetime64
    The desired time of transmission.
    
  Returns
  -------
  satellite : pd.Series
    The satellite observations if they had been made at gpst.
  """
  # copy so we don't overwrite
  sat = satellite.copy()
  # determine the new time of arrival and transmission.
  new_toa, new_tot = resolve_toa_tot(sat, new_toa, new_tot, position_ecef)
  # don't trust the TOT in sat since it may or may not include clock errors.
  old_tot = ephemeris.time_of_transmission(sat, sat['time'], position_ecef)
  old_tof = time_utils.seconds_from_timedelta(sat['time'] - old_tot)

  # Compute the new time of flight.  Since we are possibly subtract two
  # very small times, we first convert them to float seconds rather
  # than dealing with timedelta objects which cap out at nano seconds.
  new_tof = time_utils.seconds_from_timedelta(new_toa - new_tot)

  # compute the change in physical distance between target and observation
  delta_dist = (new_tof - old_tof) * c.GPS_C

  # compute the new sat positions at time of transmit.
  new_sat_state = ephemeris.calc_sat_state(sat, new_tot)
  sat.update(new_sat_state)
  # update the observations.
  sat['carrier_phase'] += delta_dist / c.GPS_L1_LAMBDA
  # the pseudorange gets pushed forward by delta distance
  sat['raw_pseudorange'] += delta_dist
  sat['pseudorange'] += delta_dist
  sat['time'] = new_toa
  sat['tot'] = new_tot

  return sat
