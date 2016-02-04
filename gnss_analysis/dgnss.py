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

"""
A set of utility functions related to differential global navigation
satellite system (DGNSS) algorithms.

In particular this handles computation of single and double differences
between rover and base stations.
"""

import numpy as np

from swiftnav.observation import SingleDiff

from gnss_analysis import propagate


def make_propagated_single_differences(rover, base, base_pos_ecef):
  """
  Takes a pair of observations from a rover and a base station and
  propagates the base observation to the rover observation time,
  the computes the single difference between the two.
  """
  rover, base = rover.align(base, axis=0, join='inner')
  # TODO:  It seems that it may actually be better to propagate to
  #   a common time of transmission, which would allow you to assume
  #   the satellite hadn't moved.  Currently the use of nano second
  #   precision when generating synthetic observations adds too much
  #   error to be able to discern if this is true or not.
  prop_base = propagate.delta_tof_propagate(base_pos_ecef, base,
                                            new_tot=rover['tot'].values)
  return single_difference(rover, prop_base)


def single_difference(rover, base):
  """
  Computes the single difference between a pair of observations,
  one from a rover and one from a base station (though really
  they can be any two observation sets).
  """
  assert rover.index.name == 'sid'
  assert base.index.name == 'sid'
  # take only the satellites the two have in common
  rover, base = rover.align(base, axis=0, join='inner')
  assert 'pseudorange' in rover and 'raw_pseudorange' in rover
  assert 'pseudorange' in base and 'raw_pseudorange' in base
  # this sets which variables will be differenced
  diff_vars = ['pseudorange', 'carrier_phase', 'doppler']
  # actually perform the difference
  sdiffs = base[diff_vars] - rover[diff_vars]
  # handle the signal to noise ratio
  if 'signal_noise_ratio' in rover:
    sdiffs['signal_noise_ratio'] = np.minimum(rover.signal_noise_ratio,
                                              base.signal_noise_ratio)
  # keep track of both the locks simultaneously
  if 'lock' in rover:
    sdiffs['lock'] = (rover['lock'] + base['lock'])
  # return a copy of rover, but with single differnces instead of obs.
  out = rover.copy()
  out.update(sdiffs)
  return out


def double_difference(sdiffs, drop_ref=True):
  """
  Computes the double difference given a set of single differences.
  This is done by picking a reference satellite, then computing the
  difference between it's single difference and all other single
  differences.
  """
  assert sdiffs.index.name == 'sid'
  # subset to only the variables for which differencing makes sense
  ss = sdiffs.reset_index()[['pseudorange', 'carrier_phase', 'sid']]
  # Choose a reference satellite arbitrarily.  We don't need to take
  # the difference between all permutations of satellites since they
  # will largely be linear combinations of each other.
  ref = ss.iloc[0]
  if drop_ref:
    ddiffs = ss.iloc[1:].copy()
  else:
    ddiffs = ss.copy()
  # subtract out the reference satellites single differences
  ddiffs[['pseudorange', 'carrier_phase']] -= ref[['pseudorange', 'carrier_phase']]
  # create a new ref_sid variable so we know which satellite was used.
  ddiffs['ref_sid'] = ref['sid']
  return ddiffs


def create_single_difference_objects(sdiffs):
  """
  Converts pandas DataFrame single difference objects to swiftnav c-type
  SingleDiff objects.
  """
  assert sdiffs.index.name == 'sid'
  for sid, sdiff in sdiffs.iterrows():
    if isinstance(sid, basestring):
      if sid.startswith('G'):
        sid = int(sid[1:])
      else:
        raise NotImplementedError("The sid %s appears to be a GLONAS"
                                  " satellite or other non supported type"
                                  % sid)

    yield SingleDiff(pseudorange=sdiff.pseudorange,
                     carrier_phase=sdiff.carrier_phase,
                     doppler=sdiff.doppler,
                     sat_pos=sdiff[['sat_x', 'sat_y', 'sat_z']].values,
                     sat_vel=sdiff[['sat_v_x', 'sat_v_y', 'sat_v_z']].values,
                     snr=sdiff.signal_noise_ratio,
                     lock_counter=sdiff.lock,
                     sid={'sat':sid, 'band':0, 'constellation': 0})


def omega_dot_unit_vector(base_pos, sat, baseline_estimate):
  """
  The single differences are proportional to omega (which is
  often very nearly 1) times the unit vector pointing from
  base station to the satellite:
  
    (omega e)^T x = lambda (p_s - p_r)
  
  Where e is the unit vector, x is the baseline, omega is
  
    omega = |2 h - x| / (|h| + |h - x|),
    
  and h is the vector pointing from receiver to satellite.
  The vector `omega e` is then:
  
    omega e = (2 * h - x) / (|h| + |h - x|)
  
  Reference:
    Xiao-wen Chang and Christopher C. Paige and Lan Yin
    Code and Carrier Phase Based Short Baseline GPS Positioning: Computational Aspects
    Equation 1
  """
  sat_pos = np.atleast_2d(sat[['sat_x', 'sat_y', 'sat_z']].values)
  # TODO: maybe add sagnac rotation here? Maybe not worth it?
  h = sat_pos - base_pos
  x = baseline_estimate
  denom = (np.linalg.norm(h, axis=1) + np.linalg.norm(h - x, axis=1))
  return (2 * h - baseline_estimate) / denom[:, None]
