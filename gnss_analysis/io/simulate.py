# Copyright (C) 2015 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
simulate.py

Functions for simulating what a receiver would be observing from
observations files (such as RINEX and sbp logs).

These simulate functions were built with the goal of having a standard
way of developing filters or performing analysis in a way that is
indpendent of the input format.  More specifically, each simulate
function is responsible for taking some input format and producing
an iterable of what we call "observation sets".  The observation sets
produced from  RINEX file should be (nearly) identical to those
produced from piksi log files.

An observation set is generated for each epoch.  These self describing
observation_sets typically consist of rover observations (assumed
to be valid for the given epoch), ephemerides (also valid for, but
not necessarily received on, the given epoch), base observations 
(valid on or before the given epoch) and additional information required
to describe the observation set.

Observation Set : An observation set is intended to represent a collection
  of all the information available to a receiver at a given epoch.  This
  information is contained in a dictionary with the following fields:
  
  'rover': A pd.DataFrame that consists of all the rover observations
      valid on the epoch.  This field is always contained in an
      observation set.
  'rover_info': A dictionary of attributes associated with the
      rover observations.  These attributes may include things like
      a flag indicating whether the observations have been adjusted for
      receiver clock error.
  'base': A pd.DataFrame that consists of the most recent base observations.
      These observations are often (but not neccesarily) valid for the
      epoch.  If the base observations were delayed during transmission
      (for example) they may be valid for an epoch previous to the
      current one.
  'base_info': A dictionary of attributes associated with the
      rover observations.  These attributes may include things like
      a flag indicating whether the observations have been adjusted for
      receiver clock error.
  'ephemeris' : A pd.DataFrame that contains all the ephemeris/navigation
      parameters that describe satellites available in rover and
      base observations.  The set of satellites in the ephemeris do
      no neccesarily contain all, nor are they constrained to, the set of
      satellites observed by the rover or base.
  'ephemeris_info': A dictionary of attributes associated with the navigation
      (aka ephemeris) data.  This may include things like the offset
      relative to UTC etc.
  'epoch' : A np.datetime64[ns] representation of the epoch the observation
      set is valid for.
  'attributes': A dictionary containing any additional information
      pertinent to the understanding of the observations in the set.
      Typically this should not contain any information that will change
      between epochs.
      
Observation : A pd.DataFrame that contains all measurements made by
  a reciever.  The data frame rows are indexed by satellite id (sid).
  Measurements from the same satellite but different frequencys are
  treated as two seperate measurements, so it is possible to have
  multiple rows corresponding to a single satellite, in which case
  the frequency column should distinguish between the two.  The columns
  will typically (but not always) include:
    'carrier_phase', 'raw_pseudorange', 'time', 'signal_strength',
    'lock', 'pseudorange', 'tot'.
"""

import copy
import logging
import numpy as np
import pandas as pd
from functools import partial

import sbp.navigation as nav
import sbp.observation as ob

from gnss_analysis import log_utils
from gnss_analysis.io import sbp_utils, rinex


def get_unique_value(x):
  """
  Convenience function that tries to reduce x to a single
  unique value.
  """
  uniq = np.unique(x)
  assert uniq.size == 1
  return uniq[0]


def simulate_from_log(log, initial_observation_set={}):
  """
  Iterates through a log file and emits a dictionary
  of DataFrames each of which holds a set of observations
  for a single epoch.

  Parameters
  ----------
  log : sbp.base_logger.LogIterator
    An sbp log iterator that should yield msg, data pairs.
    NOTE!!! JSONLogIterator incorrectly iterates, so you
    actually need to pass in JSONLogIterator().next() to
    this method.
  initial_observation_set : (optional) dict
    An optional dictionary containing the observations at the
    epoch immediately before logging.  This allows the user
    to bootstrap simulation or set default header information.

  Returns
  -------
  obs_sets : generator
    A generator which yields observations for each epoch.
    Each obs_set dict is keyed by source ('rover', 'base', 'ephemeris')
    and then indexed by satellite id.  For example,
    obs_sets.next()['rover'].ix[16] would contain a Series holding
    the rover stations observations of satellite 16 at the first
    reported epoch.

    A set of observations is emitted for each observed rover epoch.
  """
  _processors = {ob.MsgObs: sbp_utils.update_observation,
                 ob.MsgObsDepA: sbp_utils.update_observation,
                 ob.MsgEphemeris: sbp_utils.update_ephemeris,
                 ob.MsgEphemerisDepA: sbp_utils.update_ephemeris,
                 ob.MsgEphemerisDepB: sbp_utils.update_ephemeris,
                 nav.MsgPosECEF: partial(sbp_utils.update_position,
                                         suffix='spp_ecef'),
                 nav.MsgPosLLH: partial(sbp_utils.update_position,
                                        suffix='spp_llh'),
                 nav.MsgBaselineNED: partial(sbp_utils.update_position,
                                             suffix='rtk_ned'),
                 nav.MsgBaselineECEF: partial(sbp_utils.update_position,
                                              suffix='rtk_ecef'),
                 nav.MsgGPSTime: sbp_utils.update_gps_time}

  default_obs_set = {'rover': pd.DataFrame(),
                     'base': pd.DataFrame(),
                     'ephemeris': pd.DataFrame(),
                     'rover_info': {'original_format': 'sbp',
                                    'reciever_offset_applied': True},
                     'base_info': {'original_format': 'sbp',
                                   'reciever_offset_applied': True},
                     'ephemeris_info': {'original_format': 'sbp', }, }
  # use either the initial observation set provided, or a blank set
  # of observations.
  obs_set = default_obs_set
  obs_set.update(initial_observation_set)

  for msg, data in log_utils.complete_messages_only(log):
    if type(msg) in _processors:
      obs_set = _processors[type(msg)](obs_set, msg, data)
      if sbp_utils.is_rover_observation(msg) and not obs_set['rover'].empty:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug("%d, %d" % (msg.header.t.wn, msg.header.t.tow))
        if 'ephemeris' in obs_set:
          # yield a copy so we don't accidentally modify things.
          obs_set_copy = copy.deepcopy(obs_set)
          obs_set_copy['epoch'] = get_unique_value(obs_set['rover']['time'])
          obs_set_copy['ephemeris']['time'] = obs_set_copy['epoch']
          yield obs_set_copy
    else:
      logging.debug("No processor available for message type %s"
                    % type(msg))


def simulate_from_rinex(rover, navigation, base=None):
  """
  Takes filelike objects for rover, navigation and base files
  and generates observations for each epoch in the rover files
  which contains the most recent ephemeris and base
  observations up to that epoch.
  """

  # Here we initialize with the first navigation message.
  nav_header, iter_nav = rinex.read_navigation_file(navigation)
  obs_set = {'ephemeris': iter_nav.next(),
             'ephemeris_info': nav_header}
  next_nav = iter_nav.next()

  # Read the rover file
  rover_header, iter_rover = rinex.read_observation_file(rover)
  obs_set['rover_info'] = rover_header

  # The base observation file is optional.
  if base is not None:
    base_header, iter_base = rinex.read_observation_file(base)
    next_base = iter_base.next()
    obs_set['base_info'] = base_header

  # The goal is to produce a generator that yields an obs
  # for each observed rover epoch.  Here we iterate over
  # rover epoch's and search for the corresponding base and
  # ephemeris messages.
  for rover_obs in iter_rover:
    obs_set['epoch'] = get_unique_value(rover_obs['time'])
    # We assume that the ephemeris should never be
    # newer than the rover observation.
    assert obs_set['ephemeris']['toc'][0] <= rover_obs['time'][0]
    # If the next navigation message comes before
    # the current rover observation we update the obs_set
    while np.any(next_nav['toc'] <= obs_set['epoch']):
      obs_set['ephemeris'].update(next_nav)
      next_nav = iter_nav.next()
    # If we are using a base station we search for the
    # most recent base observation that came at or before the
    # rover epoch.
    if base is not None:
      while get_unique_value(next_base['time']) <= obs_set['epoch']:
        obs_set['base'] = next_base
        next_base = iter_base.next()
    # update the rover obs
    obs_set['rover'] = rover_obs
    # and return a copy of the dict (though not a deep copy since
    # that causes uneccesary overhead.

    yield copy.deepcopy(obs_set)
