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

Contains a set of utility functions that run through piksi log
files and simulate the chip state which can be used to perform
analysis and iterate on the algorithm.
"""

import copy
import time
import logging
import datetime
import numpy as np
import pandas as pd

import sbp.observation as ob

from swiftnav import gpstime
from sbp.utils import exclude_fields

import gnss_analysis.constants as c


def get_source(msg):
  """
  Determines if a message originated at the rover or base station.
  """
  # if the source field is 0 the source of observations is
  # from the base, otherwise  the rover.
  return 'rover' if msg.sender else 'base'


def get_sid(msg):
  """
  Infer the satellite id from an sbp message (or observation).
  This looks first for an sid attribute, then a prn attribute.
  """
  # does the msg have an sid attribute?
  if hasattr(msg, 'sid'):
    # the sbp interface changed and in some cases the
    # satellite id is held in msg.sid.sat.
    if hasattr(msg.sid, 'sat'):
      return msg.sid.sat
    else:
      return msg.sid
  # does the msg have a prn attribute?
  elif hasattr(msg, 'prn'):
    return msg.prn
  else:
    raise AttributeError("msg does not contain a satellite id")


def observation_to_dataframe(msg, data):
  """
  Convert an observation message to a dataframe which
  is indexed by the msg source and satellite id.

  Parameters
  ----------
  msg : ob.MsgObs or ob.MsgObsDepA
    An sbp ephemeris message.
  data :
    The data corresponding to the msg.

  Returns
  -------
  df : pd.DataFrame
    A DataFrame holding all the observations from various
    satellites.  The DataFrame is indexed by satellite id
    and holds information such as the gps time and pseudorange
    carrier phase observations.
  """
  # make sure the message is an observation message
  assert isinstance(msg, (ob.MsgObs, ob.MsgObsDepA))

  def extract_observations(obs):
      # Convert pseudorange, carrier phase to SI units.
      v = {'P': obs.P / c.CM_TO_M,
           'L': obs.L.i + obs.L.f / c.Q32_WIDTH,
           'cn0': obs.cn0,
           'lock': obs.lock,
           'host_time': data['timestamp'],
           'host_offset': data['delta'],
           'wn': msg.header.t.wn,
           # notice that in observations the tow is in msecs
           'tow': msg.header.t.tow / c.MSEC_TO_SECONDS,
           }
      return v

  # Combine into a dataframe.
  df = pd.DataFrame([extract_observations(o) for o in msg.obs],
                    index=pd.Index([get_sid(o) for o in msg.obs],
                                   name='sid'))
  return df


def ephemeris_to_dataframe(msg, data):
  """
  Converts an ephermesis log message to a pandas Dataframe
  that is indexed by the source and satellite id.

  Parameters
  ----------
  msg : ob.MsgEphemeris, ob.MsgEphemerisDepA ob.MsgEphemerisDepB
    An sbp ephemeris message.
  data :
    The data corresponding to the msg.

  Returns
  -------
  eph : pd.DataFrame
    A DataFrame indexed by the satellite id and holds the
    ephemeris parameters for each satellite.
  """
  assert isinstance(msg, (ob.MsgEphemeris,
                          ob.MsgEphemerisDepA,
                          ob.MsgEphemerisDepB))
  # if the message is healthy and valid we emit the corresponding DataFrame
  if msg.healthy == 1 and msg.valid == 1:
    # determine if the ephemeris was from the rover or base
    source = get_source(msg)
    # determine the satellite id.
    sid = get_sid(msg)
    msg = exclude_fields(msg)
    # make sure the satellite id attribute isn't propagated since
    # it is part of the index
    [msg.pop(x, None) for x in ['sid', 'prn']]
    eph = pd.DataFrame(msg, index=pd.Index([sid], name='sid'))
    return eph


def tdcp_doppler(old, new):
  """
  Compute the doppler by using the time difference of the
  carrier phase (TDCP).

  Parameters:
  -----------
  old : pd.DataFrame
    A DataFrame holding a previous set of carrier phase
    observations along with corresponding tow
  new : pd.DataFrame
    A DataFrame holding new carrier phase observations
    along with corresponding tow

  Returns:
  --------
  doppler : pd.Series
    A Series holding the doppler estimates.

  See also: libswiftnav/src/track.c:tdcp_doppler
  """
  # TODO: make sure the week numbers are the same since
  # (at least for now) we don't take them into account

  # delta time in seconds
  dt = new['tow'] - old['tow']
  # make sure dt is positive.
  assert not np.any(dt.values <= 0.)
  # compute the rate of change of the carrier phase
  doppler = (new.L - old.L) / dt
  # mark any computations with differing locks as nan
  invalid = new['lock'] != old['lock']
  doppler[invalid] = np.nan
  return doppler


def update_ephemeris(state, msg, data):
  """
  Updates the current state with a new ephemeris message.
  An ephemeris message typically contains the
  ephemeris for a single satellite.  This new ephemeris
  information is added to the previous ephemerides,
  overwriting if necessary.
  """
  updates = ephemeris_to_dataframe(msg, data)
  prev_ephs = state['ephemeris']
  # if no updates, return the original state
  if updates is None:
    return state
  # if there is no previous state return the updates.
  if prev_ephs.empty:
    state['ephemeris'] = updates
    return state
  # doing align upfront avoid's doing alignment twice
  updates, prev_ephs = updates.align(prev_ephs)
  # pd.DataFrame.combine_first(updates, state) would
  # work here as well, but it is extremely slow.  This
  # version is not nearly as robust but is an order
  # of magnitude faster.
  updates[updates.isnull()] = prev_ephs
  state['ephemeris'] = updates
  return state


def update_observation(state, msg, data):
  """
  Updates the current state with a new set of observations.
  An observation message typically contains information
  from multiple satellites.  When new observations are
  received the previous observations are simply thrown out,
  so this method replaces the old obs with the new obs.
  """
  new_obs = observation_to_dataframe(msg, data)
  # determine if the message was from the rover or base and
  # get the previous observations.
  source = get_source(msg)
  prev_obs = state[source]
  # align the previous observations to match the new_obs index.
  obs, prev_obs = new_obs.align(prev_obs, 'left')
  # update the doppler information.
  new_obs['doppler'] = tdcp_doppler(prev_obs, new_obs)
  return state


def is_rover_observation(msg):
  """
  Returns true if msg is an observation type and it came
  from the rover.
  """
  return isinstance(msg, (ob.MsgObs, ob.MsgObsDepA)) and msg.sender


def simulate_from_log(log, initial_state=None):
  """
  Iterates through a log file and emits a dictionary
  of DataFrames holding the current state of observations
  for each rover observation.

  Parameters
  ----------
  log : sbp.base_logger.BaseLogger
    An sbp log iterator that should yield msg, data pairs.
  initial_state : (optional) dict
    An optional dictionary containing the state at the
    beginning of logging.

  Returns
  -------
  states : generator
    A generator which yields current estimates of state.
    Each state is keyed by source ('rover', 'base', 'ephemeris')
    and then indexed by satellite id.  For example,
    state['rover'][16] would contain a DataFrame holding
    the rover stations observations of satellite 16.

    A state is only emitted for every rover observation.
  """
  _processors = {ob.MsgObs: update_observation,
                 ob.MsgObsDepA: update_observation,
                 ob.MsgEphemeris: update_ephemeris,
                 ob.MsgEphemerisDepA: update_ephemeris,
                 ob.MsgEphemerisDepB: update_ephemeris}

  # use either the initial state provided, or a blank set
  # of observations.
  state = initial_state or {'rover': pd.DataFrame(),
                            'base': pd.DataFrame(),
                            'ephemeris': pd.DataFrame()}

  # what's up with iterating over log.next()?
  for msg, data in log.next():
    if type(msg) in _processors:
      # TODO: process other messages
      state = _processors[type(msg)](state, msg, data)
      # TODO: take care of integrity checks
      # TODO: remove out-dated information?
      if is_rover_observation(msg) and state is not None:
        # yield a copy so we don't accidentally modify things.
        yield copy.deepcopy(state)
    else:
      logging.debug("No processor available for message type %s"
                    % type(msg))
