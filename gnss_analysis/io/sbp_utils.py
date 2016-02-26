import re
import logging
import numpy as np
import pandas as pd

import sbp.observation as ob

from sbp.utils import exclude_fields

from gnss_analysis import constants as c
from gnss_analysis import time_utils
from gnss_analysis.io import common


def count_rover_observation_messages(input_path):
  with open(input_path, 'r') as f:
    content = f.read()
  # find all message types that are observations and have a non zero
  # sender (which implies it's a rover observation).
  matches = re.findall('\"sender\": [^0]{1,5}.{2} \"msg_type\": 67', content)

  return len(matches)


def normalize(sbp_obs):
  """
  Makes changes to an sbp observation in order to conform with
  industry standard (RINEX) conventions.
  """
  # TODO: Eventually sbp logs will already conform to RINEX
  #   conventions, in which case we can remove this function.
  sbp_obs = sbp_obs.copy()
  sbp_obs['carrier_phase'] = -sbp_obs['carrier_phase']
  return sbp_obs


def get_source(msg):
  """
  Determines if a message originated at the rover or base station.
  """
  # if the source field is 0 the source of observations is
  # from the base, otherwise  the rover.
  return 'rover' if msg.sender else 'base'


def get_sat(msg):
  """
  Infer the satellite id from an sbp message (or observation).
  This looks first for an sid attribute, then a prn attribute.
  """
  # does the msg have an sid attribute?
  if hasattr(msg, 'sid'):
    # the sbp interface changed and in some cases the
    # satellite id is held in msg.sid.sat.
    if hasattr(msg.sid, 'sat'):
      sat = msg.sid.sat
    else:
      sat = msg.sid
  # does the msg have a prn attribute?
  elif hasattr(msg, 'prn'):
    sat = msg.prn
  else:
    raise AttributeError("msg does not contain a satellite id")
  return '%.2d' % (sat + 1)


def update_position(obs, msg, data, suffix, in_mm=False):
  """
  Convert the position estimate to a DataFrame and update
  the correct obs field.
  """
  field = '%s_%s' % (get_source(msg), suffix)
  obs[field] = position_to_dataframe(msg, data, in_mm)
  return obs


def position_to_dataframe(msg, data, in_mm):
  """
  Extracts any position information from the message and
  converts units, then creates a one dimensional DataFrame
  from the result, indexed by the host offset.

  Parameters
  ----------
  msg : nav.MsgPos*
    An sbp position message.
  data :
    The data corresponding to the msg.

  Returns
  -------
  df : pd.DataFrame
    A DataFrame holding the position estimate and any
    meta data.  Units (if applicable) will be in meters
    and seconds.
  """
  m = exclude_fields(msg)
  host_offset = data['delta']
  m.update({'host_time': data['timestamp']})
  # The time of week is in milliseconds, convert to seconds
  m['tow'] /= c.MSEC_TO_SECONDS

  if in_mm:
    # For baseline computations all units are in mm but we want meters
    # here we convert north, east, down coords if they are present.
    to_convert = ['x', 'y', 'z', 'n', 'e', 'd']
    for k in to_convert:
      if k in m:
        m[k] /= c.MM_TO_M

  return pd.DataFrame(m, index=pd.Index([host_offset], name='host_offset'))


def observation_to_dataframe(msg, data):
  """
  Convert an observation message to a dataframe which
  is indexed by the msg source and satellite id.

  Parameters
  ----------
  msg : ob.MsgObs or ob.MsgObsDepA
    An sbp observation message.
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
      # notice that in observations the tow is in msecs
      tow = msg.header.t.tow / c.MSEC_TO_SECONDS
      time = time_utils.tow_to_datetime(wn=msg.header.t.wn,
                                        tow=tow)
      sat = get_sat(obs)
      # Convert pseudorange, carrier phase to SI units.
      v = {'raw_pseudorange': obs.P / c.CM_TO_M,
           'carrier_phase': obs.L.i + obs.L.f / c.Q32_WIDTH,
           'signal_noise_ratio': obs.cn0,
           'lock_count': obs.lock,
           'host_time': data['timestamp'],
           'host_offset': data['delta'],
           'epoch': time,
           # piksi propagates all observations to match the epoch
           # so the epoch and valid time are identical.
           'time': time,
           'sat': sat,
           'band': '1',
           'constellation': 'GPS',
           }
      return v

  if not len(msg.obs):
    return pd.DataFrame()

  # Combine into a dataframe.
  df = pd.DataFrame([extract_observations(o) for o in msg.obs])
  df = common.normalize(df)
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
    # determine the satellite id.
    sat = get_sat(msg)
    msg = exclude_fields(msg)
    # make sure the satellite id attribute isn't propagated since
    # it is part of the index
    [msg.pop(x, None) for x in ['sid', 'prn']]
    # TODO: once iodc and ura gets incorporated into the libswiftnav.ephemeris
    # python bindings, decode these from the message.
    if not 'fit_interval' in msg:
      msg['fit_interval'] = 4
    msg['ura'] = 0
    # notice that we don't pop toe_tow.  It is required for calc_sat_state.
    msg['toe'] = time_utils.tow_to_datetime(msg.pop('toe_wn'),
                                            msg.get('toe_tow'))
    msg['toc'] = time_utils.tow_to_datetime(msg.pop('toc_wn'),
                                            msg.pop('toc_tow'))
    msg['constellation'] = 'GPS'
    eph = pd.DataFrame(msg, index=pd.Index([sat], name='sat'))
    return eph


def update_ephemeris(obs, msg, data):
  """
  Updates the current obs with a new ephemeris message.
  An ephemeris message typically contains the
  ephemeris for a single satellite.  This new ephemeris
  information is added to the previous ephemerides,
  overwriting if necessary.
  """
  updates = ephemeris_to_dataframe(msg, data)
  if not (msg.valid and msg.healthy):
    return obs
  prev_ephs = obs.get('ephemeris', None)
  # if no updates, return the original obs
  if updates is None:
    return obs
  # if there is no previous obs return the updates.
  if prev_ephs is None:
    obs['ephemeris'] = updates
    return obs
  # doing align upfront avoid's doing alignment twice
  updates, prev_ephs = updates.align(prev_ephs)
  # pd.DataFrame.combine_first(updates, obs) would
  # work here as well, but it is extremely slow.  This
  # version is not nearly as robust but is an order
  # of magnitude faster.
  prev_ephs.update(updates)
  obs['ephemeris'] = prev_ephs
  return obs


def update_observation(obs_set, msg, data):
  """
  Updates the current obs with a new set of observations.
  An observation message typically contains information
  from multiple satellites.  When new observations are
  received the previous observations are simply thrown out,
  so this method replaces the old obs with the new obs.
  """
  new_obs = observation_to_dataframe(msg, data)
  if not new_obs.size:
    return obs_set
  # determine if the message was from the rover or base and
  # get the previous observations.
  source = get_source(msg)
  prev_obs = obs_set.get(source, pd.DataFrame())
  if (logging.getLogger().getEffectiveLevel() == logging.DEBUG
      and not np.all(prev_obs.index == new_obs.index)):
    added = set(new_obs.index.values).difference(set(prev_obs.index.values))
    logging.debug("Added satellites, %s" % str(list(added)))
    lost = set(prev_obs.index.values).difference(set(new_obs.index.values))
    logging.debug("Lost satellites %s" % str(list(lost)))
  # align the previous observations to match the new_obs index.
  new_obs, prev_obs = new_obs.align(prev_obs, 'left')
  # update the doppler information.
  lock_changed = (new_obs['lock_count'] != prev_obs['lock_count']).astype('int')
  new_obs['lock'] = lock_changed
  new_obs['raw_doppler'] = common.tdcp_doppler(prev_obs, new_obs)
  # the actual doppler is the raw_doppler + clock_rate_err * GPS_L1_HZ
  # but since we don't know the clock_rate_err yet we leave that for later.
  obs_set[source] = new_obs
  return obs_set


def is_rover_observation(msg):
  """
  Returns true if msg is an observation type and it came
  from the rover.
  """
  return isinstance(msg, (ob.MsgObs, ob.MsgObsDepA)) and msg.sender
