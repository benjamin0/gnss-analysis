import numpy as np
import pandas as pd

from gnss_analysis import constants as c


def timedelta_from_seconds(seconds):
  seconds = np.array(seconds)
  return (seconds * 1e9).astype('timedelta64[ns]')


def seconds_from_timedelta(td):
  # make sure td is a np.timdelta64 object
  assert td.dtype.kind == 'm'
  if not td.dtype == '<m8[ns]':
    td = td.astype('timedelta64[ns]')
  seconds = td.astype('int64') * 1.e-9
  # This may seem out of order, but we first create
  # the seconds variable so it preserves the type of td,
  # then we convert td to a numpy array for the NaT
  # comparison.
  #
  # We have to do the conversion because datetime handling in
  # pandas/numpy is totally whack, and very inconsistent.
  # For example:
  #
  #   > tmp
  #   sid
  #   0   NaT
  #   2   NaT
  #   Name: time, dtype: timedelta64[ns]
  #
  #   > tmp == np.timedelta64('NaT', 'ns')
  #   sid
  #   0    False
  #   2    False
  #   Name: time, dtype: bool
  #
  #   > tmp.values == np.timedelta64('NaT', 'ns')
  #   array([ True,  True], dtype=bool)
  #   tmp.values == np.timedelta64('NaT')
  #   False
  td = np.asarray(td)
  is_nat = td == np.timedelta64('NaT', 'ns')
  if td.size == 1 and is_nat:
    seconds = np.nan
  elif np.any(is_nat):
    seconds[is_nat] = np.nan
  return seconds


def gpst_to_utc(gpst, delta_utc):
  """
  Converts from a GPS time in 
  Parameters
  -----------
  gpst : dict-like (or datetime64)
    A dictionary like that has attributes 'wn' and 'tow' which
    correspond to week number and time of week (in seconds) from
    GPST week zero.
  delta_utc : float
    delta_utc is a required argument that defines the difference
    between UTC and GPST for the times being converted.  This includes
    things such as leap second deviations and fractional estimation
    errors.  Both are typically reported in ephemeris (navigation)
    messages.  Note that if converting an array of times, the
    delta may be different for each.  Take care you pass in the
    correct values as validity cannot be checked.  
    
  Returns
  -------
  utc : np.datetime64
    Returns a np.datetime64 object (or an array of them) that holds
    the UTC representation of the corresponding gpst.
    
  """
  delta_utc = np.asarray(delta_utc)
  # check if we need to convert from wn/tow to datetime
  if 'wn' in gpst and 'tow' in gpst:
    gpst = tow_to_datetime(gpst)
  # now subtract out the delta_utc value (which is given in float seconds)
  return gpst - (delta_utc * 1e9).astype('timedelta64[ns]')


def tow_to_datetime(wn, tow):
  """
  Converts a time using week number and time of week representation
  into a python datetime object.  Note that this does NOT convert
  into UTC.  The resulting datetime object is still in GPS time.
  
  Parameters
  -----------
  wn : int
    An integer corresponding to the week number of a time.
  tow : float
    A float corresponding to the time of week (in seconds) from
    the beginning of the week number (wn).

  Returns
  -------
  utc : np.datetime64
    Returns a np.datetime64 object (or an array of them) that holds
    the UTC representation of the corresponding gpst.
    
  See also: gpst_to_utc, datetime_to_tow
  """
  seconds = np.array(c.WEEK_SECS * wn + tow)
  # np.timedelta64 doesn't accept int64 values as input, WTF?!?
  # instead we multiply by (float) nanoseconds then convert
  # to timedelta using astype.
  dt = c.GPS_WEEK_0 + (seconds * 1.e9).astype('timedelta64[ns]')
  return dt


def datetime_to_tow(dt):
  """
  Converts from a datetime to week number and time of week format.
  NOTE: This does NOT convert between utc and gps time.  The result
  will still be in gps time (so will be off by some number of
  leap seconds).
  
  Parameters
  ----------
  dt : np.datetime64, pd.Timestamp, datetime.datetime
    A datetime object (possibly an array) that is convertable to
    datetime64 objects using pd.to_datetime (see the pandas docs
    for more details).
    
  Returns
  --------
  wn_tow : dict
    Dictionary with attributes 'wn' and 'tow' corresponding to the
    week number and time of week.
    
  See also: tow_to_datetime
  """
  dt = pd.to_datetime(dt)
  if hasattr(dt, 'to_datetime64'):
    dt = dt.to_datetime64()
  elif hasattr(dt, 'values'):
    dt = dt.values
    assert dt.dtype == '<M8[ns]'
  else:
    raise NotImplementedError("Expected either a Timestamp or datetime64 array")
  seconds = (dt - c.GPS_WEEK_0)
  # to actually convert to float seconds we divide by the timedelta64 units
  assert seconds.dtype == '<m8[ns]'
  seconds = seconds.astype('int64') / 1.e9
  return {'wn': (seconds / c.WEEK_SECS).astype('int64'),
          'tow': np.mod(seconds, c.WEEK_SECS)}


def utc_to_gpst(utc, delta_utc):
  """
  Converts from times in utc to the corresponding gps time in
  week number, time of week format.
  
  Parameters
  ----------
  utc : np.datetime64, pd.Timestamp, datetime.datetime
    A datetime object (possibly an array) that is convertable to
    datetime64 objects using pd.to_datetime (see the pandas docs
    for more details).  Take care the times are actually UTC.
  delta_utc : float (optional)
    delta_utc is optional (default 0.) and that defines the difference
    between UTC and GPST for the times being converted.  This includes
    things such as leap second deviations and fractional estimation
    errors.  Both are typically reported in ephemeris (navigation)
    messages.  Note that if converting an array of times, the
    delta may be different for each.  Take care you pass in the
    correct values as validity cannot be checked.
  
  Returns
  -------
  gpst : dict
    A dictionary with attributes 'wn' and 'tow' holding the
    week number and time of week that correspond to the input utc times.
    
  See also: datetime_to_tow, gpst_to_utc
  """
  utc = pd.to_datetime(utc)
  delta_utc = np.array(delta_utc)
  return datetime_to_tow(utc - (delta_utc * 1e9).astype('timedelta64[ns]'))


def diff_time(end, start):
  """
  Returns the time difference in seconds between two times
  stored in week number and time of week representations.
  
    time difference (seconds) = end - start
  
  Parameters
  ----------
  end : dict-like
    A dictionary like object which must have keys 'wn' and 'tow'
    corresponding to the week number and time of week.
  start : dict-like
    A dictionary like object which must have keys 'wn' and 'tow'
    corresponding to the week number and time of week.
  
  Returns
  -------
  time_difference : float
    The number of seconds between end and start.
  """
  assert 'wn' in end
  assert 'tow' in end
  assert 'wn' in start
  assert 'tow' in start
  return end['tow'] - start['tow'] + (end['wn'] - start['wn']) * c.WEEK_SECS


def get_unique_time(obs):
  """
  Extracts the unique time of week (tow) and week number (wn) pairs
  from a set of observations.  The resulting time should correspond
  to the timeof arrival of the variables in `obs`.
  
  Parameters
  ----------
  obs : pd.DataFrame
    A DataFrame which must contain variables 'tow' and 'wn'.
  
  Returns
  -------
  time : dict
    A dictionary with keys 'wn' and 'tow' holding scalar values
    which correspond to the unique values in obs.
  """
  # get the time of arrival for the base observations
  tow = np.unique(obs['tow'].values)
  # make sure the observations are all from a single time
  assert tow.size == 1
  wn = np.unique(obs.wn.values)
  assert wn.size == 1
  return {'wn': wn.item(), 'tow': tow.item()}
