"""
Parses satellite observaiton files that have been reported using:

  RINEX: The Receiver Independent Exchange Format Version 2.11

Reference: ftp://igs.org/pub/data/format/rinex211.txt
"""
import struct
import string
import logging
import datetime
import itertools
import numpy as np
import pandas as pd

from gnss_analysis import time_utils
from gnss_analysis.io import sbp_utils


def split_every(n, iterable):
  """
  Breaks an iterable into chunks of size n.
  """
  i = iter(iterable)
  piece = list(itertools.islice(i, n))
  while len(piece) == n:
    yield piece
    piece = list(itertools.islice(i, n))


def build_parser(format_strings):
  """
  Takes a format string (or list of format strings) and
  returns a function which will parse a line into a tupple.
  
  See struct.Struct
  """
  # if format_strings is already a string we simply use it
  if not isinstance(format_strings, basestring):
    format_strings = ' '.join(format_strings)
  # create a Struct object and return the unpack method.
  return struct.Struct(format_strings).unpack_from


def apply_parser(parser, field_names, funcs, line):
  """  
  This takes a parser that splits a line into len(field_names)
  fields.  It then applies a conversion function to each of
  the corresponding fields and creates a dictionary indexed
  by the field names.
  
  This is equivalent to:
    {k: f(v) for k, v, f in zip(field_names, parser(line), funcs)
  
  with a few sanity checks.
  """
  # split the line into what we expect should be len(field_name) chunks
  values = parser(line)
  assert len(field_names) == len(values)
  assert len(funcs) == len(values)
  # apply the conversion functions (if they aren't None)
  values = [v if f is None else f(v)
            for f, v in zip(funcs, values)]
  # assemble into a dictionary
  return dict(zip(field_names, values))


def parse_line(fields, line):
  """
  A utility function that performs the most common form of line parsing.
  This takes a list of tuples (fields) that defines how a line should
  be parsed, builds the parser and applies it to line.
  
  Parameters:
  -----------
  fields : iterable of tuples
    Each tuple should consist of three elements; the name of the field,
    the corresponding format string, and a conversion function.
    For example:
        ('version', '2x 18s', float)
    would take the next 20 characters, treat the first two as padding
    and pass the remaining 18 through the float function then store
    the result in the output dictionary under the key 'version'.
  line : string
    A single line that should be at least as long as the sum of the
    field format strings.
    
  Returns
  -------
  field_dict : dict
    A dictionary with keys corresponding to the field names and
    values corresponding to the parsed values.
  """
  field_names, format_strings, funcs = zip(*fields)
  parser = build_parser(format_strings)
  return apply_parser(parser, field_names, funcs, line)


def parse_version(x):
  """
  Parsers the version line in a RINEX header returning a
  dictionary with fields 'version', 'file_type' and 'sat_system'
    
 |RINEX VERSION / TYPE| - Format version (2.11)                  | F9.2,11X,  |
 |                    | - File type ('O' for Observation Data)   |   A1,19X,  |
 |                    | - Satellite System: blank or 'G': GPS    |   A1,19X   |
  """
  # This will only ever be done once per file so it isn't really worth
  # pre-building a parser and applying later.
  return parse_line([('version', '9s 11x', string.strip),
                     ('file_type', '1s 19x', string.strip),
                     ('sat_system', '1s 19x', string.strip)],
                    x)


def parse_types_of_observ(x):
  """
  Parses the types of observations field in a header.  The names are converted
  from RINEX single characters to more readable strings (pseudorange etc ...)
  
  |# / TYPES OF OBSERV | - Number of different observation types  |     I6,    |
  |                    |   stored in the file                     |            |
  |                    | - Observation types                      |            |
  |                    |   - Observation code                     | 9(4X,A1,   |
  |                    |   - Frequency code                       |         A1)|
  |                    |   If more than 9 observation types:      |            |
  |                    |     Use continuation line(s) (including  |6X,9(4X,2A1)|
  """
  fmt_str = '6s ' + ' '.join(9 * ['4x 1s 1s'])
  divs = build_parser(fmt_str)(x)
  cnt = int(divs[0])
  if cnt > 9:
    raise NotImplementedError("RINEX supports more than 9 observation types"
                              " but this parser doesn't (yet)")
  renames = {'C': 'raw_pseudorange',
             'P': 'p_code',
             'L': 'carrier_phase',
             'S': 'signal_noise_ratio'}
  # split the observation codes and frequency codes into pairs then take
  # only the first count pairs.
  pairs = split_every(2, divs[1:])
  # These pairs will be tuples of (observation_code, frequency_code)
  pairs = itertools.islice(pairs, 0, cnt)
  pairs = [(renames[x], freq) for x, freq in pairs]
  return {'observation_types': pairs}


def parse_almanac(x):
  """
  Parses almanac coefficients from a navigation message header file.
  
 +--------------------+------------------------------------------+------------+
*|DELTA-UTC: A0,A1,T,W| Almanac parameters to compute time in UTC| 3X,2D19.12,|*
 |                    | (page 18 of subframe 4)                  |     2I9    |
 |                    | A0,A1: terms of polynomial               |            |
 |                    | T    : reference time for UTC data       |      *)    |
 |                    | W    : UTC reference week number.        |            |
 |                    |        Continuous number, not mod(1024)! |            |
 +--------------------+------------------------------------------+------------+
  """
  return parse_line([('a0', '3x 19s', float_or_nan),
                     ('a1', '19s', float_or_nan),
                     ('t', '9s', int),
                     ('w', '9s', int)],
                    x)


def parse_offset_applied(x):
  return parse_line([('receiver_offset_applied', '6s', bool)], x)


def parse_header(f):
  """
  Parses both navigation and observation RINEX headers.
  """
  # negative widths represent ignored padding fields
  header = parse_version(f.next())
  assert header['version'] == '2.11'
  # we only support observation and navigation files
  assert header['file_type'] in ['O', 'N']
  # we only support GPS (or mixed) data at the moment.
  # A blank sat_system implies we're looking at a navigation message.
  assert header['sat_system'] in ['', 'M', 'G']
  # each line in a header is 60 char of content and 20 indicating the type
  content_parser = build_parser('60s 20s')
  # This holds all the possible header content types we processs
  handlers = {'COMMENT': logging.debug,
              'RINEX VERSION / TYPE': parse_version,
              '# / TYPES OF OBSERV': parse_types_of_observ,
              'DELTA-UTC: A0,A1,T,W': parse_almanac,
              'RCV CLOCK OFFS APPL': parse_offset_applied,
              }
  # Here we set default values
  header['receiver_offset_applied'] = False
  # iterate over each of the lines
  for line in f:
    # the last line of the header will end with "END OF HEADER"
    if "END OF HEADER" in line:
      break
    # splits apart the line type and line content
    content, line_type = content_parser(line)
    # remove stray characters around the line type
    line_type = line_type.strip()
    # if we can process this line we pass it to the appropriate handler
    if line_type in handlers:
      # handlers can either return a dictionary of key/values to update
      new_attributes = handlers[line_type](content)
      # the new_attributes might be none (with comments for example)
      if new_attributes is not None:
        # if they aren't None we update the header
        header.update(new_attributes)
  return header


def int_or_zero(x):
  """
  Attempts to convert x to an integer (if it's non-empty)
  otherwise (if x is '') returns 0
  """
  x = x.strip()
  if len(x):
    return int(x)
  else:
    return 0


def float_or_nan(x):
  """
  This will convert a string representation of a float into
  the corresponding float, or into a nan if the string is empty.
  
  This also takes care of converting RINEX conventions for scientific
  notation ('1D-3') to the python notation ('1e-3').
  """
  x = x.strip()
  if len(x):
    if 'D' in x:
      x = x.replace('D', 'e')
    return float(x)
  else:
    return np.nan


def convert_to_datetime(dictlike):
  """
  Converts time that is represented using year, month, day, hour, min, sec
  attributes to a single datetime object.  The result is stored in a new
  attribute ('time') and the others are all removed from the dict.
  
  Parameters
  ----------
  dictlike : dict
    A dictionary like object (must have pop and get/set attr methods) that
    contains a year, month, day, hour, min, sec representation of time.
    All values (except the seconds) should be integers.
    
  Returns
    None (the dict is modified in place)
  """
  # create the list of arguments to create a datetime objects, and remove
  # the corresponding fields from the epoch dictionary
  dt_args = [dictlike.pop(x) for x in ['year', 'month', 'day',
                                       'hour', 'min']]
  # this will fail if any of the args are not integers (which is desirable)
  time = datetime.datetime(*dt_args)
  # now add a float amount of seconds
  time += datetime.timedelta(seconds=dictlike.pop('sec'))
  return time


def parse_epoch(epoch_line):
  """
 +-------------+-------------------------------------------------+------------+
 | EPOCH/SAT   | - Epoch :                                       |            |
 |     or      |   - year (2 digits, padded with 0 if necessary) |  1X,I2.2,  |
 | EVENT FLAG  |   - month,day,hour,min,                         |  4(1X,I2), |
 |             |   - sec                                         |   F11.7,   |
 |             |                                                 |            |
 |             | - Epoch flag 0: OK                              |   2X,I1,   |
 |             | - Number of satellites in current epoch         |     I3,    |
 |             | - List of PRNs (sat.numbers with system         | 12(A1,I2), |
 |             |   identifier, see 5.1) in current epoch         |            |
 |             | - receiver clock offset (seconds, optional)     |   F12.9    |
 +-------------+-------------------------------------------------+------------+
 """
  fields = [('year', '1x 2s', lambda x: int(x) + 2000),
            ('month', '1x 2s', int),
            ('day', '1x 2s', int),
            ('hour', '1x 2s', int),
            ('min', '1x 2s', int),
            ('sec', '11s', float),
            ('epoch_flag', '2x 1s', int),
            ('n_sats', '3s', int),
            ('prns', '36s', lambda x: x),
            ('receiver_clock_offset', '12s', float_or_nan)]
  epoch = parse_line(fields, epoch_line)
  # the prns line
  if epoch['n_sats'] > 12:
    raise NotImplementedError("RINEX supports more than 12 satellites"
                              " but this parser doesn't not.")
  assert epoch['epoch_flag'] == 0
  # we don't know the number of satellites until we parse the line, so we first
  # parse the line, then convert the prn field into a list of satellite prns
  prns = epoch['prns'].strip()
  # the prn string is a list of length 3 identifiers, this splits
  # them apart.
  epoch['prns'] = [''.join(x) for x in split_every(3, prns)]
  # make sure the number of prns found matches that reported.
  assert len(epoch['prns']) == epoch['n_sats']
  # this converts the year, month, day, hour, min, sec into a 'time' attribute
  epoch['time'] = convert_to_datetime(epoch)
  return epoch


def build_observation_parser(header):
  """
  From the header file we know how many observation types exist in
  the file, from that we can decide how many lines each observation
  message will cover, and pre-build a parser function that we can
  use for the entirity of the RINEX file.
  """
  # RINEX can only fit 5 observations per line, here we decide
  # how many lines we'll need to join before parsing.
  n_types = len(header['observation_types'])
  n_lines = n_types / 5 + 1
  obs_parser = build_parser(['14s 1s 1s'] * n_types)


  def parser(lines):
    # grab the next `n_lines` lines and join them into one
    joined = ''.join(lines.next() for i in range(n_lines))
    # then parse it into observations
    obs = obs_parser(joined)
    values, locks, strengths = zip(*split_every(3, obs))
    # each observation should be in floating point, nans for empty values
    values = map(float_or_nan, values)
    # The locks also hold information about anti-spoofing (which adds noise?)
    locks = map(int_or_zero, locks)
    strengths = map(int_or_zero, strengths)
    # Here we assemble the observations into a DataFrame
    # There is probably a faster way to do this.
    value_entries = [(freq, obs_type, val)
                     for (obs_type, freq), val
                     in zip(header['observation_types'], values)]
    lock_entries = [(freq, '%s_LLI' % obs_type, val)
                     for (obs_type, freq), val
                     in zip(header['observation_types'], locks)]
    strength_entries = [(freq, '%s_SN' % obs_type, val)
                        for (obs_type, freq), val
                        in zip(header['observation_types'], locks)]
    entries = list(itertools.chain(value_entries, lock_entries, strength_entries))
    entries = pd.DataFrame(entries, columns=['frequency', 'variable', 'value'])
    return entries.pivot(index='frequency', columns='variable', values='value')

  return parser


def build_navigation_parser(header):
  """
  Builds a parser that is capable of parsing a navigation message that is
  stored according to TABLE A4 in ftp://igs.org/pub/data/format/rinex211.txt

  Parameters
  ----------
  None
  
  Returns
  ----------
  parser : function
    A function that takes an iterable of lines (of length 80) and returns the
    next navigation message.
  """


  # Here we define the parser fields for each of the 8 lines in a nav message
  line_fields = [# Line 1
                 [('sid', '2s', lambda x: 'G%2s' % x.strip().zfill(2)),
                  ('year', '1x 2s', lambda x: int(x) + 2000),
                  ('month', '1x 2s', int),
                  ('day', '1x 2s', int),
                  ('hour', '1x 2s', int),
                  ('min', '1x 2s', int),
                  ('sec', '5s', float_or_nan),
                  ('af0', '19s', float_or_nan),# sat clock bias
                  ('af1', '19s', float_or_nan),# sat clock drift
                  ('af2', '19s', float_or_nan)],# sat clock drift rate
                 # Line 2
                 [('iode', '3x 19s', float_or_nan),
                  ('c_rs', '19s', float_or_nan),
                  ('dn', '19s', float_or_nan),
                  ('m0', '19s', float_or_nan)],
                 # Line 3
                 [('c_uc', '3x 19s', float_or_nan),
                  ('ecc', '19s', float_or_nan),
                  ('c_us', '19s', float_or_nan),
                  ('sqrta', '19s', float_or_nan)],
                 # Line 4
                 [('toe_tow', '3x 19s', float_or_nan),
                  ('c_ic', '19s', float_or_nan),
                  ('omega0', '19s', float_or_nan),
                  ('c_is', '19s', float_or_nan)],
                 # Line 5
                 [('inc', '3x 19s', float_or_nan),
                  ('c_rc', '19s', float_or_nan),
                  ('w', '19s', float_or_nan),
                  ('omegadot', '19s', float_or_nan)],
                 # Line 6
                 [('inc_dot', '3x 19s', float_or_nan),
                  ('L2_codes', '19s', float_or_nan),
                  ('toe_wn', '19s', float_or_nan),
                  ('L2_p_flag', '19s', float_or_nan)],
                 # Line 7
                 [('valid', '3x 19s', float_or_nan),
                  ('healthy', '19s', float_or_nan),
                  ('tgd', '19s', float_or_nan),
                  ('iodc', '19s', float_or_nan)],
                 # Line 8
                 [('tot_tow', '3x 19s', float_or_nan),
                  ('fit_interval', '19s', float_or_nan)],
                 ]

  # pre-build the parsers and zip the fields
  line_fields = [zip(*fields) for fields in line_fields]
  line_parsers = [build_parser(format_strings)
                  for _, format_strings, _ in line_fields]

  def navigation_parser(lines):
    """
    A function which takes an iterable over lines in a RINEX navigation
    message file and returns the next navigation message as a dict.
    """
    def parse_one_line(line_parser, field_defs, line):
      # The Struct objects for parsing are pre-built, this actually
      # uses the parser applies the conversion function and creates a dict
      # with field_names as keys
      field_names, _, funcs = field_defs
      return apply_parser(line_parser, field_names, funcs, line)
    # parse each line in turn
    parsed_lines = [parse_one_line(*x)
                    for x in zip(line_parsers, line_fields, lines)]
    # combine the list of dictionaries into a single dictionary
    # (note that there are faster ways to do this)
    nav_message = {k: v for part in parsed_lines for k, v in part.items()}
    nav_message['toc'] = convert_to_datetime(nav_message)
    nav_message['toe'] = time_utils.tow_to_datetime(wn=nav_message['toe_wn'],
                                                    tow=nav_message['toe_tow'])
    # N indicates navigation measurements for the GPS system, which
    # we assume below when prepending G to the prn
    assert header['file_type'] == 'N'
    # since we aren't using the delta_utc corrections yet, we make sure they
    # are so small that they don't matter (up to nanosecond precision) even
    # if the epherides are more than a day old:
    #   (1e-9 / (24 * 60 * 60) = 1.1574074074074075e-14
    assert header.get('a0', 0.) <= 1e-9
    assert header.get('a1', 0.) <= 1e-14
    return nav_message

  return navigation_parser


def parse_observation_set(lines, observation_parser):
  """
  Uses a pre-built observaiton parser and parses out the
  next set of observations.
  """
  # the first line in a observation set is the epoch
  epoch = parse_epoch(lines.next())

  # then the next lines correspond to actual observations.
  def add_sid(x, prn):
    x['sid'] = prn
    return x
  dfs = [add_sid(observation_parser(lines), k)
         for k in epoch['prns']]
  # concatenate all the observations together
  df = pd.concat(dfs)
  # add a time column
  df.ix[:, 'time'] = epoch['time']
  # switch to using 'sid' as the index
  df.reset_index(inplace=True)
  df.set_index('sid', inplace=True)
  return df


def iter_observations(filelike):
  """
  Takes a file like object that iterates over lines from a RINEX
  observaiton file and generates observations sets.  This is done by
  first reading the header, then iteratively parsing and yielding 
  DataFrames consisting of the observations at each epoch.
  
  Parameters
  ----------
  filelike : file-like
    A file like object whose only real requirement is that it
    is an iterable of lines.
    
  Returns
  --------
  observations : generator
    Produces a generator that iterates over observations sets,
    with one for each epoch.
  """
  lines = ('{: <80}'.format(l) for l in filelike)
  header = parse_header(lines)
  observation_parser = build_observation_parser(header)

  prev_obs = parse_observation_set(lines, observation_parser)
  while True:
    obs = parse_observation_set(lines, observation_parser)
    obs['raw_doppler'] = sbp_utils.tdcp_doppler(prev_obs, obs)
    yield obs


def iter_navigations(filelike):
  """
  Takes a file like object that iterates over lines from a RINEX
  navigation file and generates observations sets.  This is done by
  first reading the header, then iteratively parsing and yielding 
  DataFrames consisting of the ephemerides at each epoch.
  
  ----------
  filelike : file-like
    A file like object whose only real requirement is that it
    is an iterable of lines.
    
  Returns
  --------
  observations : generator
    Produces a generator that iterates over navigation sets
    (aka ephemerides) with one for each epoch.
  """
  lines = ('{: <80}'.format(l) for l in filelike)
  header = parse_header(lines)
  nav_parser = build_navigation_parser(header)

  def iter_by_prn():
    nav_message = nav_parser(lines)
    while len(nav_message):
      yield pd.Series(nav_message)
      nav_message = nav_parser(lines)

  for t, grp in itertools.groupby(iter_by_prn(), key=lambda x: x['toc']):
    yield pd.DataFrame(list(grp)).set_index('sid')


def simulate_from_rinex(rover, navigation, base=None):
  """
  Takes filelike objects for rover, navigation and base files
  and generates states that emulate the states generated
  from piksi logs (see simulate.py).  A state is emitted for each
  rover epoch and contains the most recent ephemeris and base
  observations up to that epoch.
  """
  if base is not None:
    iter_base = iter_observations(open(base, 'r'))
    next_base = iter_base.next()

  iter_nav = iter_navigations(open(navigation, 'r'))
  state = {'ephemeris': iter_nav.next(), }
  next_nav = iter_nav.next()

  for obs in iter_observations(open(rover, 'r')):
    assert state['ephemeris']['toc'][0] <= obs['time'][0]
    if next_nav['toc'][0] <= obs['time'][0]:
      state['ephemeris'].update(next_nav)
      next_nav = iter_nav.next()

    if base is not None:
      while next_base['time'][0] <= obs['time'][0]:
        state['base'] = next_base
        next_base = iter_base.next()

    state['rover'] = obs
    yield state


