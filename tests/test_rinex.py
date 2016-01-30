import os
import pytest
import string
import datetime
import numpy as np
import pandas as pd

from gnss_analysis.io import rinex


def test_parse_line():
  """
  Tests several cases using both parse_line and
  the combination of build_parser / apply_parser
  """
  test_fields = [('1', '3x 5s', rinex.float_or_nan),
                 ('2', '1x 5s', string.strip),
                 ('3', '1x 5s', rinex.int_or_zero),
                 ('4', '1x 5s', int)]
  cases = [('   0.000, foo ;    8/    3',
            {'1': 0., '2': 'foo', '3': 8, '4': 3}),
           ('        ,foooo;     /    3',
            {'1': np.nan, '2': 'foooo', '3': 0, '4': 3}),
           ('  11.001                 3',
            {'1': 1.001, '2': '', '3': 0, '4': 3})]

  field_names = [x for x, _, _ in test_fields]
  format_strings = [y for _, y, _ in test_fields]
  funcs = [z for _, _, z in test_fields]

  for line, expected in cases:
    actual = rinex.parse_line(test_fields, line)
    assert actual == expected

    parser = rinex.build_parser(format_strings)
    another = rinex.apply_parser(parser, field_names,
                                 funcs, line)
    assert another == expected


def test_convert_to_datetime():
  d = {'year': 2000,
       'month': 1,
       'day': 1,
       'hour': 0,
       'min': 0,
       'sec': 0}

  d_copy = d.copy()
  actual = rinex.convert_to_datetime(d)
  assert len(d) == 0
  assert actual == datetime.datetime(2000, 1, 1)

  d = d_copy.copy()
  d['sec'] = 1e-9
  d['foo'] = 'bar'
  actual = rinex.convert_to_datetime(d)
  assert 'foo' in d
  assert len(d) == 1
  expected = datetime.datetime(2000, 1, 1) + datetime.timedelta(seconds=1e-9)
  assert actual == expected



def test_split_every():
  # try splitting in pairs
  actual = list(rinex.split_every(2, np.arange(10).repeat(2)))
  expected = list(zip(range(10), range(10)))
  np.testing.assert_array_equal(np.array(actual),
                                np.array(expected))
  # try splitting in 3s
  actual = list(rinex.split_every(3, np.arange(10).repeat(3)))
  expected = list(zip(range(10), range(10), range(10)))
  np.testing.assert_array_equal(np.array(actual),
                                np.array(expected))


def test_build_parser():
  single_parser = rinex.build_parser('1x 5s')
  # make sure the parser knows to discard fields with xs
  actual = single_parser(' hello')
  assert actual == ('hello',)
  # make sure the parser doesn't strip anything
  actual = single_parser('  hi  ')
  assert actual == (' hi  ',)

  # make sure the parser handles multiple format strings
  multi_parser = rinex.build_parser(['1x 5s', '4s 2x'])
  actual = multi_parser(' hellodude  ')
  assert actual == ('hello', 'dude')
  actual = multi_parser('  hi   man  ')
  assert actual == (' hi  ', ' man')


def test_simulate(rinex_observation, rinex_navigation):
  states = rinex.simulate_from_rinex(rover=rinex_observation,
                                     navigation=rinex_navigation)
  # simply iterate over the first 10 states and make sure
  # nothing fails
  for state in [x for _, x in zip(range(10), states)]:
    pass


def test_navigation(rinex_navigation):
  # this will only work on a specific RINEX file.  Make sure we're using it
  assert os.path.basename(rinex_navigation) == 'seat0320.16n'
  f = open(rinex_navigation, 'r')
  # So far this test just iterates through to a specific known navigation
  # parameters and makes sure that value is parsed correctly.
  for i, nav in enumerate(rinex.iter_navigations(f)):
    if nav.ix['G02', 'toe'] == datetime.datetime(2016, 1, 31, 22, 0, 0):
      assert nav.ix['G02', 'af0'] == 5.994844250381e-04
      break
    # This test should pass after four iterations, if not we fail
    if i > 10:
      assert False


def test_observation(rinex_observation):
  # this will only work on a specific RINEX file.  Make sure we're using it
  assert os.path.basename(rinex_observation) == 'seat0320.16o'
  f = open(rinex_observation, 'r')
  # So far this test just iterates through to a specific known observation
  # and makes sure that value is parsed correctly.
  for i, obs in enumerate(rinex.iter_observations(f)):
    if np.any(obs['time'] == datetime.datetime(2016, 2, 1, 12, 0, 4)):
      np.testing.assert_array_equal(obs.ix['G02', 'carrier_phase'],
                                    np.array(126636838.718))
      break
    # This test should pass after four iterations, if not we fail
    if i > 10:
      assert False


@pytest.fixture
def rinex_with_intermixed_header(datadir):
  basename = 'rinex_observation_with_intermixed_header.16o'
  return datadir.join(basename).strpath


def test_parse_with_intermixed_header(rinex_with_intermixed_header):
  # Some RINEX files occasionally throw in header information after
  # epoch lines.  This makes sure we can handle an example file that
  # does that.  We just make sure it makes it through the file and
  # catches the first and last observation.
  obs = list(rinex.iter_observations(rinex_with_intermixed_header))
  assert np.all(obs[0]['time'] == datetime.datetime(2016, 1, 1, 0, 59, 59))
  assert np.all(obs[-1]['time'] == datetime.datetime(2016, 1, 1, 1, 0, 0))
