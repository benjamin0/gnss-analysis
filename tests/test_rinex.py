import os
import pytest
import string
import datetime
import operator
import numpy as np

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


def test_navigation(rinex_navigation):
  # this will only work on a specific RINEX file.  Make sure we're using it
  assert os.path.basename(rinex_navigation) == 'seat0320.16n'
  f = open(rinex_navigation, 'r')
  # So far this test just iterates through to a specific known navigation
  # parameters and makes sure that value is parsed correctly.
  header, iter_nav = rinex.read_navigation_file(f)

  assert header['a0'] == 4.656612873077e-09
  assert header['a1'] == 5.329070518201e-15

  for i, nav in enumerate(iter_nav):
    if nav.ix['02', 'toe'] == datetime.datetime(2016, 1, 31, 22, 0, 0):
      assert nav.ix['02', 'af0'] == 5.994844250381e-04
      break
    # This test should pass after four iterations, if not we fail
    if i > 10:
      assert False


def test_navigation_when_out_of_order(datadir):
  nav_file = datadir.join('rinex_211_examples/cebr049x45.16n').strpath
  header, iter_nav = rinex.read_navigation_file(nav_file)

  def get_unique_epoch(x):
    uniq = np.unique(x['epoch'].values)
    assert uniq.size == 1
    return uniq[0]

  navs = list(iter_nav)
  # make sure the nav sets are increasing chronologically (despite
  # being out of order in the file).
  epochs = [get_unique_epoch(x) for x in navs]
  assert np.all(np.diff(epochs) > 0)

  assert epochs[-1] == np.datetime64('2016-02-18T16:00:00.000000000-0800')


def test_observation(rinex_observation):
  # this will only work on a specific RINEX file.  Make sure we're using it
  assert os.path.basename(rinex_observation) == 'seat0320.16o'
  f = open(rinex_observation, 'r')
  # So far this test just iterates through to a specific known observation
  # and makes sure that value is parsed correctly.
  header, iter_obs = rinex.read_observation_file(f)

  assert header['x'] == -2300592.8570
  assert header['y'] == -3637848.2430
  assert header['z'] == 4691079.2150

  for i, obs in enumerate(iter_obs):
    if np.any(obs['time'] == datetime.datetime(2016, 2, 1, 12, 0, 4)):
      np.testing.assert_array_equal(obs.ix['GPS-02-1', 'carrier_phase'],
                                    np.array(126636838.718))
      break
    # This test should pass after four iterations, if not we fail
    if i > 10:
      assert False


@pytest.mark.parametrize('observation', ['short_baseline_cors/ssho032/ssho0320.16o',
                                         'rinex_211_examples/cebr049a00.16o',
                                         'rinex_211_examples/nnor049a00.16o',
                                         'rinex_211_examples/rinex_2_11_observation_with_intermixed_header.16o'])
def test_iter_observations(observation, datadir):
  path = datadir.join(observation).strpath
  header, iter_obs = rinex.read_observation_file(path)
  obs = [y for _, y in zip(range(5), iter_obs)]
  # make sure all observations are non null
  assert all([x.size for x in obs])
  assert len(header)


@pytest.mark.parametrize('navigation', ['short_baseline_cors/ssho032/ssho0320.16n',
                                         'rinex_210_examples/nnor049a00.16n',])
def test_iter_navigations(navigation, datadir):
  path = datadir.join(navigation).strpath
  header, iter_nav = rinex.read_navigation_file(path)
  nav = [y for _, y in zip(range(5), iter_nav)]
  # make sure all observations are non null
  assert all([x.size for x in nav])
  assert len(header)
