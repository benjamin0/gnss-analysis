"""
This file contains configurations and additional fixtures
for the py.test framework.
"""
import os
import pytest
import functools
import numpy as np
import pandas as pd

from distutils import dir_util

from gnss_analysis import filters


@pytest.fixture
def datadir(tmpdir):
  """
  Fixture responsible for moving all contents to a temporary directory so
  tests can use them freely.
  """
  test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
  assert os.path.exists(test_dir)
  if os.path.isdir(test_dir):
      dir_util.copy_tree(test_dir, bytes(tmpdir))
  return tmpdir


@pytest.fixture
def piksi_log_path(datadir):
  basename = 'partial_serial-link-20151221-142236.log.json'
  return datadir.join(basename).strpath


@pytest.fixture()
def piksi_observation_sets(piksi_log_path):
  from gnss_analysis.io import simulate
  return simulate.simulate_from_log(piksi_log_path)


@pytest.yield_fixture
def hdf5log(datadir):
  """
  Loads an example hdf5 log from disk.
  """
  basename = 'serial-link-20150429-163230.log.json.hdf5'
  filename = datadir.join(basename).strpath
  with pd.HDFStore(filename) as df:
    yield df


@pytest.fixture()
def ephemerides(piksi_log_path):
  """
  Loads the first available ephemeris data from log
  """
  from gnss_analysis.io import simulate
  for state in simulate.simulate_from_log(piksi_log_path):
    if state['ephemeris'].shape[0] >= 4:
      return state['ephemeris']


@pytest.fixture()
def synthetic_observations(ephemerides):
  from gnss_analysis import synthetic

  # use swift nav's approximate location (in ECEF) as reference
  ref_loc = np.array([-2704369.61784456,
                      - 4263211.09418205,
                      3884641.21270987])

  # Pick a common (arbitrary) time of transmission.
  tot = ephemerides['toe'].values[0] + np.timedelta64(40, 's')
  return synthetic.observation(ephemerides, ref_loc, tot,
                               rover_clock_error=0.)


@pytest.fixture()
def synthetic_observation_set(ephemerides):
  from gnss_analysis import synthetic, locations

  rover_ecef = locations.NOVATEL_ABSOLUTE
  base_ecef = locations.LEICA_ABSOLUTE

  tot = ephemerides['toc'].values[0] + np.timedelta64(100, 's')

  return synthetic.synthetic_observation_set(ephemerides,
                                   rover_ecef, base_ecef,
                                   tot)


@pytest.fixture()
def synthetic_stationary_observations(ephemerides):
  """
  Returns an iterator over synthetic states for which the rover and
  base station are stationary.
  """
  from gnss_analysis import synthetic, locations, ephemeris

  def iter_states(time_steps):
    rover_ecef = locations.NOVATEL_ABSOLUTE
    base_ecef = locations.LEICA_ABSOLUTE

    for dt in time_steps:
      toa = np.max(ephemerides['toe'].values) + np.timedelta64(100 + int(1e3 * dt), 'ms')
      state = synthetic.synthetic_observation_set(ephemerides, rover_ecef,
                                        base_ecef, toa)
      state['base'] = ephemeris.add_satellite_state(state['base'],
                                                    state['ephemeris'])
      state['rover'] = ephemeris.add_satellite_state(state['rover'],
                                                     state['ephemeris'])
      yield state.copy()

  return iter_states(np.linspace(0., 100., 1001.))


@pytest.fixture
def rinex_observation(datadir):
  basename = 'short_baseline_cors/seat032/seat0320.16o'
  return datadir.join(basename).strpath


@pytest.fixture
def rinex_base(datadir):
  basename = 'short_baseline_cors/ssho032/ssho0320.16o'
  return datadir.join(basename).strpath


@pytest.fixture
def rinex_navigation(datadir):
  basename = 'short_baseline_cors/seat032/seat0320.16n'
  return datadir.join(basename).strpath


@pytest.fixture
def rinex_observation_sets(rinex_observation,
                           rinex_navigation,
                           rinex_base):
  from gnss_analysis.io import simulate
  return simulate.simulate_from_rinex(rinex_observation,
                                      rinex_navigation,
                                      rinex_base)

@pytest.fixture
def cors_drops_reference(datadir):
  rover = datadir.join('cors_drops_reference/seat032/partial_seat0320.16o').strpath
  nav = datadir.join('cors_drops_reference/seat032/seat0320.16n').strpath
  base = datadir.join('cors_drops_reference/ssho032/partial_ssho0320.16o').strpath
  return rinex_observation_sets(rover, nav, base)


@pytest.fixture
def cors_short_baseline(datadir):
  rover = datadir.join('short_baseline_cors/seat032/seat0320.16o').strpath
  nav = datadir.join('short_baseline_cors/seat032/seat0320.16n').strpath
  base = datadir.join('short_baseline_cors/ssho032/ssho0320.16o').strpath
  return rinex_observation_sets(rover, nav, base)


@pytest.fixture
def piksi_roof(datadir):
  log_path = datadir.join('partial_serial-link-20151221-142236.log.json').strpath
  return piksi_observation_sets(log_path)


@pytest.fixture(params=['cors_short_baseline', 'cors_drops_reference', 'piksi'])
def observation_sets(datadir,
                     request):

  sets = {'cors_drops_reference': cors_drops_reference,
          'cors_short_baseline': cors_short_baseline,
          'piksi': piksi_roof}

  if not request.param in sets:
    raise NotImplementedError("unhandled observation set type %s"
                              % request.param)

  return sets[request.param](datadir)


# This fixture produces a subset of the observation_sets
@pytest.fixture(params=['cors_short_baseline'])
def cors_observation_sets(datadir,
                          request):
  return observation_sets(datadir, request)


@pytest.fixture(params=['static_kalman', 'dynamic_kalman',
                        'raim_disable_swiftnav'])
def dgnss_filter_class(request):
  if request.param == "static_kalman":
    return filters.StaticKalmanFilter
  elif request.param == "dynamic_kalman":
    return filters.DynamicKalmanFilter
  elif request.param == 'raim_disable_swiftnav':
    pytest.skip("The swiftnav filter isn't working.")
    return functools.partial(filters.SwiftNavDGNSSFilter,
                             disable_raim=True)
  else:
      raise ValueError("invalid internal test config")
