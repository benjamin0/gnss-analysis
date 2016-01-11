"""
This file contains configurations and additional fixtures
for the py.test framework.
"""
import os
import pytest
import numpy as np
import pandas as pd

from distutils import dir_util

from sbp.client.loggers.json_logger import JSONLogIterator


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
def jsonlogpath(datadir):
  basename = 'partial_serial-link-20151221-142236.log.json'
  return datadir.join(basename).strpath


@pytest.yield_fixture
def jsonlog(jsonlogpath):
  with JSONLogIterator(jsonlogpath) as log:
    yield log


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
def ephemerides(jsonlog):
  """
  Loads the first available ephemeris data from log
  """
  from gnss_analysis import simulate
  for state in simulate.simulate_from_log(jsonlog):
    if state['ephemeris'].shape[0] >= 4:
      return state['ephemeris']


@pytest.fixture()
def synthetic_observations(ephemerides):
  from gnss_analysis import synthetic, ephemeris

  # use swift nav's approximate location (in ECEF) as reference
  ref_loc = np.array([-2704369.61784456,
                      - 4263211.09418205,
                      3884641.21270987])

  # Pick a common (arbitrary) time of transmission.
  ref_time = ephemerides.iloc[0][['toe_wn', 'toe_tow']].copy()
  tot = ref_time.rename({'toe_wn': 'wn', 'toe_tow': 'tow'})
  tot['tow'] += 40

  return synthetic.observations_from_tot(ephemerides, ref_loc, tot,
                                         rover_clock_error=0.,
                                         account_for_sat_error=True)
