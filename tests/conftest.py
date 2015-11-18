"""
This file contains configurations and additional fixtures
for the py.test framework.
"""
import os
import pytest
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

@pytest.yield_fixture
def jsonlog(datadir):
  basename = 'serial_link_log_20150314-190228_dl_sat_fail_test1.log.json.dat'
  filename = datadir.join(basename).strpath
  with JSONLogIterator(filename) as log:
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
