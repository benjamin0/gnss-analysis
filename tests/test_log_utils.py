# Copyright (C) 2015 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

import mock
import pytest
import numpy as np

import sbp.observation as ob

from sbp.msg import SBP
from sbp.client.loggers.json_logger import JSONLogIterator

from gnss_analysis import log_utils


def test_runs(jsonlog):
  """
  Straight forward test that simply makes sure the test log can
  be parsed without failure.
  """
  for msg, data in log_utils.complete_messages_only(jsonlog.next()):
    pass


def _make_n_obs(count, total):
  return (total << 4) | count


def test_count_total():
  """
  Checks to make sure the get_count and get_total
  method work properly.
  """
  # count is stored in four bits, so we should
  # have a max count size of 2^4 - 1
  for i in np.arange(2 ** 4):
    total = np.random.randint(2 ** 4)
    # build a random n_obs
    n_obs = _make_n_obs(i, total)
    mock_msg = mock.Mock(**{'header.n_obs': n_obs})
    actual_count = log_utils.get_count(mock_msg)
    assert actual_count == i
    actual_total = log_utils.get_total(mock_msg)
    assert actual_total == total


def test_is_split_message():
  """
  Creates a bunch of messages and tests to make sure is_split_message
  processes them properly
  """
  _tests = [(True, mock.Mock(**{'header.n_obs': _make_n_obs(0, 2)})),
            (True, mock.Mock(**{'header.n_obs': _make_n_obs(0, 3)})),
            (False, mock.Mock(**{'header.n_obs': _make_n_obs(0, 1)})),
            (True, mock.Mock(**{'header.n_obs': _make_n_obs(1, 3)})),
            (False, mock.Mock(**{'header': 1}))]

  for expected, msg in _tests:
    assert log_utils.is_split_message(msg) == expected


def test_log_iterator(jsonlogpath):
  """
  Make sure log_iterator returns an iterator over messages
  and data if you pass in a path, log_iterator or log_iterator.next()
  """

  def assert_is_log_iterator(log):
    msg, data = log.next()
    assert isinstance(msg, SBP)
    assert isinstance(data, dict)

  assert_is_log_iterator(log_utils.log_iterator(jsonlogpath))

  with JSONLogIterator(jsonlogpath) as log:
    assert_is_log_iterator(log_utils.log_iterator(log))

  with JSONLogIterator(jsonlogpath) as log:
    assert_is_log_iterator(log_utils.log_iterator(log.next()))




def test_is_consistent():
  """
  Make sure is_consistent() works on a set of synthetic messages.
  """
  def mock_message(count, total, wn, tow):
    return mock.Mock(**{'header.n_obs': _make_n_obs(count, total),
                        'header.t.wn': wn,
                        'header.t.tow': tow})

  wn, tow = (1870, 1000)
  prev_messages = [mock_message(i, 4, wn, tow) for i in range(3)]

  _tests = [(True, (prev_messages[:2],
                    mock_message(2, 4, wn, tow))),
            (True, (prev_messages,
                    mock_message(3, 4, wn, tow))),
            # count is message 5 of a total of 4, so False
            (False, (prev_messages,
                     mock_message(4, 4, wn, tow))),
            # count is earlier than the last message, so False
            (False, (prev_messages,
                     mock_message(2, 4, wn, tow))),
            ]

  for expected, (prev, new) in _tests:
    print prev, new
    assert log_utils.is_consistent(prev, new) == expected
