# Copyright (C) 2015 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
Contains a set of utility functions that help when dealing with SBP log files.

In particular this module provides a complete_messages_only wrapper which can
be used to wrap a LogIterator such that it will reassemble messages that have
been split across log entries.
"""

import types
import logging

from collections import defaultdict

from sbp.client.loggers.json_logger import JSONLogIterator


def is_split_message(msg):
  """
  Returns true if msg is part of a sequence of
  split messages.
  """
  return (hasattr(msg, 'header') and
          hasattr(msg.header, 'n_obs') and
          get_total(msg) > 1)


def get_count(msg):
  """
  Obtains the message count from an sbp message header
  by masking out the first four bits.
  """
  assert isinstance(msg.header.n_obs, int)
  return 0x0F & msg.header.n_obs


def get_total(msg):
  """
  Obtains the total message count from an sbp message
  header by shifting to obtain the last four bits
  """
  assert isinstance(msg.header.n_obs, int)
  return msg.header.n_obs >> 4


def get_time(msg):
  """
  Simply returns a tuple indicating the time of a message.
  """
  return (msg.header.t.wn, msg.header.t.tow)


def is_consistent(prev_messages, msg):
  """
  Checks to ensure that a new message (msg) is consistent
  with any messages we've already received (prev_messages).
  This assumes that the previous messages have all been checked
  for validity.

  Parameters
  ----------
  prev_messages : list of messages
    A list of messages we've already received (in chronological order)
  msg : message
    A new message that is to be checked against previous messages

  Returns
  -------
  consistent : bool
    A boolean indicating whether message is consistent with the
    rest of the messages.
  """
  # trival case where there are no previous messages
  # so the new message is marked as consistent.
  if not prev_messages:
    return get_count(msg) == 0

  # check if the new message is next in the sequence
  if get_count(prev_messages[-1]) != get_count(msg) - 1:
    prev_counts = [get_count(x) for x in prev_messages]
    logging.warn("A message was skipped because it did not fit"
                 " the previous messages count sequence."
                 " previous counts %s, new count %s."
                 % (str(prev_counts), get_count(msg)))
    return False

  # check if the timestamps are the same
  prev_times = set(get_time(x) for x in prev_messages)
  assert len(prev_times) == 1
  prev_times = prev_times.pop()
  if get_time(msg) != prev_times:
    logging.warn("The timestamp on the new message, %s, does not"
                 " match the rest of the messages ,%s."
                 % (str(prev_times), str(get_time(msg))))
    return False

  # all tests passed, return true.
  return True


def log_iterator(log):
  """
  Takes either a path to a log, a JSONLogIterator and
  returns a generator that returns (msg, data) pairs.

  This is a convenience function that can be used to make
  a function immune to issues created by JSONLogIterator,
  which doesn't follow the standard python iterator construct.
  """
  # if log is a string we create a JSON log iterator from it.
  if isinstance(log, basestring):
    return JSONLogIterator(log).next()
  # if log is a JSONLogIterator, return log.next() because
  # JSONLogIterator improperly uses the iter construct.
  elif isinstance(log, JSONLogIterator):
    return log.next()
  elif isinstance(log, types.GeneratorType):
    return log
  else:
    raise ValueError("Unknown log type: %s" % type(log))


def complete_messages_only(log):
  """
  Occasionally SBP log entries are split over several messages.  In
  particular when the number of satellites observed is too large
  to hold in a single SBP observation message.

  This method takes an iterator over SBP log messages but only
  yields complete messages.  This is done by creating a buffer
  of partial messages, waiting for a complete set and then
  yielding a single combined message.

  Parameters
  ----------
  log : string or sbp.client.loggers.base_logger.LogIterator
    This takes an iterator over log messages.  This can either
    be a path to a log, a LogIterator or a JSONLogIterator (which
    you would typically need to iter over using [x for x in log.next()].

  Returns
  -------
  A generator function that yeilds (msg, data) pairs.

  msg : sbp.msg.SBP
    An SBP message
  data : dict
    A dict holding the timestamp on the host chip.
  """
  buffer = defaultdict(list)

  for msg, data in log_iterator(log):
    if is_split_message(msg):
      # we received an insconsistent message so
      # we're discarding the entire sequence
      if not is_consistent(buffer[type(msg)], msg):
        logging.warn("Inconsistent log message found, skipping it.")
        # clear the buffer
        buffer[type(msg)] = []

      # add to latest message the buffer
      buffer[type(msg)].append(msg)

      if get_count(msg) == get_total(msg) - 1:
        # if this is the last message we reassemble the buffer into
        # a single message.
        iter_buffer = iter(buffer[type(msg)])
        # get a template message then iteratively extend it.
        full_msg = iter_buffer.next()
        # TODO: this only works for observation types at the moment.
        for m in iter_buffer:
          full_msg.obs.extend(m.obs)
        # only the most recent data is used.
        yield full_msg, data
        buffer[type(msg)] = []
    else:
      yield msg, data

