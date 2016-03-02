import logging
import numpy as np
import pandas as pd


def ecef_from_info(info_dict):
  """
  Given an info dictionary this returns the ecef position of the
  receiver.  If the position attributes don't exist this will
  return nans.
  """
  return np.array([info_dict.get('x', np.nan),
                   info_dict.get('y', np.nan),
                   info_dict.get('z', np.nan)])


def baseline_from_obs_set(obs_set):
  """
  Retrieves the rover and base ecef positions and computes
  the exepected baseline from them.
  """
  rover_pos = ecef_from_info(obs_set.get('rover_info', {}))
  base_pos = ecef_from_info(obs_set.get('base_info', {}))
  return rover_pos - base_pos


def baseline_error(soln, filter_name):
  """
  Given a solution set output from solution.solution called with a filter
  this will compute and return the baseline error.
  """
  if not 'base' in soln:
    raise ValueError("Observation set doesn't have a base observation.")
  if not filter_name in soln:
    raise ValueError("Observation set doesn't have %s field."
                     % filter_name)
  if not 'baseline_x' in soln[filter_name]:
    error = np.nan
  else:
    baseline = soln[filter_name][['baseline_x', 'baseline_y', 'baseline_z']]
    baseline = baseline.values[0]
    expected = baseline_from_obs_set(soln)
    error = np.linalg.norm(baseline - expected)
  return pd.DataFrame({'baseline_error': error},
                      index=soln[filter_name].index)


def get_time(metric_result):
  """
  Retrieves the time from the output of a metric, which is assumed to
  hold the time in the index of a pd.DataFrame.
  """
  out = metric_result.index.values[0]
  assert out.dtype.kind == 'M'
  return out


def time_to_convergence(iter_baseline_error, threshold=None,
                        required_time=None):
  """
  Takes a sequence of outputs from baseline_error() and returns the time
  of the first epoch and the time when the filter converged.  Convergence
  is defined is the   first time for which the baseline_error was less than
  or equal to `threshold` and stayed undert `threshold` for the subsequent
  `required_time`.

  Parameters
  ----------
  iter_baseline_error : list(output_from_baseline_error)
    An iterable of output from a sequence of solution sets that have been
    post-processed with the baseline_error() function. For example:
      [baseline_error(soln) for soln in solution.solution(obs_sets, filter)
  threshold : float
    The threshold that defines convergence (meters).  Default is 1.
  required_type : np.timedelta64
    A timedelta defining the amount of time the error needs to be below
    threshold to count as convergence. Default is 10 seconds.

  Returns
  -------
  first_time : np.datetime64
    The time of the first epoch.
  convergence_time : np.datetime64
    The time marking the start of the first period for which
    the convergence criterea were satisfied.
  """
  # Set the default parameters
  threshold = 1 if threshold is None else threshold
  required_time = (np.timedelta64(10, 's')
                   if required_time is None
                   else required_time)

  first_time = None
  conv_time = None
  time_below = np.timedelta64(0)
  # iterate over each of the error outputs.
  for err in iter_baseline_error:
    # record the very first time in the sequence.
    if first_time is None:
      first_time = get_time(err)
    # check the baseline error
    baseline_error = np.asscalar(err['baseline_error'].values)
    # if the baseline dipped below the fix threshold we
    # start monitoring how long it stays below
    if baseline_error <= threshold:
      # if this was the first time dipping below threshold
      # we record it as a possible convergence time
      if conv_time is None:
        conv_time = get_time(err)
      time_below = get_time(err) - conv_time
    else:
      # if the error either has been or just jumped above threshold
      # we reset the time_below and conv_time
      conv_time = None
      time_below = np.timedelta64(0)
    # convergence has been met,
    if time_below > required_time:
      return first_time, conv_time

  # If we've exhausted all the epochs we return NaT
  return first_time, np.datetime64('NaT')


def errors_after_convergence(baseline_errors, threshold=None,
                             required_time=None):
  """
  Iterates through baseline_errors and returns an array of the errors
  after convergence has been satisfied.
  """
  baseline_errors = list(baseline_errors)
  # determine the time of convergence
  first_time, conv_time = time_to_convergence(baseline_errors,
                                             threshold=threshold,
                                             required_time=required_time)
  if conv_time == np.datetime64('NaT'):
    return np.array([])
  # then filter through the baseline errors, only return those that are
  # after the convergence time
  return np.array([np.asscalar(x['baseline_error'].values)
                  for x in baseline_errors
                  if get_time(x) > conv_time])


def print_evaluations(soln_sets, filter_name, threshold=None, required_time=None):
  """
  Given a sequence of solution sets and a filter_name, this will produce
  summary statistics of the filter performance and print the results
  to standard out.
  """
  baseline_errors = [baseline_error(soln, filter_name) for soln in soln_sets]
  start_time, conv_time = time_to_convergence(baseline_errors,
                                             threshold=threshold,
                                             required_time=required_time)
  errors = errors_after_convergence(baseline_errors, threshold=threshold)

  fmt_str = '{:30}: {!s}'

  logging.info("========== Stats for {} ============".format(filter_name))
  logging.info(fmt_str.format("# Epochs", len(soln_sets)))
  logging.info(fmt_str.format("Time to Convergence", (conv_time - start_time).astype('timedelta64[s]')))
  logging.info(fmt_str.format("# Samples After Convergence", errors.size))
  logging.info(fmt_str.format("Mean Error", np.mean(errors)))
  logging.info(fmt_str.format("Std. Dev Error", np.std(errors)))

