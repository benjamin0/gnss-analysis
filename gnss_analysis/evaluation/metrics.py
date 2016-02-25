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
  out = metric_result.index.values[0]
  assert out.dtype.kind == 'M'
  return out


def time_to_convergence(iter_baseline_error, threshold=1.,
                        required_time=np.timedelta64(10, 's')):
  first_time = None
  fix_time = None
  time_below = np.timedelta64(0)

  for err in iter_baseline_error:
    # record the very first time in the sequence.
    if first_time is None:
      first_time = get_time(err)
    # check the baseline error
    baseline_error = np.asscalar(err['baseline_error'].values)
    # if the baseline dipped below the fix threshold we
    # start monitoring how long it stays below
    if baseline_error <= threshold:
      if fix_time is None:
        fix_time = get_time(err)
      time_below = get_time(err) - fix_time
    else:
      fix_time = None
      time_below = np.timedelta64(0)

    if time_below > required_time:
      return first_time, fix_time

  return np.datetime64('NaT'), np.datetime64('NaT')


def errors_after_convergence(baseline_errors, threshold=1.):
  baseline_errors = list(baseline_errors)
  # burn through until the first fix.
  first_time, fix_time = time_to_convergence(baseline_errors, threshold)
  if first_time == np.datetime64('NaT'):
    return np.array([np.nan])
  # then exhaust the rest of the errors and compute summary statistics.
  return np.array([np.asscalar(x['baseline_error'].values)
                  for x in baseline_errors
                  if get_time(x) > fix_time])


def print_evaluations(soln_sets, filter_name):
  baseline_errors = [baseline_error(soln, filter_name) for soln in soln_sets]
  start_time, conv_time = time_to_convergence(baseline_errors,
                                             threshold=1.)
  errors = errors_after_convergence(baseline_errors, threshold=1.)

  fmt_str = '{:30}: {!s}'

  print "========== Stats for {} ============".format(filter_name)
  print fmt_str.format("# Epochs", len(soln_sets))
  print fmt_str.format("Time to Convergence", (conv_time - start_time).astype('timedelta64[s]'))
  print fmt_str.format("# Samples After Convergence", errors.size)
  print fmt_str.format("Mean Error", np.mean(errors))
  print fmt_str.format("Std. Dev Error", np.std(errors))

