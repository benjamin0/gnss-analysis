import os
import sys
import time
import logging
import argparse
import itertools
import progressbar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sbp.client.loggers.json_logger import JSONLogIterator

from gnss_analysis import filters
from gnss_analysis import solution
from gnss_analysis import ephemeris
from gnss_analysis.io import simulate, hdf5, rinex, sbp_utils
from gnss_analysis.tools import common


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


def baseline_error(x):
  """
  Given a solution set output from solution.solution called with a filter
  this will compute and return the baseline error.
  """
  if not 'base' in x:
    raise ValueError("Observation set doesn't have a base observation.")
  if not 'rover_pos' in x:
    raise ValueError("Observation set doesn't have position estimates.")
  baseline = x['rover_pos'][['baseline_x', 'baseline_y', 'baseline_z']]
  baseline = baseline.values[0]
  expected = baseline_from_obs_set(x)
  error = np.linalg.norm(baseline - expected)
  return pd.DataFrame({'baseline_error': error},
               index=x['rover_pos'].index)


def plot_metric(obs_sets, filters,
                metric, metric_name, metric_units,
                draw_every=10):
  """
  Takes an iterable of observation sets, a set of filters
  and a metric and plots the result for each filter iteratively.
  """
  filter_names, filters = zip(*filters.iteritems())

  # produce an iterable of solution sets for each filter
  # note that this relies on solution.solution not mutating, which
  # should be tested in test_solution.py
  obs_set_copies = itertools.tee(obs_sets, len(filters))
  soln_sets = [solution.solution(o, f())
               for f, o in zip(filters, obs_set_copies)]

  # we end up iterating through solutions and plotting the results
  # as they come in.  metric_buffer aggregates all the values fo
  # the metric to aid with plotting.
  metric_buffer = pd.DataFrame()

  # iterate over epochs, producing the solution set for each
  # filter at the given epoch.
  for i, solns in enumerate(itertools.izip(*soln_sets)):
    # compute the metric for each of the filter's outputs
    next_val = pd.concat([metric(s) for s in solns], axis=1)
    # add to the buffer
    metric_buffer = pd.concat([metric_buffer, next_val])

    if i == 0:
      # initialize the plot if this is the first iteration
      sns.set_style('darkgrid')
      fig, ax = plt.subplots(1, 1)
      plt.ion()
      lines = plt.plot(np.arange(metric_buffer.index.size),
                       metric_buffer.values)
      plt.legend(filter_names)
      plt.ylabel("%s (%s)" % (metric_name, metric_units))
      plt.xlabel("Epochs")
      plt.show()
      plt.pause(0.001)
    else:
      # otherwise update the plot by first resetting the data of the line plots
      for line, vals in zip(lines, metric_buffer.values.T):
        line.set_data(np.arange(metric_buffer.index.size),
                      vals)
      # then (possibly) adjusting the xlimits of the plot
      if metric_buffer.index.size > ax.get_xlim()[1]:
        ax.set_xlim([0, 1.61 * metric_buffer.index.size])
      # then we look at the y limits which is a bit tricker since
      # we dont' know for sure that the values are monotonic and/or zero origin.
      # first we look at the current limits of the data.
      cur_lim = np.array([np.min(metric_buffer.values),
                          np.max(metric_buffer.values)])
      # then check if any the day falls outside the plot limits
      if cur_lim[0] <= ax.get_ylim()[0] or cur_lim[1] >= ax.get_ylim()[1]:
        # if it does fall outside we increase the y axis by 60 percent in a
        # way that keeps the data centered
        delta = 1.61 * (np.max(metric_buffer.values) - np.min(metric_buffer.values))
        mid = np.mean(cur_lim)
        new_lim = [mid - delta / 2, mid + delta / 2]
        ax.set_ylim(new_lim)
      # the draw function takes a lot of computation time, by only drawing every
      # so often we can keep the computation bottleneck the filter, not the plotting
      if np.mod(i, draw_every) == 0:
        plt.draw()
        plt.pause(0.001)


def compare(args):
  """
  Interprets the arguments and runs the corresponding comparison.
  """

  cnt, obs_sets = common.get_observation_sets(args.input.name,
                                              args.base,
                                              args.navigation)

  # if a number of observations was specified we take
  # only the first n of them and update the total count
  if args.n is not None:
    obs_sets = (x for _, x in itertools.izip(range(args.n), obs_sets))
    cnt = min(args.n, cnt)

  # TODO: in the future we may want to plot different metrics, as
  # long as these three things are defined we should be able to
  # reuse the logic below.
  metric = baseline_error
  metric_name = 'baseline error'
  metric_units = 'm'

  plot_metric(obs_sets, args.filters,
              metric, metric_name, metric_units,
              draw_every=args.draw_every)



def create_parser(parser):
  parser.add_argument('input', type=argparse.FileType('r'),
                      help='Specify the input file that contains the rover'
                           ' (and possibly base/navigation) observations.'
                           ' The file type is infered from the extension,'
                           ' (SBP=".json", RINEX="*o", HDF5=[".h5", ".hdf5"])'
                           ' and the appropriate parser is used.')
  parser.add_argument('--base',
                      help='Optional source of base observations.',
                      default=None)
  parser.add_argument('--navigation',
                      help='Optional source of navigation observations.',
                      default=None)
  parser.add_argument('--output', type=argparse.FileType('w'), default=None,
                      help='Optional output path to use, default creates'
                           ' one from the input path and other arguments.')
  parser.add_argument("-n", type=int, default=None,
                      help="The number of observation sets that will be read,"
                           " default uses all available.")
  parser.add_argument('--profile', default=False, action="store_true")
  parser.add_argument("--filters",
                      choices=filters.lookup.keys(), nargs='*',
                      required=True)
  parser.add_argument("--draw-every", type=int, default=10,
                      help="The number of epochs between plot refreshes,"
                           " refreshing every iteration (1) significantly"
                           " slows down the script. (default=10)")
  return parser


if __name__ == "__main__":
  script_name = os.path.basename(sys.argv[0])
  parser = argparse.ArgumentParser(description=
                                   """
%(script_name)s

A tool for comparing filters.
"""
                                    % {'script_name': script_name})
  parser = create_parser(parser)
  args = parser.parse_args()

  args.filters = common.resolve_filters(args.filters)

  logging.basicConfig(stream=sys.stderr, level=logging.INFO)

  if args.profile:
    import cProfile
    cProfile.runctx('compare(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    compare(args)
