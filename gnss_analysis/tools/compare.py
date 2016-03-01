import os
import sys
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from gnss_analysis import filters
from gnss_analysis.evaluation import visual
from gnss_analysis import solution
from gnss_analysis.tools import common
from gnss_analysis.evaluation import metrics


def peek(iterator):
  """
  Pops the first element from an iterator, then it along
  with a reconstruction of the complete iterator.
  """
  next_element = iterator.next()
  return next_element, itertools.chain([next_element], iterator)


def maybe_add_filters(obs_sets, filters):
  """
  Takes a sequence of observation_sets and possibly adds filter
  solutions to them.  If the filter output already exists in the
  first observation set, that filter is skipped under the assumption
  it has already been run.
  """
  # peek at the first observation set to decide if some of the
  # filters have already been run
  first_obs_set, obs_sets = peek(obs_sets)
  # reduce to (and instantiate) the set of filters that need to be run
  filters_to_run = {fname: fclass() for fname, fclass in filters.iteritems()
                    if fname not in first_obs_set}
  # only run the solution thread if there are filters left to be run
  if len(filters_to_run):
    obs_sets = solution.solution(obs_sets, **filters_to_run)
  return obs_sets


def iter_metric(soln_sets, filters, metric):
  """
  Takes an iterable of observation sets, a set of filters
  and a metric and plots the result for each filter iteratively.
  """
  def compute_metric(soln, metric):
    out = pd.concat([metric(soln, fname) for fname in filters.keys()],
                    axis=1)
    out.columns = pd.Index(filters.keys())
    return out

  return (compute_metric(soln, metric) for soln in soln_sets)


def plot_compare(soln_sets, args):
  """
  Creates visual comparison of the filter results.
  """
  sns.set_style('darkgrid')
  metric = metrics.baseline_error
  metric_name = 'baseline error'
  metric_units = 'm'

  metric_iter = iter_metric(soln_sets, args.filters, metric)
  if args.draw_every > 0:
    visual.plot_metric_iterative(metric_iter,
                                 metric_name, metric_units,
                                 draw_every=args.draw_every)
  else:
    visual.plot_metric(pd.concat(metric_iter), metric_name, metric_units)
    plt.show()


def stat_compare(soln_sets, args):
  """
  Prints summary statistics to standard out.
  """
  soln_sets = list(soln_sets)
  for fname in args.filters.keys():
    metrics.print_evaluations(soln_sets, fname)


def compare(args):
  """
  Interprets the arguments and runs the corresponding comparison.
  """
  obs_sets = common.get_observation_sets(args.input,
                                         args.base,
                                         args.navigation,
                                         max_epochs=args.n)
  soln_sets = maybe_add_filters(obs_sets, args.filters)
  # use the function specific to the subparser to compare
  args.compare_func(soln_sets, args)


def create_parser(parser):
  parser.add_argument('--profile', default=False, action="store_true")

  subparsers = parser.add_subparsers(title="Comparison Modes")
  # Add the subparser for the plot command.
  plot_parser = subparsers.add_parser("plot", help="Produces visual plots of"
                                       " one or more filter's performance."
                                       " See plot --help for details.")
  common.add_io_arguments(plot_parser)
  plot_parser.add_argument("--draw-every", type=int, default=10,
                           help="The number of epochs between plot refreshes,"
                                " refreshing every iteration (1) significantly"
                                " slows down the script. A value of zero (or"
                                " less) will wait until all observations were"
                                " processed before creating the plot. Default"
                                " is 10")
  plot_parser.set_defaults(compare_func=plot_compare)

  # Add the subparser for the stats command
  stats_parser = subparsers.add_parser("stats",
                                       help="Produces summary statistics of"
                                       " one or more filter's performance and"
                                       " prints the results to standard out."
                                       " See stats --help for details.")
  stats_parser = common.add_io_arguments(stats_parser)
  stats_parser.set_defaults(compare_func=stat_compare)

  return parser


if __name__ == "__main__":
  script_name = os.path.basename(sys.argv[0])
  parser = argparse.ArgumentParser(description=
                                   """
%(script_name)s

A tool for comparing filters.  There are multiple modes of comparison
which allow for both visual and quantitative comparison.
"""
                                    % {'script_name': script_name})

  parser = create_parser(parser)
  args = parser.parse_args()
  args.filters = common.resolve_filters(args.filter_names)

  logging.basicConfig(stream=sys.stderr, level=logging.INFO)

  if args.profile:
    import cProfile
    cProfile.runctx('compare(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    compare(args)
