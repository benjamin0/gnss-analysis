import os
import sys
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

from gnss_analysis import filters
from gnss_analysis.evaluation import visual
from gnss_analysis import solution
from gnss_analysis.tools import common
from gnss_analysis.evaluation import metrics


def peek(iterator):
  next_element = iterator.next()
  return next_element, itertools.chain([next_element], iterator)


def maybe_add_filters(obs_sets, filters):
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


def compare(args):
  """
  Interprets the arguments and runs the corresponding comparison.
  """
  obs_sets = common.get_observation_sets(args.input,
                                         args.base,
                                         args.navigation,
                                         max_epochs=args.n)

  soln_sets = maybe_add_filters(obs_sets, args.filters)



  if True:
    sns.set_style('darkgrid')
    metric = metrics.baseline_error
    metric_name = 'baseline error'
    metric_units = 'm'
    visual.plot_metric_iterative(iter_metric(soln_sets, args.filters, metric),
                                   metric_name, metric_units,
                                   draw_every=args.draw_every)
  else:
    soln_sets = list(soln_sets)
    for fname in args.filters.keys():
      metrics.print_evaluations(soln_sets, fname)



def create_parser(parser):
  parser.add_argument('input',
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
  parser.add_argument('--output', default=None,
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
