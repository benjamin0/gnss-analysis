import os
import sys
import time
import logging
import argparse
import itertools
import progressbar
import pandas as pd

from sbp.client.loggers.json_logger import JSONLogIterator

from gnss_analysis import filters
from gnss_analysis import solution
from gnss_analysis import ephemeris
from gnss_analysis.io import simulate, hdf5, rinex, sbp_utils
from gnss_analysis.tools import common


def add_satellite_state(obs_sets):
  """
  Takes an iterator of observation_sets and returns a generator which
  consists of the same observation sets, but with satellite state
  added to them.
  """
  def add_state(obs_set):
    # Adds satellite state to a single observation set
    obs_set['rover'] = ephemeris.add_satellite_state(obs_set['rover'],
                                                     obs_set['ephemeris'])
    # Check if there are base observations and add state to those as well
    if 'base' in obs_set:
      obs_set['base'] = ephemeris.add_satellite_state(obs_set['base'],
                                                       obs_set['ephemeris'])
    return obs_set
  # keep this a generator so we don't accidentally add state to all
  # observations despite having specified (using -n) to save only the first n
  return (add_state(x) for x in obs_sets)


def convert(args):
  """
  Interprets the arguments and runs the corresponding conversion.
  """
  if not args.output.rsplit(".", 1)[1] in ['h5', 'hdf5']:
    raise ValueError("Expected the output path to end with either '.h5',"
                     " or '.hdf5'.")

  cnt, obs_sets = common.get_observation_sets(args.input.name,
                                              args.base,
                                              args.navigation)

  # optionally add the satellite state to rover and base
  if args.calc_sat_state:
    obs_sets = add_satellite_state(obs_sets)

  # if a number of observations was specified we take
  # only the first n of them and update the total count
  if args.n is not None:
    obs_sets = (x for _, x in itertools.izip(range(args.n), obs_sets))
    cnt = min(args.n, cnt)

  logging.info("About to parse %d epochs of observations" % cnt)
  bar = progressbar.ProgressBar(maxval=cnt)
  obs_sets = bar(obs_sets)

  # optionally run a filter on the observations before saving.
  if args.filter is not None:
    logging.info("Running filter (%s) using the observations"
                 % type(args.filter))
    obs_sets = solution.solution(obs_sets, args.filter)

  logging.info("Writting to HDF5")
  hdf5.to_hdf5(obs_sets, args.output)


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
  parser.add_argument('--calc-sat-state', action="store_true", default=False,
                      help="If specified the satellite state is computed"
                           " prior to saving to HDF5.")
  parser.add_argument('--profile', default=False, action="store_true")
  parser.add_argument("--filter", dest="filter_name",
                      choices=filters.lookup.keys(),
                      default=None)
  return parser


if __name__ == "__main__":
  script_name = os.path.basename(sys.argv[0])
  parser = argparse.ArgumentParser(description=
                                   """
%(script_name)s
A tool for converting from a variety of different formats to HDF5,
with the option of precomputing satellite states or running a
filter before saving.
"""
                                    % {'script_name': script_name})
  parser = create_parser(parser)
  args = parser.parse_args()
  args.output = common.infer_output(args.output, args.input, args.filter_name)
  args.filter = common.resolve_filters(args.filter_name)

  logging.basicConfig(stream=sys.stderr, level=logging.INFO)

  if args.profile:
    import cProfile
    cProfile.runctx('convert(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    convert(args)
