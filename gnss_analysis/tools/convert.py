import os
import sys
import time
import logging
import argparse
import progressbar
import pandas as pd

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from sbp.client.loggers.json_logger import JSONLogIterator

from gnss_analysis import filters
from gnss_analysis import solution
from gnss_analysis import ephemeris
from gnss_analysis.io import simulate, hdf5, rinex, sbp_utils


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


def get_observation_sets(input_path, base_path=None, nav_path=None):
  """
  Takes care of the logic required to determine which format an input
  file (and optional base and navigation files) are stored in.  Once
  the format.
  
  Parameters
  ----------
  input_path : string
    The path to an input file.  Could be an SBP json log file, a RINEX
    observation file or an HDF5 file.  The extension on this file is used
    to determine the format time.
  base_path : string
    An optional argument which specifies the file that holds base
    observations.
  nav_path : string
    An optional argument which specifies the file that holds navigation
    messages.  For RINEX this can be left out if the navigation file
    path is the same as the input_path with the 'o' replaced to 'n'.
    
  Returns
  -------
  cnt : int
    The (approximate) number of observations held in the file, this helps
    us build a progress bar to monitor conversion.
  obs_sets : generator
    The output of the simulate function that corresponds to the input
    path.  Should be a generator of observation sets, each of which
    holds rover and (optionall) ephemeris and base fields.
  """
  if input_path.endswith('.json'):
    if base_path is not None or nav_path is not None:
      raise NotImplementedError("The base and navigation arguments are not"
                                " applicable when using an sbp log file.")
    cnt = sbp_utils.count_rover_observation_messages(input_path)
    return cnt, simulate.simulate_from_log(input_path)

  elif input_path.endswith('.hdf5'):
    if base_path is not None or nav_path is not None:
      # TODO: we could easily change this
      raise NotImplementedError("The base and navigation arguments are not"
                                " applicable when using an HDF5 file.")

    with pd.HDFStore(input_path, 'r') as store:
      cnt = store['rover'].shape[0]

    # return the observations sets from the HDF5 file
    return cnt, simulate.simulate_from_hdf5(input_path)

  elif input_path.endswith('o'):
    # either use the explicitly provided nav path or try and infer it.
    nav_path = nav_path or rinex.infer_navigation_path(input_path)
    # If we want to show progress we read the whole file first, then
    # pass in an iterator over lines.
    # it's fine if base_path is None here.
    cnt = rinex.count_observations(input_path)
    return cnt, simulate.simulate_from_rinex(input_path, nav_path, base_path)


def convert(args):
  """
  Interprets the arguments and runs the corresponding conversion.
  """
  if not args.output.rsplit(".", 1)[1] in ['h5', 'hdf5']:
    raise ValueError("Expected the output path to end with either '.h5',"
                     " or '.hdf5'.")

  cnt, obs_sets = get_observation_sets(args.input.name,
                                       args.base,
                                       args.navigation)

  # optionally add the satellite state to rover and base
  if args.calc_sat_state:
    obs_sets = add_satellite_state(obs_sets)

  # if a number of observations was specified we take
  # only the first n of them and update the total count
  if args.n is not None:
    obs_sets = (x for _, x in zip(range(args.n), obs_sets))
    cnt = min(args.n, cnt)

  logging.info("About to parse %d epochs of observations" % cnt)
  bar = progressbar.ProgressBar(maxval=cnt)
  obs_sets = bar(obs_sets)

  # optionally fun a filter on the observations before saving.
  if args.filter is not None:
    logging.info("Running filter (%s) using the observations"
                 % type(args.filter))
    obs_sets = solution.solution(obs_sets, args.filter)

  logging.info("Writting to HDF5")
  hdf5.to_hdf5(obs_sets, args.output)


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
  parser.add_argument("--filter", choices=['static', 'dynamic'],
                      default=None)
  args = parser.parse_args()

  # if the output file was not provided we infer it from the input
  if args.output is None:
    # split into directory and basename
    dirname = os.path.dirname(args.input.name)
    basename = os.path.basename(args.input.name)
    # add the filter name to the output file if possible
    basename = '_'.join([args.filter, basename])
    # reassmble the output name

    args.output = os.path.join(dirname, '%s.hdf5' % basename)
    logging.info("output path: %s" % args.output)
  else:
    # we only need the output path if it was specified.
    args.output = args.output.name

  # instantiate the filter if it was provided
  all_filters = {'static': filters.StaticKalmanFilter(),
                 'dynamic': filters.DynamicKalmanFilter()}
  if args.filter is not None:
    args.filter = all_filters[args.filter]

  if args.profile:
    import cProfile
    cProfile.runctx('convert(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    convert(args)
