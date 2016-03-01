import os
import logging
import itertools
import progressbar
import pandas as pd

from gnss_analysis import filters
from gnss_analysis.io import sbp_utils, simulate, rinex


def get_observation_sets(input_path, base_path=None, nav_path=None,
                         max_epochs=None, show_progress=True):
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
    # piksi observations
    if base_path is not None or nav_path is not None:
      raise NotImplementedError("The base and navigation arguments are not"
                                " applicable when using an sbp log file.")
    cnt = sbp_utils.count_rover_observation_messages(input_path)
    obs_sets = simulate.simulate_from_log(input_path)

  elif input_path.endswith('.hdf5'):
    # HDF5 observations
    if base_path is not None or nav_path is not None:
      # TODO: we could change this to accept pulling
      #   base and ephemeris observations from a different
      #   HDF5 file.
      raise NotImplementedError("The base and navigation arguments are not"
                                " applicable when using an HDF5 file.")

    with pd.HDFStore(input_path, 'r') as store:
      cnt = store['rover'].index.levels[0].size

    # return the observations sets from the HDF5 file
    obs_sets = simulate.simulate_from_hdf5(input_path)

  elif input_path.endswith('o'):
    # RINEX observations
    # either use the explicitly provided nav path or try and infer it.
    nav_path = nav_path or rinex.infer_navigation_path(input_path)
    # If we want to show progress we read the whole file first, then
    # pass in an iterator over lines.
    # it's fine if base_path is None here.
    cnt = rinex.count_observations(input_path)
    obs_sets = simulate.simulate_from_rinex(input_path, nav_path, base_path)
  else:
    raise ValueError("Input path %s has an unknown extension." % input_path)

  # if a number of observations was specified we take
  # only the first n of them and update the total count
  if max_epochs is not None:
    obs_sets = (x for _, x in itertools.izip(xrange(max_epochs), obs_sets))
    cnt = min(max_epochs, cnt)

  # optionally displays a progress bar in stdout when iterating
  # through the obs_sets generator
  if show_progress:
    logging.info("About to parse %d epochs of observations" % cnt)
    bar = progressbar.ProgressBar(maxval=cnt)
    obs_sets = bar(obs_sets)
  return obs_sets


def infer_output(output, input):
  """
  A utility function that determines the output path by
  either extracting it from the 'output' argument (which
  can be a string or argparse.FileType).  If output is None
  a default output path is inferred from the input path.
  """
  # if the output file was not provided we infer it from the input
  if output is None:
    # split into directory and basename
    dirname = os.path.dirname(input)
    basename = os.path.basename(input)
    # reassmble the output name
    output_name = os.path.join(dirname, '%s.hdf5' % basename)
    logging.info("Using default output path: %s" % output_name)
  else:
    # we only need the output path if it was specified.
    output_name = getattr(output, 'name', output)
  return output_name


def resolve_filters(filter_name):
  """
  A utility function that post processes typical arguments filling
  in default things such as args.output, and interpreting filter
  names.
  """
  # if no name was provided we return none for the filter class
  if filter_name is None:
    return None

  # lookup the filter (or filters) using the filter module lookup
  if isinstance(filter_name, basestring):
    if not filter_name in filters.lookup:
      raise ValueError("Expected filter name to be one of %s, got %s."
                       % (','.join(filters.lookup.keys()), filter_name))
    return {filter_name: filters.lookup[filter_name]}
  else:
    all_filters = [resolve_filters(f) for f in filter_name]
    return dict(itertools.chain(*[x.iteritems() for x in all_filters]))


def add_io_arguments(parser):

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
  parser.add_argument("--filters", dest="filter_names", nargs='*',
                      choices=filters.lookup.keys(),
                      default=None)

  return parser