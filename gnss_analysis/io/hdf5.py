import logging
import numpy as np
import pandas as pd


def to_hdf5(obs_sets, output_path, strict=True,
            **kwdargs):
  """
  Takes an iterator over observation sets and serializes them to
  an HDF5 file.
  
  Parameters
  ----------
  obs_sets : iterable of observations sets (see simulate.py)
    Takes an iterator over observation sets.  These observation
    sets are expected to contain a rover field and otherwise
    can contain any number of fields as long as they map to
    pandas data frames with a defined index.  Corresponding
    info dictionaries are serialized to the HDF5 file attributes.
  output_path: string
    The path to the location where the HDF5 file is written
  **kwdargs
    Any additional keyword arguments are passed on to
    pd.HDF5Store() allowing modification of defaults.
    Default behavior is to use zlib compression
  
  Returns
  -------
  None
  """
  logging.info("Reading input")
  # This may actually take a while if the observations haven't
  # already been read.
  obs_sets = list(obs_sets)
  # the list of all epochs in the observation set
  epochs = [x['epoch'] for x in obs_sets]
  def build_df(group_name, obs_sets):
    # Creates a list of tuples, the first element is the epoch
    # and the second is the observation for group_name at that epoch
    # any epochs without observations are skipped
    pieces = [(np.repeat(e, x[group_name].shape[0]), x[group_name])
              for e, x in zip(epochs, obs_sets)
              if group_name in x]
    # Concatenates observations into a single data frame
    # and creates a multiindex which makes iteration
    df = pd.concat([y for _, y in pieces])
    epoch_index = np.concatenate([y for y, _ in pieces])
    assert df.index.name is not None
    # reorganize so the index is epoch and the current index
    df.index = pd.MultiIndex.from_arrays([epoch_index, df.index],
                                         names=['epoch', df.index.name])
    # epoch has been added to the index, if it's a col get rid of it.
    if 'epoch' in df:
      df.drop('epoch', axis=1, inplace=True)
    return df

  store_args = {'complevel': 5, 'complib': 'zlib'}
  store_args.update(kwdargs)
  with pd.HDFStore(output_path, mode='w', **store_args) as hdf:
    logging.info("Concatenating into DataFrames")
    def add_group(key):
      df = build_df(key, obs_sets)
      hdf.put(key, df)
      # Next we check if there is a corresponding _info field for the
      # key.  If so we make sure it's unique before adding them as
      # attributes.
      info_name = '%s_info' % key
      # read all the information dictionaries and make sure they're unique
      info = np.unique([x.get(info_name, None) for x in obs_sets])
      if info.size > 1:
        logging.warn("Ambiguous attributes found in %s" % info_name)
        if strict:
          raise ValueError("Expected unique attributes in %s" % info_name)
      info = info[0]
      if info is not None:
        # Write attributes
        hdf.get_storer(key).attrs.info = info

    logging.info("Writting to file %s" % output_path)
    # Get all the keys in the full set of obs_sets
    groups = reduce(lambda x, y: x.union(y), [set(x.keys()) for x in obs_sets])
    # the remove any of the info keys, those will be processesed sererately
    groups = [x for x in groups if not x.endswith('_info')]
    groups.pop(groups.index('epoch'))
    # and write to file.
    [add_group(k) for k in groups]


def read_group(hdf5_store, group_name):
  """
  Reads a single group and it's (optional) attributes from an
  HDF5 store and returns them in a format suitable for using
  simulate.simulate_from_iterables.
  """
  def iter_obs():
    for _, one_epoch in hdf5_store[group_name].groupby(level='epoch'):
      old_index = one_epoch.index.names[1]
      one_epoch.reset_index(inplace=True)
      one_epoch.set_index(old_index, inplace=True)
      yield one_epoch

  info = getattr(hdf5_store.get_storer(group_name).attrs, 'info', {})
  return info, iter_obs()
