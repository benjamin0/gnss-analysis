# gnss-analysis

Tools for analyzing libswiftnav filters on stored GNSS data.

# Setup and install

First, you need to install

 - [libswiftnav](https://github.com/swift-nav/libswiftnav/)
 - [hdf5](https://www.hdfgroup.org/HDF5/)(required for `tables`)

After which you can install gnss-analysis and the rest of its dependencies by running:

```shell
pip install -r ./requirements.txt
pip install -e ./
```

## Virtual Environments

`gnss_analysis` requires `numpy`, `pandas` and `scipy` all
of which need to be specific versions and can cause compliation
nightmares.  You may find it easier to install into a conda
virtual environment,
```shell
conda create -n gnss_analysis numpy==1.9.3 scipy==0.16.0 cython matplotlib pandas==0.16.1
source activate gnss_analysis
```
which will get you most of the way after which you can follow the standard install above.

# Usage

You can create an hdf5 file holding aggregated piksi log
files by running,

```shell
python records2table.py path_to_log.json -o output.hdf5
```

TODO: Migrate to sphinx and automatically include help output.