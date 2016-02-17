#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
  import sys
  reload(sys).setdefaultencoding("UTF-8")
except:
  pass

try:
  from setuptools import setup, find_packages
except ImportError:
  print 'Please install or upgrade setuptools or pip to continue.'
  sys.exit(1)


INSTALL_REQUIRES = ['cython',
                    # These specific versions are required for
                    # a python environment that cooperates with
                    # libswiftnav / sbp_log_analysis etc, but
                    # aren't explicitly required by gnss-analysis
                    'numpy==1.9.3',
                    'pandas==0.16.1',
                    'scipy==0.16.0',
                    'tables',
                    'matplotlib',
                    'sbp',
                    'scikits.statsmodels',
                    # this will have to be installed by either
                    # running `pip install -r requirements.txt`,
                    # which will grab the latest version of libswiftnav
                    # or can be installed from a local clone of
                    # libswiftnav.
                    'swiftnav',
                    'progressbar'
                    ]
TEST_REQUIRES = ['pytest', 'mock']


setup(name='gnss_analysis',
      description='Analysis and Testing of libswiftnav RTK filters',
      version='0.24',
      author='Swift Navigation',
      author_email='dev@swiftnav.com',
      maintainer='Swift Navigation',
      maintainer_email='dev@swiftnav.com',
      url='https://github.com/swift-nav/gnss-analysis',
      keywords='',
      classifiers=['Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Programming Language :: Python :: 2.7'
                   ],
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      tests_require=TEST_REQUIRES,
      platforms="Linux,Windows,Mac",
      use_2to3=False,
      zip_safe=False)
