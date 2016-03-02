import os
import pytest
import argparse

from gnss_analysis.tools import common, convert

@pytest.fixture(params=['cors', 'piksi'])
def input_observations(request, datadir):
  def get_path(x):
    return datadir.join(x).strpath
  if request.param == 'cors':
    return ('%(rover)s --navigation %(nav)s --base %(base)s'
            % {'rover': get_path('cors_drops_reference/seat032/partial_seat0320.16o'),
               'nav': get_path('cors_drops_reference/seat032/seat0320.16n'),
               'base': get_path('cors_drops_reference/ssho032/partial_ssho0320.16o')
              })
  elif request.param == 'piksi':
    return get_path('partial_serial-link-20151221-142236.log.json')


@pytest.fixture(params=['default', 'partial', 'filter', 'filters', 'add_state'])
def options(request):
  if request.param == 'default':
    return ''
  elif request.param == 'partial':
    return '-n 3'
  elif request.param == 'filter':
    return '-n 3 --filters L1_static'
  elif request.param == 'filters':
    return '-n 3 --filters L1_static L1_dynamic'
  elif request.param == 'add_state':
    return '-n 3 --calc-sat-state'


def test_converts(datadir, input_observations, options):

  parser = argparse.ArgumentParser()
  parser = convert.create_parser(parser)

  full_args = ' '.join([input_observations, options])
  args = parser.parse_args(full_args.split())

  args.output = common.infer_output(args.output, args.input)
  args.filters = common.resolve_filters(args.filter_names)

  convert.convert(args)

  # make sure the output file was created and is nonzero
  assert os.path.exists(args.output)
  assert os.path.getsize(args.output)

