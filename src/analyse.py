import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)
# store results
with open(os.path.join('../run/', 'results.pkl'), 'wb') as fh:

    id2config = fh.get_id2config_mapping()
    incumbent = fh.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])