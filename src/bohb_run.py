import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)



parser = argparse.ArgumentParser(description='Automl Final Project')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=10)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=20)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=10)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='../run/')
parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.')

args=parser.parse_args()


from pytorch_worker import PyTorchWorker as worker



# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


if args.worker:
	import time
	time.sleep(5)	# short artificial delay to make sure the nameserver is already running
	w = worker(run_id=args.run_id, host=host, timeout=120)
	w.load_nameserver_credentials(working_directory=args.shared_directory)
	w.run(background=False)
	exit(0)


# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# Let us load the old run now to use its results to warmstart a new run with slightly
# different budgets in terms of datapoints and epochs.
# Note that the search space has to be identical though!
if args.previous_run_dir is not None:
	previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)

	# Run an optimizer
	bohb = BOHB(configspace = worker.get_configspace(),
				run_id = args.run_id,
				host=host,
				nameserver=ns_host,
				nameserver_port=ns_port,
				result_logger=result_logger,
				min_budget=args.min_budget, max_budget=args.max_budget,
				previous_result = previous_run) #uncomment for warmstart
else:
	# Run an optimizer
	bohb = BOHB( configspace=worker.get_configspace(),
				 run_id=args.run_id,
				 host=host,
				 nameserver=ns_host,
				 nameserver_port=ns_port,
				 result_logger=result_logger,
				 min_budget=args.min_budget, max_budget=args.max_budget)  # uncomment for warmstart

res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))




