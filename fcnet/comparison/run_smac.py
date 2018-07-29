import os
import sys
import json
import argparse

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
sys.path.append("/home/kleinaa/devel/git/nas_benchmark_github/tabular_benchmarks")
from wrn_cifar10_3h import WRNCIFAR103HBenchmark
from fcnet_year_prediction import FCNetYearPredictionBenchmark
from fcnet_benchmark import FCNetBenchmark


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="wrn_cifar10", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')

args = vars(parser.parse_args())

if args.benchmark == "wrn_cifar10":
    b = WRNCIFAR103HBenchmark(data_dir=args.data_dir)

elif args.benchmark == "fcnet_regression":
    b = FCNetYearPredictionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetBenchmark(dataset=args.data_dir)

output_path = args.output_path
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()


scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "true",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})


def objective_function(config, **kwargs):
    y, c = b.objective_function(config)
    return float(y)


tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
smac = SMAC(scenario=scenario, tae_runner=tae)
smac.optimize()

res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
