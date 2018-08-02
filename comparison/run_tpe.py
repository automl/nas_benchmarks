import os
from copy import deepcopy
import json
import ConfigSpace
import argparse

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

args = parser.parse_args()

if args.benchmark == "wrn_cifar10":
    b = WRNCIFAR103HBenchmark(data_dir=args.data_dir)

elif args.benchmark == "fcnet_regression":
    b = FCNetYearPredictionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetBenchmark(dataset=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetBenchmark(dataset=args.data_dir)

output_path = os.path.join(args.output_path, "tpe")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

space = {}
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q=1)
    else:
        space[h.name] = hp.choice(h.name, h.choices)


def objective(x):
    config = deepcopy(x)
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            
            config[h.name] = h.sequence[int(x[h.name])]

    y, c = b.objective_function(config)

    return {
        'config': config,
        'loss': y,
        'cost': c,
        'status': STATUS_OK}


trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.n_iters,
            trials=trials)

res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
