import os
import sys
import ConfigSpace
import json
import argparse
import logging
logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

sys.path.append("/home/kleinaa/devel/git/nas_benchmark_github/tabular_benchmarks")
from wrn_cifar10 import WRNCIFAR10Benchmark
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

# if args.benchmark == "wrn_cifar10":
#     b = WRNCIFAR103HBenchmark(data_dir=args.data_dir)

if args.benchmark == "fcnet_regression":
    b = FCNetYearPredictionBenchmark(data_dir=args.data_dir)
    min_budget = 4
    max_budget = 100

elif args.benchmark == "protein_structure":
    b = FCNetBenchmark(dataset=args.data_dir)
    min_budget = 4
    max_budget = 100

elif args.benchmark == "slice_localization":
    b = FCNetBenchmark(dataset=args.data_dir)
    min_budget = 4
    max_budget = 100

output_path = os.path.join(args.output_path, "bohb")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()


class MyWorker(Worker):
    def compute(self, config, budget, *args, **kwargs):
        c = ConfigSpace.Configuration(cs, values=config)
        y, cost = b.objective_function(c, budget=int(budget))
        return ({
            'loss': float(y),
            'info': float(cost)})


hb_run_id = '0'

NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

num_workers = 1

workers = []
for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                 run_id=hb_run_id,
                 id=i)
    w.run(background=True)
    workers.append(w)

bohb = BOHB(configspace=cs,
               run_id=hb_run_id,
               eta=3, min_budget=min_budget, max_budget=max_budget,
               nameserver=ns_host,
               nameserver_port=ns_port,
               ping_interval=10)

results = bohb.run(args.n_iters, min_n_workers=num_workers)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
