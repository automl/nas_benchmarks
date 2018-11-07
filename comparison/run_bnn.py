import sys
import numpy as np
import torch
import os
import argparse
import json
import torch.nn as nn
import collections
import time
import random
import ConfigSpace

from scipy.stats import norm
from pybnn.bohamiann import Bohamiann
from functools import partial
from copy import deepcopy

sys.path.append("/home/kleinaa/devel/git/nas_benchmark_github/")

from tabular_benchmarks.fcnet_benchmark import FCNetBenchmark
from tabular_benchmarks.fcnet_year_prediction import FCNetYearPredictionBenchmark


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(config):
    y, cost = b.objective_function(config)
    return 1 - y  # we maximize


def random_architecture():
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    # pick random dimension

    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]

    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    # drop current values from potential choices
    choices.remove(parent_arch[hyper.name])

    # flip hyperparameter
    idx = np.random.randint(len(choices))

    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(acq, cycles, population_size, sample_size):

    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = acq([model.arch.get_array()])
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy = acq([child.arch.get_array()])
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()
    cands_value = [i.accuracy for i in history]
    best = np.argmax(cands_value)
    x_new = history[best].arch
    return x_new


def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-4))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)



def thompson_sampling(candidates, model):
    mu, var, samples = model.predict(candidates, return_individual_predictions=True)
    idx = np.random.randint(samples.shape[0])

    return samples[idx]


def ucb(candidates, model, beta=1):
    mu, var = model.predict(candidates)

    return mu + beta * np.sqrt(var)


def expected_improvement(candidates, model, y_star):
    mu, var = model.predict(candidates)
    s = np.sqrt(var)

    diff = (mu - y_star)
    f = np.max([0, diff]) + s * norm.pdf(diff / (s + 1e-10)) - np.abs(diff) * norm.cdf(diff / (s + 1e-10))

    return f


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--acquisition', default="ts", type=str, nargs='?', help='specifies the acquisition function type')
parser.add_argument('--n_init', default=20, type=int, nargs='?', help='number of data points for the initial design')


args = parser.parse_args()

if args.benchmark == "year_prediction":
    b = FCNetYearPredictionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetBenchmark(dataset=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetBenchmark(dataset=args.data_dir)

cs = b.get_configuration_space()
dim = len(cs.get_hyperparameters())
n_init = args.n_init
num_iters = args.n_iters

assert n_init < num_iters

# initial design

X = np.zeros([n_init, dim])
y = np.zeros([n_init])

for i in range(n_init):
    x_new = cs.sample_configuration()
    y_new = train_and_eval(x_new)
    y[i] = y_new
    X[i] = x_new.get_array()

# BO loop (note we maximize here)

st = time.time()
for i in range(args.n_iters - n_init):

    print("iteration %d : runtime %f" % (i, time.time() - st))
    bnn = Bohamiann(get_network=get_default_network, use_double_precision=False)
    bnn.train(X, y, verbose=False, lr=1e-5, num_burn_in_steps=10000, num_steps=10110)

    if args.acquisition == "ts":
        acquisition = partial(thompson_sampling, model=bnn)
    elif args.acquisition == "ucb":
        acquisition = partial(ucb, model=bnn)
    elif args.acquisition == "ei":
        acquisition = partial(expected_improvement, model=bnn, y_star=np.argmax(y))

    x_new = regularized_evolution(acq=acquisition, cycles=1000, population_size=100, sample_size=10)
    y_new = train_and_eval(x_new)
    print("observed function value %f" % y_new)
    X = np.concatenate((X, [x_new.get_array()]), axis=0)
    y = np.concatenate((y, [y_new]), axis=0)

    print("current best %f" % np.max(y))

output_path = os.path.join(args.output_path, "bnn_%s" % args.acquisition)
os.makedirs(os.path.join(output_path), exist_ok=True)

res = b.get_results()
fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
