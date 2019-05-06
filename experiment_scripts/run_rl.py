"""
REINFORCE baseline

The code for the reinforcement learning agent was generously provided by Chris Ying.
For the original implementation see here:

https://github.com/google-research/nasbench

For a more detailed description see the following paper:

Ying, C., Klein, A., Real, E., Christiansen, E., Murphy, K., and Hutter, F.
NAS-Bench-101: Towards reproducible neural architecture search.
arXiv:1902.09635 [cs.LG].


"""
import os
import argparse
import json
import ConfigSpace
import tensorflow as tf
import tensorflow_probability as tfp

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._denominator = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._momentum = momentum

    def update(self, value):
        """Update the moving average with a new sample."""
        self._numerator.assign(
            self._momentum * self._numerator + (1 - self._momentum) * value)
        self._denominator.assign(
            self._momentum * self._denominator + (1 - self._momentum))

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


class Reward(object):
    """Computes the fitness of a sampled model by querying NASBench."""

    def __init__(self, bench):
        self.bench = bench

    def compute_reward(self, sample):
        config = ConfigSpace.Configuration(b.get_configuration_space(), vector=sample)
        y, c = self.bench.objective_function(config)
        fitness = 1 - float(y)
        return fitness


class REINFORCEOptimizer(object):
    """Class that optimizes a set of categorical variables using REINFORCE."""

    def __init__(self, reward, cat_variables, momentum):
        # self._num_vertices = reward.num_vertices
        # self._num_operations = len(reward.available_ops)
        # self._num_edges = (self._num_vertices * (self._num_vertices - 1)) // 2
        #
        # self._edge_logits = tf.Variable(tf.zeros([self._num_edges, 2]))
        # self._op_logits = tf.Variable(tf.zeros([self._num_vertices - 2,
        #                                         self._num_operations]))
        self._num_variables = len(cat_variables)
        self._logits = [tf.Variable(tf.zeros([1, ci])) for ci in cat_variables]
        self._baseline = ExponentialMovingAverage(momentum=momentum)
        self._reward = reward
        self._last_reward = 0.0
        self._test_acc = 0.0

    def step(self):
        """Helper function for a single step of the REINFORCE algorithm."""
        # Obtain a single sample from the current distribution.
        # edge_dist = tfp.distributions.Categorical(logits=self._edge_logits)
        # op_dist = tfp.distributions.Categorical(logits=self._op_logits)
        dists = [tfp.distributions.Categorical(logits=li) for li in self._logits]
        attempts = 0
        while True:
            sample = [di.sample() for di in dists]

            # Compute the sample reward. Larger rewards are better.
            reward = self._reward.compute_reward(sample)
            attempts += 1
            if reward > 0.001:
                # print('num attempts: {}, reward: {}'.format(str(attempts), reward))
                break

        self._last_reward = reward

        # Compute the log-likelihood the sample.
        log_prob = tf.reduce_sum([dists[i].log_prob(sample[i]) for i in range(len(sample))])
        # log_prob = (tf.reduce_sum(edge_dist.log_prob(edge_sample)) +
        #             tf.reduce_sum(op_dist.log_prob(op_sample)))

        # Update the baseline to reflect the current sample.
        self._baseline.update(reward)

        # Compute the advantage. This will be positive if the current sample is
        # better than average, and will be negative otherwise.
        advantage = reward - self._baseline.value()

        # Here comes the REINFORCE magic. We'll update the gradients by
        # differentiating with respect to this value. In practice, if advantage > 0
        # then the update will increase the log-probability, and if advantage < 0
        # then the update will decrease the log-probability.
        objective = tf.stop_gradient(advantage) * log_prob

        return objective

    def trainable_variables(self):
        # Return a list of trainable variables to update with gradient descent.
        # return [self._edge_logits, self._op_logits]
        return self._logits

    def baseline(self):
        """Return an exponential moving average of recent reward values."""
        return self._baseline.value()

    def last_reward(self):
        """Returns the last reward earned."""
        return self._last_reward

    def test_acc(self):
        """Returns the last test accuracy computed."""
        return self._test_acc

    def probabilities(self):
        """Return a set of probabilities for each learned variable."""
        # return [tf.nn.softmax(self._edge_logits),
        #        tf.nn.softmax(self._op_logits)]
        # return tf.nn.softmax(self._op_logits)  # More interesting to look at ops
        return [tf.nn.softmax(li).numpy() for li in self._logits]


def run_reinforce(optimizer, learning_rate, max_time, bench, num_steps, log_every_n_steps=1000):
    """Run multiple steps of REINFORCE to optimize a fixed reward function."""
    trainable_variables = optimizer.trainable_variables()
    trace = []
    # run = [[0.0, 0.0, 0.0]]

    # step = 0
    for step in range(num_steps):
        # step += 1
        # Compute the gradient of the sample's log-probability w.r.t. the logits.
        with tf.GradientTape() as tape:
            objective = optimizer.step()

        # Update the logits using gradient ascent.
        gradients = tape.gradient(objective, trainable_variables)
        for grad, var in zip(gradients, trainable_variables):
            var.assign_add(learning_rate * grad)

        trace.append(optimizer.probabilities())
        # run.append([nasbench.training_time_spent,
        #             optimizer.last_reward(),  # validation acc
        #             optimizer.test_acc()])  # test acc (avg)
        if step % log_every_n_steps == 0:
            print('step = {:d}, baseline reward = {:.5f}'.format(
                step, optimizer.baseline().numpy()))
        # if nasbench.training_time_spent > max_time:
        #     break

    return trace


def scan_top_valid(data):
    """Selects top validation model but records the test accuracy."""
    new_data = []
    for run in data:
        new_run = [[0.0, 0.0, 0.0]]
        for r in run[1:]:
            if r[1] > new_run[-1][1]:
                new_run.append(r)
            else:
                new_run.append([r[0], new_run[-1][1], new_run[-1][2]])
        new_data.append(new_run)

    return new_data


parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--lr', default=1e-1, type=float, nargs='?', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, nargs='?',
                    help='momentum to compute the exponential averaging of the reward')

args = parser.parse_args()


if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)

output_path = os.path.join(args.output_path, "rl")
os.makedirs(os.path.join(output_path), exist_ok=True)

# Eager mode used for RL baseline
tf.enable_eager_execution()
tf.enable_resource_variables()

nb_reward = Reward(b)

cat_variables = []
cs = b.get_configuration_space()
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        cat_variables.append(len(h.sequence))
    elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
        cat_variables.append(len(h.choices))

optimizer = REINFORCEOptimizer(reward=nb_reward, cat_variables=cat_variables, momentum=args.momentum)
trace = run_reinforce(
        optimizer=optimizer,
        learning_rate=args.lr,
        max_time=5e6,
        bench=b,
        num_steps=args.n_iters,
        log_every_n_steps=100)

if args.benchmark == "nas_cifar10a" or args.benchmark == "nas_cifar10b" or args.benchmark == "nas_cifar10c":
    res = b.get_results(ignore_invalid_configs=True)
else:
    res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()

