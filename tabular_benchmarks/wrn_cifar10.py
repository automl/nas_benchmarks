import os
import json
import numpy as np
import ConfigSpace


class WRNCIFAR10Benchmark(object):

    def __init__(self, data_dir="./"):

        cs = self.get_configuration_space()
        self.names = [h.name for h in cs.get_hyperparameters()]

        self.data = json.load(open(os.path.join(data_dir, "wrn_cifar10_data.json"), "rb"))

    def get_best_configuration(self):

        best = None
        curr_valid = np.inf
        for k in self.data.keys():
            if self.data[k][0][-1] < curr_valid:
                curr_valid = self.data[k][0][-1]
                best = k

        best_config = dict()
        for i, n in enumerate(self.names):
            best_config[n] = best[i]

        return best_config, curr_valid

    def objective_function(self, config, budget=None, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        # time_per_epoch = runtime / 100
        #
        # rt = time_per_epoch * budget

        return valid[-1], runtime

    def objective_function_test(self, config, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        return test[-1], runtime

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("stride_1", [1, 2]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("stride_2", [1, 2]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("stride_3", [1, 2]))

        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("n_filters_1", [32, 64]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("n_filters_2", [32, 64]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("n_filters_3", [32, 64]))

        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("num_residual_units_1", [3, 4]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("num_residual_units_2", [3, 4]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("num_residual_units_3", [3, 4]))

        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("initial_lr", [.5e-1, 1e-1]))
        # cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("batch_size", [16, 32]))
        # cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("batch_size", [16]))

        return cs
