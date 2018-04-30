import os
import json
import numpy as np
import ConfigSpace


class WRNCIFAR10Benchmark(object):

    def __init__(self, data_dir="./"):

        cs = self.get_configuration_space()
        self.names = [h.name for h in cs.get_hyperparameters()]

        dic = json.load(open(os.path.join(data_dir, "wrn_cifar10_data.json"), "r"))

        k = dic.keys()
        v = dic.values()
        k1 = [eval(i) for i in k]
        self.data = dict(zip(*[k1, v]))

        self.X = []
        self.y = []
        self.c = []

    def get_best_configuration(self):

        """
        Returns the best configuration in the dataset that achieves the lowest test performance.

        :return: Returns tuple with the best configuration, its final validation performance and its test performance
        """

        best = None
        curr_valid = np.inf
        curr_test = np.inf
        for k in self.data.keys():
            if self.data[k][1][-1] < curr_test:
                curr_valid = self.data[k][0][-1]
                curr_test = self.data[k][1][-1]
                best = k

        best_config = dict()
        for i, n in enumerate(self.names):
            best_config[n] = best[i]

        return best_config, curr_valid, curr_test

    def objective_function(self, config, budget=None, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        # time_per_epoch = runtime / 100
        #
        # rt = time_per_epoch * budget
        self.X.append(config)
        self.y.append(valid[-1])
        self.c.append(runtime)

        return valid[-1], runtime

    def objective_function_test(self, config, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        return test[-1], runtime

    def get_results(self):

        inc, y_star_valid, y_star_test = self.get_best_configuration()

        regret_validation = []
        regret_test = []
        runtime = []
        rt = 0

        inc_valid = np.inf
        inc_test = np.inf

        for i in range(len(self.X)):

            if inc_valid > self.y[i]:
                inc_valid = self.y[i]
                inc_test, _ = self.objective_function_test(self.X[i])

            regret_validation.append(inc_valid - y_star_valid)
            regret_test.append(inc_test - y_star_test)
            rt += self.c[i]
            runtime.append(rt)

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['runtime'] = runtime

        return res

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("stride_1", [1, 2]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("stride_2", [1, 2]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("stride_3", [1, 2]))

        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_filters_1", [32, 64]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_filters_2", [32, 64]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_filters_3", [32, 64]))

        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("num_residual_units_1", [3, 4]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("num_residual_units_2", [3, 4]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("num_residual_units_3", [3, 4]))

        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("initial_lr", [.5e-1, 1e-1]))

        return cs
