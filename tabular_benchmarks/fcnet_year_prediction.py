import os
import json
import numpy as np
import ConfigSpace


class FCNetYearPredictionBenchmark(object):

    def __init__(self, data_dir="./"):

        cs = self.get_configuration_space()
        self.names = [h.name for h in cs.get_hyperparameters()]

        dic = json.load(open(os.path.join(data_dir, "fcnet_year_prediction_data.json"), "r"))

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
            if self.data[k][1] < curr_test:
                curr_valid = self.data[k][0][-1]
                curr_test = self.data[k][1]
                best = k

        best_config = dict()
        for i, n in enumerate(self.names):
            best_config[n] = best[i]

        return best_config, curr_valid, curr_test

    def objective_function(self, config, budget=100, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        time_per_epoch = runtime / 100

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)

        return valid[budget - 1], rt

    def objective_function_test(self, config, **kwargs):
        c = []
        for h in self.names:
            c.append(config[h])

        valid, test, runtime = self.data[tuple(c)]

        return test, runtime

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

        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512], default_value=64))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512], default_value=64))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_1", [0.0, 0.3, 0.6], default_value=0.0))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_2", [0.0, 0.3, 0.6], default_value=0.0))
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"], default_value='relu'))
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"], default_value='relu'))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1],
                                              default_value=1e-3))

#        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const", "exponential"],
#                                                                    default_value='const'))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"],
                                                                    default_value='const'))

#        cs.add_hyperparameter(
#            ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64, 128], default_value=32))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64], default_value=32))

        return cs
