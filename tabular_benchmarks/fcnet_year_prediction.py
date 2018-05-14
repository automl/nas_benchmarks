import os
import json
import h5py
import numpy as np
import ConfigSpace


class FCNetYearPredictionBenchmark(object):

    def __init__(self, data_dir="./", seed=None):

        cs = self.get_configuration_space()
        self.names = [h.name for h in cs.get_hyperparameters()]

        self.data = h5py.File(os.path.join(data_dir, "fcnet_year_prediction_data.hdf5"), "r")

        self.X = []
        self.y = []
        self.c = []

        self.rng = np.random.RandomState(seed)

    def get_best_configuration(self):

        """
        Returns the best configuration in the dataset that achieves the lowest test performance.

        :return: Returns tuple with the best configuration, its final validation performance and its test performance
        """

        configs, te, ve = [], [], []
        for k in self.data.keys():
            configs.append(json.loads(k))
            te.append(np.mean(self.data[k]["final_test_error"]))
            ve.append(np.mean(self.data[k]["valid_mae"][:, -1]))

        b = np.argmin(te)

        return configs[b], ve[b], te[b]

    def objective_function(self, config, budget=100, **kwargs):

        i = self.rng.randint(4)

        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        valid = self.data[k]["valid_mae"][i]
        runtime = self.data[k]["runtime"][i]

        time_per_epoch = runtime / 100

        rt = time_per_epoch * budget

        self.X.append(config)
        self.y.append(valid[budget - 1])
        self.c.append(rt)

        return valid[budget - 1], rt

    def objective_function_test(self, config, **kwargs):
        i = self.rng.randint(4)
        if type(config) == ConfigSpace.Configuration:
            k = json.dumps(config.get_dictionary(), sort_keys=True)
        else:
            k = json.dumps(config, sort_keys=True)

        test = self.data[k]["final_test_error"][i]
        runtime = self.data[k]["runtime"][i]

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

            regret_validation.append(float(inc_valid - y_star_valid))
            regret_test.append(float(inc_test - y_star_test))
            rt += self.c[i]
            runtime.append(float(rt))

        res = dict()
        res['regret_validation'] = regret_validation
        res['regret_test'] = regret_test
        res['runtime'] = runtime

        return res

    @staticmethod
    def get_configuration_space():
        cs = ConfigSpace.ConfigurationSpace()

        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_1", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_2", [0.0, 0.3, 0.6]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
        cs.add_hyperparameter(
            ConfigSpace.OrdinalHyperparameter("init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64]))
        return cs
