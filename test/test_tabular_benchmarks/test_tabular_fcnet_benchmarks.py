import unittest

import numpy as np
from tabular_benchmarks import FCNetYearPredictionBenchmark, FCNetProteinStructureBenchmark


class TestFCNetYearPrediction(unittest.TestCase):

    def setUp(self):
        self.b = FCNetYearPredictionBenchmark(data_dir="/home/kleinaa/datasets/fcnet_tabular_benchmarks/")

    def test_get_best_configuration(self):
        c, v, t = self.b.get_best_configuration()
        best_te = 6.0535393

        assert np.isclose(t, best_te, atol=1e-6)
        best_config = {'dropout_1': 0.3, 'dropout_2': 0.0, 'batch_size': 32, 'activation_fn_1': 'relu',
                       'activation_fn_2': 'relu',
                       'lr_schedule': 'cosine', 'init_lr': 0.0005, 'n_units_1': 256, 'n_units_2': 256}

        assert c == best_config

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestFCNetProteinStructure(unittest.TestCase):

    def setUp(self):
        self.b = FCNetProteinStructureBenchmark(data_dir="/home/kleinaa/datasets/fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestFCNetSliceLocalization(unittest.TestCase):

    def setUp(self):
        self.b = FCNetProteinStructureBenchmark(data_dir="/home/kleinaa/datasets/fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


if __name__ == '__main__':
    unittest.main()
