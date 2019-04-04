import unittest

from tabular_benchmarks import FCNetProteinStructureBenchmark,\
    FCNetSliceLocalizationBenchmark, FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark


class TestFCNetProteinStructure(unittest.TestCase):

    def setUp(self):
        self.b = FCNetProteinStructureBenchmark(data_dir="./fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestFCNetSliceLocalization(unittest.TestCase):

    def setUp(self):
        self.b = FCNetSliceLocalizationBenchmark(data_dir="./fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestFCNetNavalPropulsion(unittest.TestCase):

    def setUp(self):
        self.b = FCNetNavalPropulsionBenchmark(data_dir="./fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestFCNetParkinsonsTelemonitoring(unittest.TestCase):

    def setUp(self):
        self.b = FCNetParkinsonsTelemonitoringBenchmark(data_dir="./fcnet_tabular_benchmarks/")

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


if __name__ == '__main__':
    unittest.main()
