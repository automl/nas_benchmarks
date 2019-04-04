import unittest

from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C


class TestNASCifar10A(unittest.TestCase):

    def setUp(self):
        self.b = NASCifar10A(data_dir="./")

    def test_fix_configuration(self):
        cs = self.b.get_configuration_space()
        config = cs.sample_configuration()
        # inception architecture
        config["op_node_0"] = 'conv1x1-bn-relu'
        config["op_node_1"] = 'conv3x3-bn-relu'
        config["op_node_2"] = 'conv3x3-bn-relu'
        config["op_node_3"] = 'conv3x3-bn-relu'
        config["op_node_4"] = 'maxpool3x3'

        for i in range(21):
            config["edge_%d" % i] = 0

        config["edge_0"] = 1
        config["edge_1"] = 1
        config["edge_2"] = 1
        config["edge_4"] = 1
        config["edge_10"] = 1
        config["edge_14"] = 1
        config["edge_15"] = 1
        config["edge_19"] = 1
        config["edge_20"] = 1

        max_epochs = 108
        y, cost = self.b.objective_function(config, max_epochs)

        mean_test_error = self.b.y_star_test + self.b.get_results()['regret_test'][0]
        mean_test_acc = 1 - mean_test_error

        assert mean_test_acc == 0.9308560291926066

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestNASCifar10B(unittest.TestCase):

    def setUp(self):
        self.b = NASCifar10B(data_dir="./")

    def test_fix_configuration(self):
        cs = self.b.get_configuration_space()
        config = cs.sample_configuration()
        # inception architecture
        config["op_node_0"] = 'conv1x1-bn-relu'
        config["op_node_1"] = 'conv3x3-bn-relu'
        config["op_node_2"] = 'conv3x3-bn-relu'
        config["op_node_3"] = 'conv3x3-bn-relu'
        config["op_node_4"] = 'maxpool3x3'

        config["edge_0"] = 0
        config["edge_1"] = 1
        config["edge_2"] = 2
        config["edge_3"] = 4
        config["edge_4"] = 10
        config["edge_5"] = 14
        config["edge_6"] = 15
        config["edge_7"] = 19
        config["edge_8"] = 20

        max_epochs = 108
        y, cost = self.b.objective_function(config, max_epochs)

        mean_test_error = self.b.y_star_test + self.b.get_results()['regret_test'][0]
        mean_test_acc = 1 - mean_test_error

        assert mean_test_acc == 0.9308560291926066

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


class TestNASCifar10C(unittest.TestCase):

    def setUp(self):
        self.b = NASCifar10C(data_dir="./")

    def test_fix_configuration(self):
        cs = self.b.get_configuration_space()
        config = cs.sample_configuration()
        # inception architecture
        config["op_node_0"] = 'conv1x1-bn-relu'
        config["op_node_1"] = 'conv3x3-bn-relu'
        config["op_node_2"] = 'conv3x3-bn-relu'
        config["op_node_3"] = 'conv3x3-bn-relu'
        config["op_node_4"] = 'maxpool3x3'

        from tabular_benchmarks.nas_cifar10 import VERTICES
        for i in range(VERTICES * (VERTICES - 1) // 2):
            config["edge_%d" % i] = 0

        config["edge_0"] = 1
        config["edge_1"] = 1
        config["edge_2"] = 1
        config["edge_4"] = 1
        config["edge_10"] = 1
        config["edge_14"] = 1
        config["edge_15"] = 1
        config["edge_19"] = 1
        config["edge_20"] = 1
        config["num_edges"] = 9

        max_epochs = 108

        y, cost = self.b.objective_function(config, max_epochs)

        mean_test_error = self.b.y_star_test + self.b.get_results()['regret_test'][0]
        mean_test_acc = 1 - mean_test_error

        assert mean_test_acc == 0.9308560291926066

    def test_random_sampling(self):
        config = self.b.get_configuration_space().sample_configuration()
        self.b.objective_function(config)


if __name__ == '__main__':
    unittest.main()
