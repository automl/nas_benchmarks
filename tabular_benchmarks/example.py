from wrn_cifar10 import WRNCIFAR10Benchmark

DATA_DIR = "./data/"  # path to the json file that contains the tabular data

b = WRNCIFAR10Benchmark(data_dir=DATA_DIR)

# The benchmark expects the configurations encoded as dicts
x = {'n_filters_2': 32, 'stride_1': 2, 'initial_lr': 0.1, 'n_filters_3': 32, 'num_residual_units_1': 4, 'stride_2': 2,
     'stride_3': 2, 'num_residual_units_3': 4, 'n_filters_1': 32, 'num_residual_units_2': 3}
b.objective_function(x)

# After the optimization we can access the results
results = b.get_results()
