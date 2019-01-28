
from tabular_benchmarks import FCNetProteinStructureBenchmark

b = FCNetProteinStructureBenchmark(data_dir="./fcnet_tabular_benchmarks/")
cs = b.get_configuration_space()
config = cs.sample_configuration()

print("Numpy representation: ", config.get_array())
print("Dict representation: ", config.get_dictionary())

max_epochs = 100
y, cost = b.objective_function(config, budget=max_epochs)
print(y, cost)


