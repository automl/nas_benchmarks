
from tabular_benchmarks.fcnet_benchmark import FCNetBenchmark

b = FCNetBenchmark(dataset="./fcnet_protein_structure_data.hdf5")
cs = b.get_configuration_space()
config = cs.sample_configuration()

print("Numpy representation: ", config.get_array())
print("Dict representation: ", config.get_dictionary())

max_epochs = 100
y, cost = b.objective_function(config, budget=max_epochs)
print(y, cost)


