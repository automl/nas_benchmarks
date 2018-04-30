from fcnet_year_prediction import FCNetYearPredictionBenchmark

b = FCNetYearPredictionBenchmark(data_dir="./data/")

x = b.get_configuration_space().sample_configuration()

b.objective_function(x)

x = {'dropout_1': 0.3, 'batch_size': 32, 'init_lr': 0.0005, 'activation_fn_1': 'relu', 'activation_fn_2': 'relu',
     'lr_schedule': 'cosine', 'dropout_2': 0.0, 'n_units_2': 256, 'n_units_1': 256}

b.objective_function(x)

results = b.get_results()

from IPython import embed

embed()
