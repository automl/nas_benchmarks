# Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search

This repository contains code of tabular benchmarks for
 - HPOBench: joint hyperparameter and architecture optimization of feed forward neural networks on regression problems (see [1])
 - NASBench101: the architecture optimization of a convolutional neural network (see [2])
 

To download the datasets for the FC-Net benchmark:

    wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
    tar xf fcnet_tabular_benchmarks.tar.gz
    
The data for NASBench is available [here](https://github.com/google-research/nasbench).

To install it, type:

    git clone https://github.com/automl/nas_benchmarks.git
    cd nas_benchmarks
    python setup.py install
   
 The following example shows how to load the benchmark and to evaluate a random hyperparameter configuration:
  
    from tabular_benchmarks import FCNetProteinStructureBenchmark

    b = FCNetProteinStructureBenchmark(data_dir="./fcnet_tabular_benchmarks/")
    cs = b.get_configuration_space()
    config = cs.sample_configuration()

    print("Numpy representation: ", config.get_array())
    print("Dict representation: ", config.get_dictionary())

    max_epochs = 100
    y, cost = b.objective_function(config, budget=max_epochs)
    print(y, cost)
    
    
To see how you can run different open-source optimizers from the literature, have a look on the python scripts in 'experiment_scripts' folder, which were also used to conducted the experiments in the papers.


# References

    [1] Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization
        A. Klein and F. Hutter
        arXiv:1905.04970 [cs.LG]
    
    [2] NAS-Bench-101: Towards Reproducible Neural Architecture Search
        C. Ying and A. Klein and E. Real and E. Christiansen and K. Murphy and F. Hutter
        arXiv:1902.09635 [cs.LG]
