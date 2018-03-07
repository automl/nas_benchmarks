# Benchmarks for neural architecture search

To train a WRN on CIFAR-10:

    python train.py --data_dir PATH_TO_DATASET/cifar-10-batches-py/ --train_dir OUTPUT_DIR --num_epochs 20 --lr_decay cosine --initial_lr 0.01 --depthwise 0 --num_residual_units_1 4 --num_residual_units_2 4 --num_residual_units_3 4 --n_filters_1 16 --n_filters_2 32 --n_filters_3 64 --stride_1 1 --stride_2 2 --stride_3 2
    
The results (e.g learning curves) + test / validation predictions will be saved in OUTPUT_DIR.
