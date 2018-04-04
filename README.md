# Benchmarks for neural architecture search

Requires: TensorFlow 1.6 or later.

To train a WRN on CIFAR-10:

    export PATH_TO_DATASET=...
    export OUTPUT_DIR=...

    bazel run wrn_cifar10:train -- \
      --alsologtostderr \
      --use_estimator_code_path \
      --data_dir ${PATH_TO_DATASET}/cifar-10-batches-py/ \
      --train_dir ${OUTPUT_DIR} \
      --num_epochs 20 \
      --lr_decay cosine \
      --initial_lr 0.1 \
      --nodepthwise \
      --num_residual_units_1 4 \
      --num_residual_units_2 4 \
      --num_residual_units_3 4 \
      --n_filters_1 16 \
      --n_filters_2 32 \
      --n_filters_3 64 \
      --stride_1 1 \
      --stride_2 2 \
      --stride_3 2

Note, you can omit `--use_estimator_code_path` to use the original code path.
Currently, the estimator path doesn't run evaluation metrics.

The results (e.g learning curves) + test / validation predictions will be saved
in `${OUTPUT_DIR}`.

TODOs:

1. Move all code to use `tf.estimator.Estimator` and delete the old code path.
1. Add timestamps to the checkpoints. This should be possible by creating a
variable and using `tf.timestamp` to set it each time the train op is called.
