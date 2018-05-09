"""Data ingestion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

import enum
import numpy as np
import tensorflow as tf

from typing import Any, Dict, Generator, Text, Tuple, Union

logging = tf.logging
gfile = tf.gfile

IMAGE_SIZE = 32

# The number of batches to prefetch at the end of the tf.data pipeline.
_PREFETCH_NUM_BATCHES = 1 << 4


def is_python_3() -> bool:
  return sys.version[0] == '3'


def unpickle(filename: Text) -> Dict:
  logging.info('Loading dataset from %r', filename)
  with gfile.Open(filename, 'rb') as f:
    if is_python_3():
      # pylint: disable=unexpected-keyword-arg
      # pytype: disable=wrong-keyword-args
      return pickle.load(f, encoding='latin1')
      # pylint: enable=unexpected-keyword-arg
      # pytype: enable=wrong-keyword-args
    else:
      return pickle.load(f)


def load_data(dataset_dir: Text) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
  """Loads CIFAR-10 data from disk."""
  xs = []
  ys = []
  for j in range(5):
    d = unpickle(os.path.join(dataset_dir, 'data_batch_%d' % (j + 1)))
    x = d['data']
    y = d['labels']
    xs.append(x)
    ys.append(y)

  d = unpickle(os.path.join(dataset_dir, 'test_batch'))
  xs.append(d['data'])
  ys.append(d['labels'])

  x = np.concatenate(xs) / np.float32(255)
  y = np.concatenate(ys)
  x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
  x = x.reshape((x.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
  X_train = x[0:50000, :, :, :]  # pylint: disable=invalid-name
  y_train = y[0:50000]

  X_test = x[50000:, :, :, :]  # pylint: disable=invalid-name
  y_test = y[50000:]

  # Subtract per-pixel mean.
  pixel_mean = np.mean(X_train, axis=0)
  X_train -= pixel_mean  # pylint: disable=invalid-name
  X_test -= pixel_mean  # pylint: disable=invalid-name

  # Split up additional validation set.
  X_valid = X_train[45000:]  # pylint: disable=invalid-name
  y_valid = y_train[45000:]

  X_train = X_train[:45000]  # pylint: disable=invalid-name
  y_train = y_train[:45000]

  logging.info('X_train type: %r', X_train.dtype)
  logging.info('X_train shape: %r', X_train.shape)
  logging.info('y_train type: %r', y_train.dtype)
  logging.info('y_train shape: %r', y_train.shape)
  logging.info('%d train samples', X_train.shape[0])
  logging.info('%d valid samples', X_valid.shape[0])
  logging.info('%d test samples', X_test.shape[0])

  return X_train, y_train, X_valid, y_valid, X_test, y_test


# We can't use py_funcs, or their equivalents, with TPUs.
# So after the data is loaded everything should be TF ops.
def augment_data(input_op, target_op):
  original_shape = input_op.shape
  logging.info('original shape: %r', original_shape)
  input_op = tf.pad(input_op, ((4, 4), (4, 4), (0, 0)))
  input_op = tf.random_crop(input_op, original_shape)
  input_op = tf.image.random_flip_left_right(input_op)
  return input_op, target_op


def iterate_minibatches_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
  """Optionally augment, shuffle, and batch the provided data."""
  if shuffle:
    permutation = np.random.permutation(range(len(inputs)))
    inputs = inputs[permutation, ...]
    targets = targets[permutation]

  dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).repeat()

  if augment:
    dataset = dataset.map(augment_data, num_parallel_calls=16)

  return dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


def iterate_minibatches(
    sess: tf.Session,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
) -> Generator[Tuple[np.ndarray, np.ndarray], Any, Any]:
  """Creates a Python generator for iterating through the provided data.

  Uses tf.data under the hood so code can be shared with the TPU pipeline.

  Args:
    sess: A tf.Session.
    inputs: Array of input examples, with examples indexed by the first
      dimension.
    targets: Array of target examples, with examples indexed by the first
      dimension.
    batch_size: The size of the sampled minibatches.
    shuffle: Whether to shuffle the data.
    augment: Whether to augment the data.

  Yields:
    A generator of (input, target) minibatches.
  """
  dataset = iterate_minibatches_dataset(
      inputs=inputs,
      targets=targets,
      batch_size=batch_size,
      shuffle=shuffle,
      augment=augment)
  input_op, target_op = dataset.prefetch(
      _PREFETCH_NUM_BATCHES).make_one_shot_iterator().get_next()

  # Iterate once through the dataset.
  for _ in range(inputs.shape[0] // batch_size):
    input_array, target_array = sess.run([input_op, target_op])
    yield input_array, target_array


class DatasetSplit(enum.Enum):
  """Indicates a dataset subset."""
  TRAIN = 'TRAIN'
  EVAL = 'EVAL'
  TEST = 'TEST'


class EstimatorMode(enum.Enum):
  """What the Estimator will do."""
  TRAIN = 'TRAIN'
  EVAL = 'EVAL'


def make_estimator_input_fn(dataset_dir: Text, ds: DatasetSplit,
                            em: EstimatorMode, return_dataset: bool):
  """Creates an input_fn for tf.estimator.Estimator.

  Args:
    dataset_dir: The location of the CIFAR-10 dataset.
    ds: DatasetSplit
    em: EstimatorMode
    return_dataset: Whether the created function should return a Dataset, as
      opposed to Tensors.
      Used if running on TPUs with `per_host_input_for_training=
        tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2`.
      NOTE(ericmc), it's unlikely we'll use this option, so consider removing
        it.

  Returns:
    An input function.
  """

  def input_fn(params: tf.contrib.training.HParams
              ) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.data.Dataset]:
    """input_fn for tf.estimator.Estimator."""
    # pylint: disable=invalid-name
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(dataset_dir)
    # pylint: enable=invalid-name

    if ds is DatasetSplit.TRAIN:
      logging.info('Loading TRAIN split.')
      input_op = X_train
      target_op = y_train
    elif ds is DatasetSplit.EVAL:
      logging.info('Loading EVAL split.')
      input_op = X_valid
      target_op = y_valid
    else:
      logging.info('Loading TEST split.')
      input_op = X_test
      target_op = y_test

    # TPUs don't play well with tf.int64.
    target_op = target_op.astype(np.int32)

    dataset = iterate_minibatches_dataset(
        inputs=input_op,
        targets=target_op,
        batch_size=params.batch_size,
        shuffle=em is EstimatorMode.TRAIN,
        augment=em is EstimatorMode.TRAIN)

    dataset = dataset.prefetch(_PREFETCH_NUM_BATCHES)

    if return_dataset:
      return dataset
    else:
      input_op, target_op = dataset.make_one_shot_iterator().get_next()

      logging.info('input_fn output input_op: %r', input_op)
      logging.info('input_fn output target_op: %r', target_op)

      return input_op, target_op

  return input_fn
