"""High-level logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import enum
import tensorflow as tf
from typing import Text

from wrn_cifar10 import data  # pylint: disable=g-bad-import-order
from wrn_cifar10 import model  # pylint: disable=g-bad-import-order

logging = tf.logging
gfile = tf.gfile

NUM_TRAIN_EXAMPLES = 45000
NUM_EVAL_EXAMPLES = 5000


def hparams_to_json(hp: tf.contrib.training.HParams) -> Text:
  """Converts HParams to JSON."""
  d = hp.values()

  def sanitize(v):
    if isinstance(v, enum.Enum):
      return v.name
    return v

  sanitized = {k: sanitize(v) for k, v in d.items()}
  return json.dumps(sanitized, indent=2, sort_keys=True)


def train_estimator(
    use_tpu_estimator: bool,
    use_tpu: bool,
    train_dir: Text,
    data_dir: Text,
    hp: tf.contrib.training.HParams,
    master: Text,
    iterations_per_loop: int,
    save_disk_space: bool,
) -> float:
  """Trains the model.

  Args:
    use_tpu_estimator: Whether to use the TPUEstimator as opposed to the
      plain tf.estimator.Estimator.
    use_tpu: Whether to use a TPU.
    train_dir: Where to write checkpoints, etc.
    data_dir: Where the CIFAR-10 data may be found.
    hp: HParams.
    master: URL of the Cloud TPU instance.
    iterations_per_loop: Number of iterations per TPU training loop.
    save_disk_space: If true, non-essential logs are not saved to disk.

  Returns:
    Optimization loss evaluated on the validation dataset.
  """
  batch_size = hp.batch_size

  logging.info('HParams: %r', hp)
  hparams_path = os.path.join(train_dir, 'hparams.json')
  logging.info('Writing HParams to %s', hparams_path)
  with gfile.Open(hparams_path, 'w') as fh:
    fh.write(hparams_to_json(hp))

  if use_tpu_estimator:
    logging.info('Training with TPUEstimator')

    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=train_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop),
        save_checkpoints_steps=NUM_TRAIN_EXAMPLES // batch_size,
        keep_checkpoint_max=None,
    )
    logging.info('model_dir: %r', run_config.model_dir)

    def drop_batch_param(hp: tf.contrib.training.HParams,
                        ) -> tf.contrib.training.HParams:
      d = hp.values()
      d.pop('batch_size')
      return tf.contrib.training.HParams(**d)

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model.make_estimator_model_fn(
            tpu_only_ops=True, use_tpu=use_tpu, make_tpu_estimator_spec=True),
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=50,
        config=run_config,
        params=drop_batch_param(hp))
  else:
    logging.info('Training with Estimator')

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(save_checkpoints_steps=NUM_TRAIN_EXAMPLES //
                                    batch_size)
    # Override the default directory layout.
    run_config = run_config.replace(model_dir=train_dir)
    logging.info('model_dir: %r', run_config.model_dir)

    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model.make_estimator_model_fn(
            tpu_only_ops=False, use_tpu=False, make_tpu_estimator_spec=False),
        params=hp)

  start_time = time.time()

  logging.info('Training')
  num_train_steps = hp.num_epochs * NUM_TRAIN_EXAMPLES // batch_size
  estimator.train(
      input_fn=data.make_estimator_input_fn(
          dataset_dir=data_dir, ds=data.DatasetSplit.TRAIN),
      steps=num_train_steps)

  stop_time = time.time()

  logging.info('Evaluating')
  num_eval_steps = NUM_EVAL_EXAMPLES // batch_size
  eval_metrics = estimator.evaluate(
      input_fn=data.make_estimator_input_fn(
          dataset_dir=data_dir, ds=data.DatasetSplit.EVAL),
      steps=num_eval_steps)

  train_duration = stop_time - start_time
  metadata = {
      'train_start_time':
          start_time,
      'train_stop_time':
          stop_time,
      'train_duration':
          train_duration,
      'eval_loss':
          float(eval_metrics['loss']),
      'examples_per_second':
          NUM_TRAIN_EXAMPLES * hp.num_epochs / train_duration,
  }
  logging.info('metadata: %r', metadata)
  metadata_path = os.path.join(train_dir, 'metadata.json')
  logging.info('Writing training metadata to %s', metadata_path)
  with gfile.Open(metadata_path, 'w') as fh:
    json.dump(metadata, fh, indent=2, sort_keys=True)

  def remove_file(path: Text):
    assert gfile.Exists(path)
    logging.info('Removing: %s', path)
    gfile.Remove(path)

  if save_disk_space:
    # This can be reconstructed from the Python code.
    graph_path = os.path.join(train_dir, 'graph.pbtxt')
    # This can be reconstructed from the Python code and the checkpoints.
    metadata_paths = [
        os.path.join(train_dir, f)
        for f in gfile.Glob(os.path.join(train_dir, '*.meta'))
    ]
    # This is only used for TensorBoard.
    events_paths = [
        os.path.join(train_dir, f)
        for f in gfile.Glob(os.path.join(train_dir, 'events.out.tfevents.*'))
    ]
    for path in [graph_path] + metadata_paths + events_paths:
      remove_file(path)

  return metadata['eval_loss']
