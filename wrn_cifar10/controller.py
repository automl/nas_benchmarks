"""High-level logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import enum
import tensorflow as tf
from typing import Any, Dict, Text, Union

from wrn_cifar10 import data  # pylint: disable=g-bad-import-order
from wrn_cifar10 import model  # pylint: disable=g-bad-import-order

logging = tf.logging
gfile = tf.gfile

NUM_TRAIN_EXAMPLES = 45000
NUM_EVAL_EXAMPLES = 5000

# These should:
# 1) Be large enough to be efficient.
# 2) Be small enough to fit in memory on TPUs for all models.
# 3) Evenly divide the number of train and eval examples.
EVAL_BATCH_SIZE = 50
EVAL_ITERATIONS_PER_LOOP = 100


def _hparams_path(train_dir: Text) -> Text:
  return os.path.join(train_dir, 'hparams.json')


def hparams_to_json(hp: tf.contrib.training.HParams) -> Text:
  """Converts HParams to JSON."""
  d = hp.values()

  def sanitize(v):
    if isinstance(v, enum.Enum):
      return v.name
    return v

  sanitized = {k: sanitize(v) for k, v in d.items()}
  return json.dumps(sanitized, indent=2, sort_keys=True)


def update_hparams(
    hp: tf.contrib.training.HParams) -> tf.contrib.training.HParams:
  """Converts a Vizier object to a valid HParams."""
  logging.info('Updating HParams from: %r', hp)

  d = hp.values()

  # Add implicit hyperparameter.
  if 'decay_steps' not in d:
    d['decay_steps'] = int(
        hp.num_epochs * (NUM_TRAIN_EXAMPLES // hp.batch_size))

  def to_bool(value: Union[bool, Text]) -> bool:
    if isinstance(value, bool):
      return value
    return {'False': False, 'True': True}[value]

  # Use booleans.
  d['use_nesterov'] = to_bool(hp.use_nesterov)
  d['depthwise'] = to_bool(hp.depthwise)

  # Use enum types.
  d['optimizer'] = model.Optimizer[hp.optimizer]
  d['lr_decay'] = model.LRDecaySchedule[hp.lr_decay]
  d['activation_1'] = model.Activation[hp.activation_1]
  d['activation_2'] = model.Activation[hp.activation_2]
  d['activation_3'] = model.Activation[hp.activation_3]

  # For some reason floats are being cast to ints.
  d['weight_decay'] = float(hp.weight_decay)
  d['dropout_1'] = float(hp.dropout_1)
  d['dropout_2'] = float(hp.dropout_2)
  d['dropout_3'] = float(hp.dropout_3)

  hp = tf.contrib.training.HParams(**d)

  return model.validate(hp)


def json_to_hparams(s: Text) -> tf.contrib.training.HParams:
  """Converts JSON to HParams."""
  d = json.loads(s)
  hp = tf.contrib.training.HParams(**d)

  return update_hparams(hp)


def _get_estimator(
    use_tpu_estimator: bool,
    use_tpu: bool,
    train_dir: Text,
    hp: tf.contrib.training.HParams,
    master: Text,
    iterations_per_loop: int,
    use_model_parallelism: bool,
) -> Union[tf.estimator.Estimator, tf.contrib.tpu.TPUEstimator]:
  batch_size = hp.batch_size
  if use_tpu_estimator:
    logging.info('Training with TPUEstimator')

    if use_model_parallelism:
      tpu_config = tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop, computation_shape=(1, 1, 2))
    else:
      tpu_config = tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop)

    run_config = tf.contrib.tpu.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=train_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config,
        save_checkpoints_steps=NUM_TRAIN_EXAMPLES // batch_size,
        keep_checkpoint_max=None,
    )
    logging.info('model_dir: %r', run_config.model_dir)

    def drop_batch_param(hp: tf.contrib.training.HParams,
                        ) -> tf.contrib.training.HParams:
      d = hp.values()
      d.pop('batch_size')
      return tf.contrib.training.HParams(**d)

    return tf.contrib.tpu.TPUEstimator(
        model_fn=model.make_estimator_model_fn(
            tpu_only_ops=True, use_tpu=use_tpu, make_tpu_estimator_spec=True),
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=EVAL_BATCH_SIZE,
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

    return tf.estimator.Estimator(
        config=run_config,
        model_fn=model.make_estimator_model_fn(
            tpu_only_ops=False, use_tpu=False, make_tpu_estimator_spec=False),
        params=hp)


def _clean_up_files(train_dir: Text):
  """Deletes unneeded files."""

  def remove_file(path: Text):
    if gfile.Exists(path):
      logging.info('Removing: %s', path)
      gfile.Remove(path)

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

  eval_dir = os.path.join(train_dir, 'eval')
  if gfile.Exists(eval_dir):
    gfile.DeleteRecursively(eval_dir)


def train_estimator(
    use_tpu_estimator: bool,
    use_tpu: bool,
    train_dir: Text,
    data_dir: Text,
    hp: tf.contrib.training.HParams,
    master: Text,
    iterations_per_loop: int,
    save_disk_space: bool,
    use_model_parallelism: bool,
):
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
    use_model_parallelism: Whether to use model parallelism.
      This will make batch norm behave as it does on CPUs and GPUs, and provide
      twice the memory for each model.
      However, it will likely halve TPU efficiency.
  """
  batch_size = hp.batch_size

  logging.info('HParams: %r', hp)
  hparams_path = _hparams_path(train_dir)
  logging.info('Writing HParams to %s', hparams_path)
  with gfile.Open(hparams_path, 'w') as fh:
    fh.write(hparams_to_json(hp))

  estimator = _get_estimator(
      use_tpu_estimator=use_tpu_estimator,
      use_tpu=use_tpu,
      train_dir=train_dir,
      hp=hp,
      master=master,
      iterations_per_loop=iterations_per_loop,
      use_model_parallelism=use_model_parallelism)

  start_time = time.time()

  logging.info('Training')
  num_train_steps = hp.num_epochs * NUM_TRAIN_EXAMPLES // batch_size
  estimator.train(
      input_fn=data.make_estimator_input_fn(
          dataset_dir=data_dir,
          ds=data.DatasetSplit.TRAIN,
          return_dataset=False),
      steps=num_train_steps)

  stop_time = time.time()

  duration = stop_time - start_time
  metadata = {
      'start_time': start_time,
      'stop_time': stop_time,
      'duration': duration,
      'examples_per_second': NUM_TRAIN_EXAMPLES * hp.num_epochs / duration,
  }
  logging.info('Train metadata: %r', metadata)
  metadata_path = os.path.join(train_dir, 'train_metadata.json')
  logging.info('Writing train metadata to %s', metadata_path)
  with gfile.Open(metadata_path, 'w') as fh:
    json.dump(metadata, fh, indent=2, sort_keys=True)

  if save_disk_space:
    _clean_up_files(train_dir)


def evaluate_estimator(
    use_tpu_estimator: bool,
    use_tpu: bool,
    train_dir: Text,
    data_dir: Text,
    ds: data.DatasetSplit,
    master: Text,
    save_disk_space: bool,
    checkpoint_path: Text = None,
) -> Dict[Text, Any]:
  """Evaluates the model.

  Args:
    use_tpu_estimator: Whether to use the TPUEstimator as opposed to the
      plain tf.estimator.Estimator.
    use_tpu: Whether to use a TPU.
    train_dir: Where to write checkpoints, etc.
    data_dir: Where the CIFAR-10 data may be found.
    ds: The dataset split to evaluate on.
    master: URL of the Cloud TPU instance.
    save_disk_space: If true, non-essential logs are not saved to disk.
    checkpoint_path: If provided, evaluate the given checkpoint.
      Else use the latest one.

  Returns:
    Metrics.
  """
  logging.info('Evaluating on %s set', ds.name)

  hparams_path = _hparams_path(train_dir)
  logging.info('Reading HParams from %s', hparams_path)
  with gfile.Open(hparams_path, 'r') as fh:
    s = fh.read()

  hp = json_to_hparams(s)

  estimator = _get_estimator(
      use_tpu_estimator=use_tpu_estimator,
      use_tpu=use_tpu,
      train_dir=train_dir,
      hp=hp,
      master=master,
      iterations_per_loop=EVAL_ITERATIONS_PER_LOOP,
      use_model_parallelism=False)

  num_examples = {
      data.DatasetSplit.TRAIN: NUM_TRAIN_EXAMPLES,
      data.DatasetSplit.EVAL: NUM_EVAL_EXAMPLES,
  }[ds]
  batch_size = EVAL_BATCH_SIZE

  start_time = time.time()
  metrics = estimator.evaluate(
      input_fn=data.make_estimator_input_fn(
          dataset_dir=data_dir, ds=ds, return_dataset=False),
      steps=num_examples // batch_size,
      checkpoint_path=checkpoint_path,
      name=None)
  stop_time = time.time()

  duration = stop_time - start_time
  metadata = {
      'start_time': start_time,
      'stop_time': stop_time,
      'duration': duration,
      'loss': float(metrics['loss']),
      'accuracy': float(metrics['accuracy']),
      'examples_per_second': num_examples / duration,
  }
  logging.info('Eval on %s metadata: %r', ds.name, metadata)
  metadata_path = os.path.join(train_dir, 'eval_on_%s_metadata.json' % ds.name)
  logging.info('Writing eval metadata to %s', metadata_path)
  with gfile.Open(metadata_path, 'w') as fh:
    json.dump(metadata, fh, indent=2, sort_keys=True)

  if save_disk_space:
    _clean_up_files(train_dir)

  return metrics
