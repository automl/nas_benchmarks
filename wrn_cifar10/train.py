"""Code for evaluating a particular model on the CIFAR-10 task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import numpy as np
import tensorflow as tf

from wrn_cifar10 import controller  # pylint: disable=g-bad-import-order
from wrn_cifar10 import data  # pylint: disable=g-bad-import-order
from wrn_cifar10 import model  # pylint: disable=g-bad-import-order

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

# Dataset Configuration
flags.DEFINE_string('data_dir', './cifar-10-batches-py/',
                    """Path to the CIFAR-10 python data.""")

# Network Configuration
flags.DEFINE_integer('num_residual_units_1', 4,
                     """Number of residual block in first group.""")
flags.DEFINE_integer('num_residual_units_2', 4,
                     """Number of residual block in second group.""")
flags.DEFINE_integer('num_residual_units_3', 4,
                     """Number of residual block in third group.""")
flags.DEFINE_integer('n_filters_1', 16, """Number of filters in first group.""")
flags.DEFINE_integer('n_filters_2', 32,
                     """Number of filters in second group.""")
flags.DEFINE_integer('n_filters_3', 64, """Number of filters in third group.""")
flags.DEFINE_integer('stride_1', 1, """Stride in first group.""")
flags.DEFINE_integer('stride_2', 2, """Stride in second group.""")
flags.DEFINE_integer('stride_3', 2, """Stride in third group.""")
# TODO(ericmc): Make this name more informative.
flags.DEFINE_integer('k', 10, """Network width multiplier""")
flags.DEFINE_bool('depthwise', False,
                  """Whether to use depthwise convolutions""")
flags.DEFINE_enum('activation_1', model.Activation.RELU.name,
                  model.Activation.__members__,
                  """Activation function for the first group""")
flags.DEFINE_enum('activation_2', model.Activation.RELU.name,
                  model.Activation.__members__,
                  """Activation function for the second group""")
flags.DEFINE_enum('activation_3', model.Activation.RELU.name,
                  model.Activation.__members__,
                  """Activation function for the third group""")
flags.DEFINE_integer(
    'n_conv_layers_1', 2,
    """Number of convolutions in residual block for the first group""")
flags.DEFINE_integer(
    'n_conv_layers_2', 2,
    """Number of convolutions in residual block for the second group""")
flags.DEFINE_integer(
    'n_conv_layers_3', 2,
    """Number of convolutions in residual block for the third group""")
flags.DEFINE_float('dropout_1', 0, """Dropout for the first group""")
flags.DEFINE_float('dropout_2', 0, """Dropout for the second group""")
flags.DEFINE_float('dropout_3', 0, """Dropout for the third group""")

# Optimization Configuration
flags.DEFINE_integer('batch_size', 128,
                     """Number of images to process in a batch.""")
flags.DEFINE_float('l2_weight', 0.0005,
                   """L2 loss weight applied all the weights""")
flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
flags.DEFINE_enum('lr_decay', model.LRDecaySchedule.COSINE.name,
                  model.LRDecaySchedule.__members__, """LR Schedule""")
flags.DEFINE_bool('use_nesterov', False,
                  """Whether to use Nesterov momentum.""")

flags.DEFINE_float('dropout', 0.1, """Dropout""")

# Training Configuration
flags.DEFINE_string('train_dir', './train',
                    """Directory where to write log and checkpoint.""")
flags.DEFINE_integer('num_epochs', 20, """Number of batches to run.""")
flags.DEFINE_integer('checkpoint_interval', 1,
                     """Number of epochs to save parameters as a checkpoint""")
flags.DEFINE_float('gpu_fraction', 0.95,
                   """The fraction of GPU memory to be allocated""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_boolean('do_save', False, """Whether to save checkpoints.""")
flags.DEFINE_integer('training_time', 7200, """Training time in seconds.""")

flags.DEFINE_bool('use_estimator_code_path', False,
                  'Use the estimator code path.')
flags.DEFINE_bool('use_tpu_estimator', False,
                  'Whether to use the TPUEstimator vs the standard Estimator.')
flags.DEFINE_bool('use_tpu', False, 'Use TPUs rather than plain CPUs')
flags.DEFINE_string('master', 'local', 'GRPC URL of the Cloud TPU instance.')
flags.DEFINE_integer('iterations_per_loop', 1 << 6,
                     'Number of iterations per TPU training loop.')
flags.DEFINE_bool(
    'save_disk_space', False,
    'Whether to omit some state files to reduce disk space usage.')
flags.DEFINE_integer('replicate', 0, 'The index of the replicate.')

# More flags in shared_flags.py.
FLAGS = flags.FLAGS


def make_hparams_from_flags() -> tf.contrib.training.HParams:
  """Makes an HParams object from the provided flags."""
  hp = tf.contrib.training.HParams(
      # Optimization.
      optimizer=model.Optimizer.MOMENTUM,
      initial_lr=FLAGS.initial_lr,
      lr_decay=model.LRDecaySchedule[FLAGS.lr_decay],
      decay_steps=int(FLAGS.num_epochs *
                      (controller.NUM_TRAIN_EXAMPLES // FLAGS.batch_size)),
      weight_decay=FLAGS.l2_weight,
      momentum=FLAGS.momentum,
      use_nesterov=FLAGS.use_nesterov,
      # Architecture.
      n_filters_1=FLAGS.n_filters_1,
      n_filters_2=FLAGS.n_filters_2,
      n_filters_3=FLAGS.n_filters_3,
      stride_1=FLAGS.stride_1,
      stride_2=FLAGS.stride_2,
      stride_3=FLAGS.stride_3,
      depthwise=FLAGS.depthwise,
      num_residual_units_1=FLAGS.num_residual_units_1,
      num_residual_units_2=FLAGS.num_residual_units_3,
      num_residual_units_3=FLAGS.num_residual_units_2,
      k=FLAGS.k,
      activation_1=model.Activation[FLAGS.activation_1],
      activation_2=model.Activation[FLAGS.activation_2],
      activation_3=model.Activation[FLAGS.activation_3],
      n_conv_layers_1=FLAGS.n_conv_layers_1,
      n_conv_layers_2=FLAGS.n_conv_layers_2,
      n_conv_layers_3=FLAGS.n_conv_layers_3,
      dropout_1=FLAGS.dropout_1,
      dropout_2=FLAGS.dropout_2,
      dropout_3=FLAGS.dropout_3,
      # Misc.
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      replicate=FLAGS.replicate,
  )

  return model.validate(hp)


# TODO(ericmc): Integrate this logic into the tf.estimator.Estimator code path
# and remove FLAGS.use_estimator_code_path.
def train():
  """Trains model and writes results to disk."""
  logging.info('[Dataset Configuration]')
  logging.info('CIFAR-10 dir: %s', FLAGS.data_dir)

  logging.info('[Network Configuration]')
  logging.info('Batch size: %d', FLAGS.batch_size)
  logging.info('Residual blocks first group: %d', FLAGS.num_residual_units_1)
  logging.info('Residual blocks second group: %d', FLAGS.num_residual_units_2)
  logging.info('Residual blocks third group: %d', FLAGS.num_residual_units_3)
  logging.info('Filters first group: %d', FLAGS.n_filters_1)
  logging.info('Filters second group: %d', FLAGS.n_filters_2)
  logging.info('Filters third group: %d', FLAGS.n_filters_3)
  logging.info('Stride first group: %d', FLAGS.stride_1)
  logging.info('Stride second group: %d', FLAGS.stride_2)
  logging.info('Stride third group: %d', FLAGS.stride_3)
  logging.info('Use depthwise convolutions: %r', FLAGS.depthwise)
  logging.info('Network width multiplier: %d', FLAGS.k)
  logging.info('Activation function first group: %s', FLAGS.activation_1)
  logging.info('Activation function second group: %s', FLAGS.activation_2)
  logging.info('Activation function third group: %s', FLAGS.activation_3)
  logging.info('Number of convolutions in residual block first group: %d',
               FLAGS.n_conv_layers_1)
  logging.info('Number of convolutions in residual block second group: %d',
               FLAGS.n_conv_layers_2)
  logging.info('Number of convolutions in residual block third group: %d',
               FLAGS.n_conv_layers_3)
  logging.info('Dropout in residual block first group: %f', FLAGS.dropout_1)
  logging.info('Dropout in residual block second group: %f', FLAGS.dropout_2)
  logging.info('Dropout in residual block third group: %f', FLAGS.dropout_3)

  logging.info('[Optimization Configuration]')
  logging.info('L2 loss weight: %f', FLAGS.l2_weight)
  logging.info('The momentum optimizer: %f', FLAGS.momentum)
  logging.info('Initial learning rate: %f', FLAGS.initial_lr)
  logging.info('Learning decaying strategy: %s', FLAGS.lr_decay)
  logging.info('Nesterov Momentum: %s', bool(FLAGS.use_nesterov))

  logging.info('[Training Configuration]')
  logging.info('Train dir: %s', FLAGS.train_dir)
  logging.info('Training number of epochs: %d', FLAGS.num_epochs)
  logging.info('Steps per saving checkpoints: %d', FLAGS.checkpoint_interval)
  logging.info('GPU memory fraction: %f', FLAGS.gpu_fraction)
  logging.info('Log device placement: %d', FLAGS.log_device_placement)
  logging.info('Training time: %d', FLAGS.training_time)

  with tf.Graph().as_default() as g:

    run_meta = tf.RunMetadata()

    init_step = 0

    # Load the data
    # pylint: disable=invalid-name
    X_train, y_train, X_valid, y_valid, X_test, y_test = data.load_data(
        FLAGS.data_dir)
    # pylint: enable=invalid-name

    # Build a Graph that computes the predictions from the inference model.
    images = tf.placeholder(
        tf.float32, [FLAGS.batch_size, data.IMAGE_SIZE, data.IMAGE_SIZE, 3])
    labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

    # Build model
    hp = make_hparams_from_flags()

    network = model.ResNet(
        hps=hp,
        tpu_only_ops=False,
        use_tpu=False,
        is_training=None,
        images=images,
        labels=labels)
    network.build_graph()

    # Summaries(training)
    train_summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      logging.info('Restore from %s', ckpt.model_checkpoint_path)
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      logging.info('No checkpoint file found. Start from scratch.')

    # Start queue runners & summary_writer
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    # Training!
    test_best_acc = 0.0
    valid_best_acc = 0.0

    learning_curve_valid_loss_updates = []
    learning_curve_train_loss_updates = []
    learning_curve_test_loss_updates = []
    learning_curve_valid_acc_updates = []
    learning_curve_train_acc_updates = []
    learning_curve_test_acc_updates = []
    learning_curve_valid_loss_epochs = []
    learning_curve_train_loss_epochs = []
    learning_curve_test_loss_epochs = []
    learning_curve_valid_acc_epochs = []
    learning_curve_train_acc_epochs = []
    learning_curve_test_acc_epochs = []

    grad_norms_mean = []
    grad_norms_std = []
    grad_norms_q05 = []
    grad_norms_q10 = []
    grad_norms_q25 = []
    grad_norms_median = []
    grad_norms_q75 = []
    grad_norms_q90 = []
    grad_norms_q95 = []
    grad_norms_max = []
    grad_norms_min = []

    norms_mean = []
    norms_std = []
    norms_q05 = []
    norms_q10 = []
    norms_q25 = []
    norms_median = []
    norms_q75 = []
    norms_q90 = []
    norms_q95 = []
    norms_max = []
    norms_min = []

    runtime_train_epochs = []
    runtime_valid_epochs = []
    runtime_test_epochs = []

    start_training_time = time.time()

    duration_last_epoch = 0
    used_budget = 0
    e = 1

    while (e < FLAGS.num_epochs + 1) and (
        used_budget + 1.1 * duration_last_epoch < FLAGS.training_time):

      train_loss = 0
      train_acc = 0
      duration_train = 0

      norms = []
      grad_norms = []

      updates_per_epoch = 0
      for batch in data.iterate_minibatches(
          sess, X_train, y_train, FLAGS.batch_size, shuffle=True, augment=True):
        start_time_train = time.time()
        train_images, train_labels = batch

        # Update model parameters
        (_, lr_value, loss_value, acc_value, train_summary_str, norm,
         grad_norm) = sess.run(
             [
                 network.train_op, network.lrn_rate, network.loss, network.acc,
                 train_summary_op, network.norms, network.grad_norms
             ],
             feed_dict={
                 images: train_images,
                 labels: train_labels,
                 network.is_training: True
             })

        norms.append(norm)
        grad_norms.append(grad_norm)

        learning_curve_train_loss_updates.append(float(loss_value))
        learning_curve_train_acc_updates.append(float(acc_value))

        train_loss += loss_value
        train_acc += acc_value
        duration_train += time.time() - start_time_train
        updates_per_epoch += 1

      # Display & Summary(training)
      sec_per_batch = float(duration_train / updates_per_epoch)

      format_str = ('(Training) Epoch %d, loss=%.4f, acc=%.4f, lr=%f (%.3f '
                    'sec/batch)')
      logging.info(format_str, e, train_loss / updates_per_epoch,
                   train_acc / updates_per_epoch, lr_value, sec_per_batch)
      summary_writer.add_summary(train_summary_str, e)

      learning_curve_train_loss_epochs.append(
          float(train_loss / updates_per_epoch))
      learning_curve_train_acc_epochs.append(
          float(train_acc / updates_per_epoch))

      runtime_train_epochs.append(float(duration_train))

      grad_norms_mean.append(float(np.mean(grad_norms)))
      grad_norms_std.append(float(np.std(grad_norms)))
      grad_norms_q05.append(float(np.percentile(grad_norms, q=5)))
      grad_norms_q10.append(float(np.percentile(grad_norms, q=10)))
      grad_norms_q25.append(float(np.percentile(grad_norms, q=25)))
      grad_norms_median.append(float(np.median(grad_norms)))
      grad_norms_q75.append(float(np.percentile(grad_norms, q=75)))
      grad_norms_q90.append(float(np.percentile(grad_norms, q=90)))
      grad_norms_q95.append(float(np.percentile(grad_norms, q=95)))
      grad_norms_max.append(float(np.max(grad_norms)))
      grad_norms_min.append(float(np.min(grad_norms)))

      norms_mean.append(float(np.mean(norms)))
      norms_std.append(float(np.std(norms)))
      norms_q05.append(float(np.percentile(norms, q=5)))
      norms_q10.append(float(np.percentile(norms, q=10)))
      norms_q25.append(float(np.percentile(norms, q=25)))
      norms_median.append(float(np.median(norms)))
      norms_q75.append(float(np.percentile(norms, q=75)))
      norms_q90.append(float(np.percentile(norms, q=90)))
      norms_q95.append(float(np.percentile(norms, q=95)))
      norms_max.append(float(np.max(norms)))
      norms_min.append(float(np.min(norms)))

      # Validate
      duration_valid, valid_loss, valid_acc = 0.0, 0.0, 0.0

      start_time_valid = time.time()
      updates_per_epoch = 0
      for batch in data.iterate_minibatches(
          sess,
          X_valid,
          y_valid,
          FLAGS.batch_size,
          shuffle=False,
          augment=False):
        valid_images, valid_labels = batch
        loss_value, acc_value = sess.run(
            [network.loss, network.acc],
            feed_dict={
                images: valid_images,
                labels: valid_labels,
                network.is_training: False
            })

        learning_curve_valid_loss_updates.append(float(loss_value))
        learning_curve_valid_acc_updates.append(float(acc_value))

        valid_loss += loss_value
        valid_acc += acc_value
        updates_per_epoch += 1

      valid_loss /= updates_per_epoch
      valid_acc /= updates_per_epoch
      duration_valid = time.time() - start_time_valid

      runtime_valid_epochs.append(float(duration_valid))
      learning_curve_valid_loss_epochs.append(float(valid_loss))
      learning_curve_valid_acc_epochs.append(float(valid_acc))

      valid_best_acc = max(valid_best_acc, valid_acc)
      format_str = '(Valid)     Epoch %d, loss=%.4f, acc=%.4f'
      logging.info(format_str, e, valid_loss, valid_acc)

      valid_summary = tf.Summary()
      valid_summary.value.add(tag='valid/loss', simple_value=valid_loss)
      valid_summary.value.add(tag='valid/acc', simple_value=valid_acc)
      valid_summary.value.add(tag='valid/best_acc', simple_value=valid_best_acc)
      summary_writer.add_summary(valid_summary, e)
      summary_writer.flush()

      # Test
      duration_test, test_loss, test_acc = 0.0, 0.0, 0.0

      start_time_test = time.time()

      updates_per_epoch = 0
      for batch in data.iterate_minibatches(
          sess, X_test, y_test, FLAGS.batch_size, shuffle=False, augment=False):
        test_images, test_labels = batch
        loss_value, acc_value = sess.run(
            [network.loss, network.acc],
            feed_dict={
                images: test_images,
                labels: test_labels,
                network.is_training: False
            })

        learning_curve_test_loss_updates.append(float(loss_value))
        learning_curve_test_acc_updates.append(float(acc_value))

        test_loss += loss_value
        test_acc += acc_value
        updates_per_epoch += 1
      test_loss /= updates_per_epoch
      test_acc /= updates_per_epoch
      duration_test = time.time() - start_time_test

      runtime_test_epochs.append(float(duration_test))
      learning_curve_test_loss_epochs.append(float(test_loss))
      learning_curve_test_acc_epochs.append(float(test_acc))

      test_best_acc = max(test_best_acc, test_acc)
      format_str = '(Test)     Epoch %d, loss=%.4f, acc=%.4f'
      logging.info(format_str, e, test_loss, test_acc)

      test_summary = tf.Summary()
      test_summary.value.add(tag='test/loss', simple_value=test_loss)
      test_summary.value.add(tag='test/acc', simple_value=test_acc)
      test_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
      summary_writer.add_summary(test_summary, e)
      summary_writer.flush()

      # Save the model checkpoint periodically.
      if FLAGS.do_save:
        if (e > init_step and
            e % FLAGS.checkpoint_interval == 0) or (e + 1) == FLAGS.num_epochs:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(
              sess, checkpoint_path, global_step=int(e * FLAGS.batch_size))

      duration_last_epoch = (time.time() - start_training_time) - used_budget
      used_budget += duration_last_epoch
      e += 1

    # Save results
    results = dict()
    results[
        'learning_curve_valid_loss_epochs'] = learning_curve_valid_loss_epochs
    results[
        'learning_curve_train_loss_epochs'] = learning_curve_train_loss_epochs
    results['learning_curve_test_loss_epochs'] = learning_curve_test_loss_epochs
    results['learning_curve_valid_acc_epochs'] = learning_curve_valid_acc_epochs
    results['learning_curve_train_acc_epochs'] = learning_curve_train_acc_epochs
    results['learning_curve_test_acc_epochs'] = learning_curve_test_acc_epochs

    results[
        'learning_curve_valid_loss_updates'] = learning_curve_valid_loss_updates
    results[
        'learning_curve_train_loss_updates'] = learning_curve_train_loss_updates
    results[
        'learning_curve_test_loss_updates'] = learning_curve_test_loss_updates
    results[
        'learning_curve_valid_acc_updates'] = learning_curve_valid_acc_updates
    results[
        'learning_curve_train_acc_updates'] = learning_curve_train_acc_updates
    results['learning_curve_test_acc_updates'] = learning_curve_test_acc_updates

    n_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    results['number_of_parameters'] = int(n_params.value)
    results['configuration'] = hp._asdict()
    results['runtime_train_epochs'] = runtime_train_epochs
    results['runtime_test_epochs'] = runtime_test_epochs
    results['runtime_valid_epochs'] = runtime_valid_epochs

    results['grad_norms_mean'] = grad_norms_mean
    results['grad_norms_std'] = grad_norms_std
    results['grad_norms_q05'] = grad_norms_q05
    results['grad_norms_q10'] = grad_norms_q10
    results['grad_norms_q25'] = grad_norms_q25
    results['grad_norms_median'] = grad_norms_median
    results['grad_norms_q75'] = grad_norms_q75
    results['grad_norms_q90'] = grad_norms_q90
    results['grad_norms_q95'] = grad_norms_q95
    results['grad_norms_max'] = grad_norms_max
    results['grad_norms_min'] = grad_norms_min

    results['norms_mean'] = norms_mean
    results['norms_std'] = norms_std
    results['norms_q05'] = norms_q05
    results['norms_q10'] = norms_q10
    results['norms_q25'] = norms_q25
    results['norms_median'] = norms_median
    results['norms_q75'] = norms_q75
    results['norms_q90'] = norms_q90
    results['norms_q95'] = norms_q95
    results['norms_max'] = norms_max
    results['norms_min'] = norms_min

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    results['flops'] = flops.total_float_ops

    with gfile.Open(os.path.join(FLAGS.train_dir, 'results.json'), 'w') as fh:
      json.dump(results, fh)

    # Compute final validation prediction
    valid_predictions = None

    for batch in data.iterate_minibatches(
        sess, X_valid, y_valid, FLAGS.batch_size, shuffle=False, augment=False):
      valid_images, valid_labels = batch
      predictions = sess.run(
          network.predictions,
          feed_dict={
              images: valid_images,
              labels: valid_labels,
              network.is_training: False
          })

      if valid_predictions is None:
        valid_predictions = predictions
      else:
        valid_predictions = np.concatenate(
            (valid_predictions, predictions), axis=0)

    with gfile.Open(
        os.path.join(FLAGS.train_dir, 'valid_predictions.npy'), 'wb') as fh:
      np.save(fh, valid_predictions)

    # Compute final test prediction
    test_predictions = None
    for batch in data.iterate_minibatches(
        sess, X_test, y_test, FLAGS.batch_size, shuffle=False, augment=False):
      test_images, test_labels = batch
      predictions = sess.run(
          network.predictions,
          feed_dict={
              images: test_images,
              labels: test_labels,
              network.is_training: False
          })

      if test_predictions is None:
        test_predictions = predictions
      else:
        test_predictions = np.concatenate(
            (test_predictions, predictions), axis=0)

    with gfile.Open(
        os.path.join(FLAGS.train_dir, 'test_predictions.npy'), 'wb') as fh:
      np.save(fh, test_predictions)


def main(argv):
  del argv  # Unused.
  logging.set_verbosity('INFO')

  if not gfile.Exists(FLAGS.train_dir):
    gfile.MkDir(FLAGS.train_dir)

  if FLAGS.use_estimator_code_path:
    controller.train_estimator(
        use_tpu_estimator=FLAGS.use_tpu_estimator,
        use_tpu=FLAGS.use_tpu,
        train_dir=FLAGS.train_dir,
        data_dir=FLAGS.data_dir,
        hp=make_hparams_from_flags(),
        master=FLAGS.master,
        iterations_per_loop=FLAGS.iterations_per_loop,
        save_disk_space=FLAGS.save_disk_space)
  else:
    train()


if __name__ == '__main__':
  app.run(main)
