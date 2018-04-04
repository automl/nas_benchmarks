import os
import json
import numpy as np
import tensorflow as tf

from wrn_cifar10.train import iterate_minibatches, load_data

from wrn_cifar10 import model as resnet  # pylint: disable=g-bad-import-order

app = tf.app
logging = tf.logging
flags = tf.flags
gfile = tf.gfile

IMAGE_SIZE = 32

flags.DEFINE_string('model_dir', './train', """Directory where model checkpoint are stored.""")
flags.DEFINE_integer('epoch_id', 28125, """Epoch number""")
flags.DEFINE_string('save_dir', './output', """Directory where metrics will be stored.""")


FLAGS = tf.app.flags.FLAGS


def evaluate():
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
    logging.info('Use depthwise convolutions: %d', FLAGS.depthwise)
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
        # Load the data
        # pylint: disable=invalid-name
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
            FLAGS.data_dir)
        # pylint: enable=invalid-name

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32,
                                [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        hp = resnet.HParams(
            batch_size=FLAGS.batch_size,
            num_residual_units_1=FLAGS.num_residual_units_1,
            num_residual_units_2=FLAGS.num_residual_units_3,
            num_residual_units_3=FLAGS.num_residual_units_2,
            n_filters_1=FLAGS.n_filters_1,
            n_filters_2=FLAGS.n_filters_2,
            n_filters_3=FLAGS.n_filters_3,
            depthwise=FLAGS.depthwise,
            stride_1=FLAGS.stride_1,
            stride_2=FLAGS.stride_2,
            stride_3=FLAGS.stride_3,
            k=FLAGS.k,
            weight_decay=FLAGS.l2_weight,
            initial_lr=FLAGS.initial_lr,
            decay_steps=int(FLAGS.num_epochs *
                            (X_train.shape[0] // FLAGS.batch_size)),
            momentum=FLAGS.momentum,
            use_nesterov=bool(FLAGS.use_nesterov),
            activation_1=FLAGS.activation_1,
            activation_2=FLAGS.activation_2,
            activation_3=FLAGS.activation_3,
            n_conv_layers_1=FLAGS.n_conv_layers_1,
            n_conv_layers_2=FLAGS.n_conv_layers_2,
            n_conv_layers_3=FLAGS.n_conv_layers_3,
            dropout_1=FLAGS.dropout_1,
            dropout_2=FLAGS.dropout_2,
            dropout_3=FLAGS.dropout_3)

        network = resnet.ResNet(hp, images, labels, lr_decay=FLAGS.lr_decay)
        network.build_graph()

        epoch = FLAGS.epoch_id
        # Start running operations on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
                log_device_placement=FLAGS.log_device_placement))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)

        saver.restore(sess, os.path.join(FLAGS.train_dir, "model.ckpt-%d" % epoch))

        # Validate
        duration_valid, valid_loss, valid_acc = 0.0, 0.0, 0.0

        updates_per_epoch = 0

        for batch in iterate_minibatches(X_valid, y_valid, FLAGS.batch_size, shuffle=False, augment=False):
            valid_images, valid_labels = batch
            loss_value, acc_value = sess.run([network.loss, network.acc],
                                             feed_dict={images: valid_images,
                                                        labels: valid_labels,
                                                        network.is_training: False})

            valid_loss += loss_value
            valid_acc += acc_value
            updates_per_epoch += 1

        valid_loss /= updates_per_epoch
        valid_acc /= updates_per_epoch

        # Test
        duration_test, test_loss, test_acc = 0.0, 0.0, 0.0

        updates_per_epoch = 0
        for batch in iterate_minibatches(X_test, y_test, FLAGS.batch_size, shuffle=False, augment=False):
            test_images, test_labels = batch
            loss_value, acc_value = sess.run([network.loss, network.acc],
                                             feed_dict={images: test_images,
                                                        labels: test_labels,
                                                        network.is_training: False})

            test_loss += loss_value
            test_acc += acc_value
            updates_per_epoch += 1

        test_loss /= updates_per_epoch
        test_acc /= updates_per_epoch

        # Save results
        results = dict()
        results["valid_loss"] = float(valid_loss)
        results["valid_acc"] = float(valid_acc)
        results["test_loss"] = float(test_loss)
        results["test_acc"] = float(test_acc)

        n_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        results["number_of_parameters"] = int(n_params.value)
        results["configuration"] = hp._asdict()
        # results["runtime_train_epochs"] = runtime_train_epochs
        # results["runtime_test_epochs"] = runtime_test_epochs
        # results["runtime_valid_epochs"] = runtime_valid_epochs

        results["epoch"] = epoch

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        results["flops"] = flops.total_float_ops

        fh = open(os.path.join(FLAGS.save_dir, "results.json"), "w")
        json.dump(results, fh)
        fh.close()


def main(argv=None):
    logging.set_verbosity('INFO')
    evaluate()


if __name__ == '__main__':
    tf.app.run()
