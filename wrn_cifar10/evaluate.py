import os
import json
import numpy as np
import tensorflow as tf

from wrn_cifar10 import data  # pylint: disable=g-bad-import-order
from wrn_cifar10 import model # pylint: disable=g-bad-import-order
from wrn_cifar10 import controller # pylint: disable=g-bad-import-order


app = tf.app
logging = tf.logging
flags = tf.flags
gfile = tf.gfile

IMAGE_SIZE = 32

flags.DEFINE_string('data_dir', './cifar-10-batches-py/',
                    """Path to the CIFAR-10 python data.""")
flags.DEFINE_string('model_dir', './train', """Directory where model checkpoint are stored.""")
flags.DEFINE_integer('epoch_id', 28125, """Epoch number""")
flags.DEFINE_string('save_dir', './output', """Directory where metrics will be stored.""")
flags.DEFINE_float('gpu_fraction', 0.95,
                   """The fraction of GPU memory to be allocated""")
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def evaluate():

    hparams_dict = json.load(open(os.path.join(FLAGS.model_dir, "hparams.json")))

    hp = tf.contrib.training.HParams(
        # Optimization.
        optimizer=model.Optimizer.MOMENTUM,
        initial_lr=hparams_dict["initial_lr"],
        lr_decay=model.LRDecaySchedule[hparams_dict["lr_decay"]],
        decay_steps=hparams_dict["decay_steps"],
        weight_decay=hparams_dict["weight_decay"],
        momentum=hparams_dict["momentum"],
        use_nesterov=hparams_dict["use_nesterov"],
        # Architecture.
        n_filters_1=hparams_dict["n_filters_1"],
        n_filters_2=hparams_dict["n_filters_2"],
        n_filters_3=hparams_dict["n_filters_3"],
        stride_1=hparams_dict["stride_1"],
        stride_2=hparams_dict["stride_2"],
        stride_3=hparams_dict["stride_3"],
        depthwise=hparams_dict["depthwise"],
        num_residual_units_1=hparams_dict["num_residual_units_1"],
        num_residual_units_2=hparams_dict["num_residual_units_3"],
        num_residual_units_3=hparams_dict["num_residual_units_2"],
        k=hparams_dict["k"],
        activation_1=model.Activation[hparams_dict["activation_1"]],
        activation_2=model.Activation[hparams_dict["activation_2"]],
        activation_3=model.Activation[hparams_dict["activation_3"]],
        n_conv_layers_1=hparams_dict["n_conv_layers_1"],
        n_conv_layers_2=hparams_dict["n_conv_layers_2"],
        n_conv_layers_3=hparams_dict["n_conv_layers_3"],
        dropout_1=hparams_dict["dropout_1"],
        dropout_2=hparams_dict["dropout_2"],
        dropout_3=hparams_dict["dropout_3"],
        # Misc.
        batch_size=hparams_dict["batch_size"],
        num_epochs=hparams_dict["num_epochs"],
        replicate=hparams_dict["replicate"],
    )

    with tf.Graph().as_default() as g:

        run_meta = tf.RunMetadata()
        # Load the data
        # pylint: disable=invalid-name
        X_train, y_train, X_valid, y_valid, X_test, y_test = data.load_data(
            FLAGS.data_dir)
        # pylint: enable=invalid-name

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32,
                                [hp.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        labels = tf.placeholder(tf.int32, [hp.batch_size])

        # Build model
        network = model.ResNet(
            hps=hp,
            tpu_only_ops=False,
            use_tpu=False,
            is_training=None,
            images=images,
            labels=labels)
        network.build_graph()

        epoch = FLAGS.epoch_id
        # Start running operations on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
                log_device_placement=FLAGS.log_device_placement))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)

        saver.restore(sess, os.path.join(FLAGS.model_dir, "model.ckpt-%d" % epoch))

        # Validate
        duration_valid, valid_loss, valid_acc = 0.0, 0.0, 0.0

        updates_per_epoch = 0

        for batch in data.iterate_minibatches(sess, X_valid, y_valid, hp.batch_size, shuffle=False, augment=False):
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
        for batch in data.iterate_minibatches(sess, X_test, y_test, hp.batch_size, shuffle=False, augment=False):
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
        results["configuration"] = controller.hparams_to_json(hp)
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
