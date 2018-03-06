from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

HParams = namedtuple('HParams',
                     'batch_size, initial_lr, decay_steps, '
                     'n_filters_1, n_filters_2, n_filters_3, '
                     'stride_1, stride_2, stride_3, depthwise, '
                     'num_residual_units_1, num_residual_units_2, num_residual_units_3, '
                     'k, weight_decay, momentum')


def step_decay(learning_rate,
               global_step):
    def f1(): return learning_rate

    def f2(): return tf.multiply(learning_rate, 0.2)

    def f3(): return tf.multiply(learning_rate, 0.04)

    def f4(): return tf.multiply(learning_rate, 0.008)

    lr = tf.case(
        [(tf.less(global_step, 23400), f1), (tf.less(global_step, 46800), f2), (tf.less(global_step, 62400), f3)],
        default=f4, exclusive=False)
    return lr


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels,
                 lr_decay="cosine", optimizer="mom"):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]

        """
        self.hps = hps
        # self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self._images = images
        self.labels = labels
        self.use_bottleneck = False
        self.num_classes = 10
        self.optimizer = optimizer
        # self._extra_train_ops = []
        self.lr_decay = lr_decay

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        self._build_train_op()
        self.summaries = tf.summary.merge_all()

        labels = tf.cast(self.labels, tf.int64)
        predict = tf.argmax(self.predictions, axis=1)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(predict, labels)))

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            n = 3 * 3 * 16
            kernel = tf.get_variable(
                    'DW', [3, 3, 3, 16],
                    tf.float32, initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(2.0 / n)))
            conv = tf.nn.conv2d(x, kernel, self._stride_arr(1), padding='SAME')
            biases = tf.get_variable('biases', [16],
                                     initializer=tf.constant_initializer(0))
            x = tf.nn.bias_add(conv, biases)

        # strides = [1, 2, 2]
        strides = [self.hps.stride_1, self.hps.stride_2, self.hps.stride_3]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # filters = [16, 16 * 10, 32 * 10, 64 * 10]
        filters = [16, self.hps.n_filters_1 * self.hps.k,
                   self.hps.n_filters_2 * self.hps.k,
                   self.hps.n_filters_3 * self.hps.k]

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0], do_projection=True)

        for i in six.moves.range(1, self.hps.num_residual_units_1):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1], do_projection=True)

        for i in six.moves.range(1, self.hps.num_residual_units_2):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2], do_projection=True)

        for i in six.moves.range(1, self.hps.num_residual_units_3):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                             is_training=self.is_training)
            x = tf.nn.relu(x)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('loss'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.loss = tf.reduce_mean(xent, name='xent')
            self.loss += self._decay()

            tf.summary.scalar('loss', self.loss)

    def _build_train_op(self):
        """Build training specific ops for the graph."""

        if self.lr_decay == "cosine":
            self.lrn_rate = tf.train.cosine_decay(self.hps.initial_lr, self.global_step, self.hps.decay_steps)
        elif self.lr_decay == "step":
            self.lrn_rate = step_decay(self.hps.initial_lr, self.global_step)

        tf.summary.scalar('learning_rate', self.lrn_rate)

        # trainable_variables = tf.trainable_variables()
        # grads = tf.gradients(self.loss, trainable_variables)

        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, self.hps.momentum, use_nesterov=True)

        # apply_op = optimizer.apply_gradients(
        #     zip(grads, trainable_variables),
        #     global_step=self.global_step, name='train_step')
        #
        # train_ops = [apply_op] + self._extra_train_ops
        # self.train_op = tf.group(*train_ops)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.train_op = optimizer.minimize(self.loss)

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False, do_projection=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = tf.contrib.layers.batch_norm(x, center=True,
                                                 scale=True, is_training=self.is_training)
                x = tf.nn.relu(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = tf.contrib.layers.batch_norm(x, center=True,
                                                 scale=True, is_training=self.is_training)
                x = tf.nn.relu(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = tf.contrib.layers.batch_norm(x, center=True,
                                             scale=True, is_training=self.is_training)
            x = tf.nn.relu(x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            # if in_filter != out_filter:
            if do_projection:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.multiply(self.hps.weight_decay, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            if self.hps.depthwise == 1:
                n = filter_size * filter_size * out_filters
                kernel = tf.get_variable(
                    'DW', [filter_size, filter_size, in_filters, out_filters // in_filters],
                    tf.float32, initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(2.0 / n)))
                return tf.nn.depthwise_conv2d(x, kernel, strides, padding='SAME')
            else:
                n = filter_size * filter_size * out_filters
                kernel = tf.get_variable(
                    'DW', [filter_size, filter_size, in_filters, out_filters],
                    tf.float32, initializer=tf.random_normal_initializer(
                        stddev=np.sqrt(2.0 / n)))
                conv = tf.nn.conv2d(x, kernel, strides, padding='SAME')

                biases = tf.get_variable('biases', [out_filters],
                                             initializer=tf.constant_initializer(0))
                return tf.nn.bias_add(conv, biases)

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
