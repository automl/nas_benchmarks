"""ResNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from typing import Any, Dict, List, Optional, NamedTuple, Text, Union  # pytype: disable=not-supported-yet

# pylint: disable=invalid-name
HParams = NamedTuple(
    'HParams',
    [
        ('batch_size', int),
        ('initial_lr', float),
        ('decay_steps', int),
        ('n_filters_1', int),
        ('n_filters_2', int),
        ('n_filters_3', int),
        ('stride_1', int),
        ('stride_2', int),
        ('stride_3', int),
        ('depthwise', bool),
        ('num_residual_units_1', int),
        ('num_residual_units_2', int),
        ('num_residual_units_3', int),
        ('k', int),
        ('weight_decay', float),
        ('momentum', float),
        ('use_nesterov', bool),
        ('activation_1', Text),
        ('activation_2', Text),
        ('activation_3', Text),
        ('n_conv_layers_1', int),
        ('n_conv_layers_2', int),
        ('n_conv_layers_3', int),
        ('dropout_1', float),
        ('dropout_2', float),
        ('dropout_3', float),
    ],
)

# pylint: enable=invalid-name


def step_decay(learning_rate: float, global_step: tf.Tensor) -> tf.Tensor:
  """Calculates learning rate as a function of global step."""

  def f1():
    return learning_rate

  def f2():
    return tf.multiply(learning_rate, 0.2)

  def f3():
    return tf.multiply(learning_rate, 0.04)

  def f4():
    return tf.multiply(learning_rate, 0.008)

  lr = tf.case(
      [(tf.less(global_step, 23400), f1), (tf.less(global_step, 46800), f2),
       (tf.less(global_step, 62400), f3)],
      default=f4,
      exclusive=False)
  return lr


class ResNet(object):
  """ResNet model."""

  def __init__(
      self,
      hps: HParams,
      tpu_only_ops: bool,
      use_tpu: bool,
      is_training: Optional[bool],
      images: tf.Tensor,
      labels: tf.Tensor,
      lr_decay: Text = 'cosine',
      optimizer: Text = 'mom',
  ):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      tpu_only_ops: Whether to restrict to ops that can run on TPUs.
      use_tpu: Whether to actually use the TPU.
      is_training: If provided, specifies whether this graph is constructed
        for training.
        Else a placeholder will be created.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      lr_decay: Learning rate decay method.
      optimizer: Optmization method.
    """
    self.hps = hps
    self.tpu_only_ops = tpu_only_ops
    self.use_tpu = use_tpu
    if is_training is not None:
      self.is_training = is_training
    else:
      self.is_training = tf.placeholder(tf.bool)
    self._images = images
    self.labels = labels
    self.use_bottleneck = False
    self.num_classes = 10
    self.optimizer = optimizer
    self.lr_decay = lr_decay

    # Set later.
    self.global_step = None
    self.summaries = None
    self.acc = None

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.train.get_or_create_global_step()
    self._build_model()
    self._build_train_op()
    if not self.tpu_only_ops:
      self.summaries = tf.summary.merge_all()

    labels = tf.cast(self.labels, tf.int64)
    predict = tf.argmax(self.predictions, axis=1)
    self.acc = tf.reduce_mean(tf.to_float(tf.equal(predict, labels)))

  def _stride_arr(self, stride: int):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      n = 3 * 3 * 16
      kernel = tf.get_variable(
          'DW', [3, 3, 3, 16],
          tf.float32,
          initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
      conv = tf.nn.conv2d(x, kernel, self._stride_arr(1), padding='SAME')
      biases = tf.get_variable(
          'biases', [16], initializer=tf.constant_initializer(0))
      x = tf.nn.bias_add(conv, biases)

    strides = [self.hps.stride_1, self.hps.stride_2, self.hps.stride_3]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    filters = [
        16, self.hps.n_filters_1 * self.hps.k,
        self.hps.n_filters_2 * self.hps.k, self.hps.n_filters_3 * self.hps.k
    ]

    with tf.variable_scope('unit_1_0'):
      x = res_func(
          x,
          filters[0],
          filters[1],
          self._stride_arr(strides[0]),
          activate_before_residual[0],
          do_projection=True,
          activation=self.hps.activation_1,
          n_conv_layers=self.hps.n_conv_layers_1,
          dropout_rate=self.hps.dropout_1)

    for i in six.moves.range(1, self.hps.num_residual_units_1):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(
            x,
            filters[1],
            filters[1],
            self._stride_arr(1),
            False,
            activation=self.hps.activation_1,
            n_conv_layers=self.hps.n_conv_layers_1,
            dropout_rate=self.hps.dropout_1)

    with tf.variable_scope('unit_2_0'):
      x = res_func(
          x,
          filters[1],
          filters[2],
          self._stride_arr(strides[1]),
          activate_before_residual[1],
          do_projection=True,
          activation=self.hps.activation_2,
          n_conv_layers=self.hps.n_conv_layers_2,
          dropout_rate=self.hps.dropout_2)

    for i in six.moves.range(1, self.hps.num_residual_units_2):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(
            x,
            filters[2],
            filters[2],
            self._stride_arr(1),
            False,
            activation=self.hps.activation_2,
            n_conv_layers=self.hps.n_conv_layers_2,
            dropout_rate=self.hps.dropout_2)

    with tf.variable_scope('unit_3_0'):
      x = res_func(
          x,
          filters[2],
          filters[3],
          self._stride_arr(strides[2]),
          activate_before_residual[2],
          do_projection=True,
          activation=self.hps.activation_3,
          n_conv_layers=self.hps.n_conv_layers_3,
          dropout_rate=self.hps.dropout_3)

    for i in six.moves.range(1, self.hps.num_residual_units_3):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(
            x,
            filters[3],
            filters[3],
            self._stride_arr(1),
            False,
            activation=self.hps.activation_3,
            n_conv_layers=self.hps.n_conv_layers_3,
            dropout_rate=self.hps.dropout_3)

    with tf.variable_scope('unit_last'):
      x = tf.contrib.layers.batch_norm(
          x, center=True, scale=True, is_training=self.is_training)
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

      if not self.tpu_only_ops:
        tf.summary.scalar('loss', self.loss)

  def _build_train_op(self):
    """Build training specific ops for the graph."""

    if self.lr_decay == 'cosine':
      self.lrn_rate = tf.train.cosine_decay(
          self.hps.initial_lr, self.global_step, self.hps.decay_steps)
    elif self.lr_decay == 'step':
      self.lrn_rate = step_decay(self.hps.initial_lr, self.global_step)

    if not self.tpu_only_ops:
      tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.loss, trainable_variables)

    self.grad_norms = tf.global_norm(grads)
    self.norms = tf.global_norm(trainable_variables)

    if self.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(
          self.lrn_rate, self.hps.momentum, use_nesterov=self.hps.use_nesterov)

    if self.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step,
        name='train_step')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_ops = [apply_op] + update_ops
    self.train_op = tf.group(*train_ops)

  def _residual(
      self,
      x: tf.Tensor,
      in_filter: int,
      out_filter: int,
      stride: List[int],
      activate_before_residual: bool = False,
      do_projection: bool = False,
      activation: Text = 'relu',
      n_conv_layers: int = 2,
      dropout_rate: float = 0.0,
  ):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = tf.contrib.layers.batch_norm(
            x, center=True, scale=True, is_training=self.is_training)

        if activation == 'relu':
          x = tf.nn.relu(x)
        elif activation == 'elu':
          x = tf.nn.elu(x)
        elif activation == 'leaky_relu':
          x = tf.nn.leaky_relu(x)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = tf.contrib.layers.batch_norm(
            x, center=True, scale=True, is_training=self.is_training)
        if activation == 'relu':
          x = tf.nn.relu(x)
        elif activation == 'elu':
          x = tf.nn.elu(x)
        elif activation == 'leaky_relu':
          x = tf.nn.leaky_relu(x)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    for i in range(1, n_conv_layers):

      with tf.variable_scope('sub%d' % (i + 1)):
        x = tf.contrib.layers.batch_norm(
            x, center=True, scale=True, is_training=self.is_training)
        if activation == 'relu':
          x = tf.nn.relu(x)
        elif activation == 'elu':
          x = tf.nn.elu(x)
        elif activation == 'leaky_relu':
          x = tf.nn.leaky_relu(x)

        x = tf.layers.dropout(x, rate=dropout_rate, training=self.is_training)

        x = self._conv('conv%d' % (i + 1), x, 3, out_filter, out_filter,
                       [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
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

  def _conv(
      self,
      name: Text,
      x: tf.Tensor,
      filter_size: int,
      in_filters: int,
      out_filters: int,
      strides: List[int],
  ):
    """Convolution."""
    with tf.variable_scope(name):
      if self.hps.depthwise == 1:
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW',
            [filter_size, filter_size, in_filters, out_filters // in_filters],
            tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        return tf.nn.depthwise_conv2d(x, kernel, strides, padding='SAME')
      else:
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        conv = tf.nn.conv2d(x, kernel, strides, padding='SAME')

        biases = tf.get_variable(
            'biases', [out_filters], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(conv, biases)

  def _fully_connected(self, x: tf.Tensor, out_dim: int):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable(
        'biases', [out_dim], initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x: tf.Tensor):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


def make_estimator_model_fn(
    tpu_only_ops: bool,
    make_tpu_estimator_spec: bool,
    use_tpu: bool,
    lr_decay: Text,
):
  """Makes a model function for tf.estimator.Estimator.

  Args:
    tpu_only_ops: Whether to restrict to ops that can run on TPUs.
    make_tpu_estimator_spec: Whether the function should return a
      TPUEstimatorSpec rather than an EstimatorSpec.
    use_tpu: Whether to actually use the TPU.
    lr_decay: The learning rate decay method.

  Returns:
    A model function.

  Raises:
    ValueError: There are conflicting batch sizes.
  """

  def model_fn(
      features: tf.Tensor,
      labels: tf.Tensor,
      mode: Text,
      params: Dict[Text, Any],
  ) -> Union[tf.estimator.EstimatorSpec, tf.contrib.tpu.TPUEstimatorSpec]:
    """model_fn for tf.estimator.Estimator."""
    input_op = features
    del features
    target_op = labels
    del labels

    batch_size = input_op.shape[0]
    if batch_size != target_op.shape[0]:
      raise ValueError('Batch size mismatch')
    if 'batch_size' in params and batch_size != params['batch_size']:
      raise ValueError('Batch size mismatch')

    params.update({'batch_size': batch_size})
    hp = HParams(**params)

    network = ResNet(
        hps=hp,
        tpu_only_ops=tpu_only_ops,
        use_tpu=use_tpu,
        is_training=mode == tf.estimator.ModeKeys.TRAIN,
        images=input_op,
        labels=target_op,
        lr_decay=lr_decay)
    network.build_graph()

    loss_op = network.loss
    train_op = network.train_op

    if make_tpu_estimator_spec:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss_op, train_op=train_op)
    else:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions={},
          loss=loss_op,
          train_op=train_op,
          eval_metric_ops={})

  return model_fn
