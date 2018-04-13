"""A repository of benchmark configurations.

This is currently used by Google code not included in the GitHub repo.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from typing import List, NamedTuple, Text

from wrn_cifar10 import model


class ParameterType(enum.Enum):
  BOOLEAN = 'BOOLEAN'
  CATEGORICAL = 'CATEGORICAL'
  DISCRETE = 'DISCRETE'


# pylint: disable=invalid-name
Parameter = NamedTuple(
    'Parameter',
    [('name', Text), ('type', ParameterType), ('domain', List)],
)
# pylint: enable=invalid-name

DEBUG = [
    # Optimization.
    ('optimizer', ParameterType.CATEGORICAL, [model.Optimizer.MOMENTUM]),
    ('initial_lr', ParameterType.DISCRETE, [3e-2]),
    ('lr_decay', ParameterType.CATEGORICAL, [model.LRDecaySchedule.COSINE]),
    ('weight_decay', ParameterType.DISCRETE, [0.0005]),
    ('momentum', ParameterType.DISCRETE, [0.9]),
    ('use_nesterov', ParameterType.BOOLEAN, [False]),

    # Architecture.
    ('n_filters_1', ParameterType.DISCRETE, [16]),
    ('n_filters_2', ParameterType.DISCRETE, [16]),
    ('n_filters_3', ParameterType.DISCRETE, [16]),
    ('stride_1', ParameterType.DISCRETE, [1]),
    ('stride_2', ParameterType.DISCRETE, [1]),
    ('stride_3', ParameterType.DISCRETE, [1]),
    ('depthwise', ParameterType.BOOLEAN, [False]),
    ('activation_1', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_2', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_3', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('num_residual_units_1', ParameterType.DISCRETE, [1, 2]),
    ('num_residual_units_2', ParameterType.DISCRETE, [1, 2]),
    ('num_residual_units_3', ParameterType.DISCRETE, [1, 2]),
    ('k', ParameterType.DISCRETE, [10]),
    ('n_conv_layers_1', ParameterType.DISCRETE, [1]),
    ('n_conv_layers_2', ParameterType.DISCRETE, [1]),
    ('n_conv_layers_3', ParameterType.DISCRETE, [1]),
    ('dropout_1', ParameterType.DISCRETE, [0.0]),
    ('dropout_2', ParameterType.DISCRETE, [0.0]),
    ('dropout_3', ParameterType.DISCRETE, [0.0]),

    # Misc.
    ('batch_size', ParameterType.DISCRETE, [16]),
    ('num_epochs', ParameterType.DISCRETE, [1]),
    ('replicate', ParameterType.DISCRETE, [0, 1]),
]

SMALL = [
    # Optimization.
    ('optimizer', ParameterType.CATEGORICAL, [model.Optimizer.MOMENTUM]),
    ('initial_lr', ParameterType.DISCRETE, [1e-1, 3e-2]),
    ('lr_decay', ParameterType.CATEGORICAL, [model.LRDecaySchedule.COSINE]),
    ('weight_decay', ParameterType.DISCRETE, [0.0005]),
    ('momentum', ParameterType.DISCRETE, [0.9]),
    ('use_nesterov', ParameterType.BOOLEAN, [False]),

    # Architecture.
    ('n_filters_1', ParameterType.DISCRETE, [16]),
    ('n_filters_2', ParameterType.DISCRETE, [16]),
    ('n_filters_3', ParameterType.DISCRETE, [16]),
    ('stride_1', ParameterType.DISCRETE, [1, 2]),
    ('stride_2', ParameterType.DISCRETE, [1, 2]),
    ('stride_3', ParameterType.DISCRETE, [1, 2]),
    ('depthwise', ParameterType.BOOLEAN, [False]),
    ('activation_1', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_2', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_3', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('num_residual_units_1', ParameterType.DISCRETE, [1, 2]),
    ('num_residual_units_2', ParameterType.DISCRETE, [1, 2]),
    ('num_residual_units_3', ParameterType.DISCRETE, [1, 2]),
    ('k', ParameterType.DISCRETE, [10]),
    ('n_conv_layers_1', ParameterType.DISCRETE, [1]),
    ('n_conv_layers_2', ParameterType.DISCRETE, [1]),
    ('n_conv_layers_3', ParameterType.DISCRETE, [1]),
    ('dropout_1', ParameterType.DISCRETE, [0.0]),
    ('dropout_2', ParameterType.DISCRETE, [0.0]),
    ('dropout_3', ParameterType.DISCRETE, [0.0]),

    # Misc.
    ('batch_size', ParameterType.DISCRETE, [16, 32]),
    ('num_epochs', ParameterType.DISCRETE, [20]),
    ('replicate', ParameterType.DISCRETE, [0, 1, 2, 3]),
]

LARGE = [
    # Optimization.
    ('optimizer', ParameterType.CATEGORICAL, [model.Optimizer.__members__]),
    ('initial_lr', ParameterType.DISCRETE, [1e-1, 3e-2, 1e-2]),
    ('lr_decay', ParameterType.CATEGORICAL,
     [model.LRDecaySchedule.__members__]),
    ('weight_decay', ParameterType.DISCRETE, [0.0005]),
    ('momentum', ParameterType.DISCRETE, [0.9]),
    ('use_nesterov', ParameterType.BOOLEAN, [False, True]),

    # Architecture.
    ('n_filters_1', ParameterType.DISCRETE, [16, 32]),
    ('n_filters_2', ParameterType.DISCRETE, [16, 32]),
    ('n_filters_3', ParameterType.DISCRETE, [16, 32]),
    ('stride_1', ParameterType.DISCRETE, [1, 2]),
    ('stride_2', ParameterType.DISCRETE, [1, 2]),
    ('stride_3', ParameterType.DISCRETE, [1, 2]),
    ('depthwise', ParameterType.BOOLEAN, [False, True]),
    ('activation_1', ParameterType.CATEGORICAL, [model.Activation.__members__]),
    ('activation_2', ParameterType.CATEGORICAL, [model.Activation.__members__]),
    ('activation_3', ParameterType.CATEGORICAL, [model.Activation.__members__]),
    ('num_residual_units_1', ParameterType.DISCRETE, [1, 2, 4]),
    ('num_residual_units_2', ParameterType.DISCRETE, [1, 2, 4]),
    ('num_residual_units_3', ParameterType.DISCRETE, [1, 2, 4]),
    ('k', ParameterType.DISCRETE, [10]),
    ('n_conv_layers_1', ParameterType.DISCRETE, [1, 2]),
    ('n_conv_layers_2', ParameterType.DISCRETE, [1, 2]),
    ('n_conv_layers_3', ParameterType.DISCRETE, [1, 2]),
    ('dropout_1', ParameterType.DISCRETE, [0.0, 0.5]),
    ('dropout_2', ParameterType.DISCRETE, [0.0, 0.5]),
    ('dropout_3', ParameterType.DISCRETE, [0.0, 0.5]),

    # Misc.
    ('batch_size', ParameterType.DISCRETE, [16, 32]),
    ('num_epochs', ParameterType.DISCRETE, [20]),
    ('replicate', ParameterType.DISCRETE, [0, 1, 2, 3]),
]


FIRST = [
    # Optimization.
    ('optimizer', ParameterType.CATEGORICAL, [model.Optimizer.MOMENTUM]),
    ('initial_lr', ParameterType.DISCRETE, [0.05, 0.1]),
    ('lr_decay', ParameterType.CATEGORICAL, [model.LRDecaySchedule.COSINE]),
    ('weight_decay', ParameterType.DISCRETE, [0.0005]),
    ('momentum', ParameterType.DISCRETE, [0.9]),
    ('use_nesterov', ParameterType.BOOLEAN, [False]),

    # Architecture.
    ('n_filters_1', ParameterType.DISCRETE, [32, 64]),
    ('n_filters_2', ParameterType.DISCRETE, [32, 64]),
    ('n_filters_3', ParameterType.DISCRETE, [32, 64]),
    ('stride_1', ParameterType.DISCRETE, [1, 2]),
    ('stride_2', ParameterType.DISCRETE, [1, 2]),
    ('stride_3', ParameterType.DISCRETE, [1, 2]),
    ('depthwise', ParameterType.BOOLEAN, [False]),
    ('activation_1', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_2', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('activation_3', ParameterType.CATEGORICAL, [model.Activation.RELU]),
    ('num_residual_units_1', ParameterType.DISCRETE, [3, 4]),
    ('num_residual_units_2', ParameterType.DISCRETE, [3, 4]),
    ('num_residual_units_3', ParameterType.DISCRETE, [3, 4]),
    ('k', ParameterType.DISCRETE, [10]),
    ('n_conv_layers_1', ParameterType.DISCRETE, [2]),
    ('n_conv_layers_2', ParameterType.DISCRETE, [2]),
    ('n_conv_layers_3', ParameterType.DISCRETE, [2]),
    ('dropout_1', ParameterType.DISCRETE, [0.0]),
    ('dropout_2', ParameterType.DISCRETE, [0.0]),
    ('dropout_3', ParameterType.DISCRETE, [0.0]),

    # Misc.
    ('batch_size', ParameterType.DISCRETE, [128]),
    ('num_epochs', ParameterType.DISCRETE, [200]),
    ('replicate', ParameterType.DISCRETE, [0]),
]