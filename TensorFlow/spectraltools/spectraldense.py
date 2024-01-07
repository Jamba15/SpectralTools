try:
    from keras.engine.base_layer import Layer
    from keras import initializers, regularizers, constraints, activations
except ModuleNotFoundError:
    from tensorflow.python.keras.engine.base_layer import Layer
    from tensorflow.python.keras import initializers, regularizers, constraints, activations

from tensorflow.python.util.tf_export import keras_export
from tensorflow import multiply as mul
from tensorflow import reduce_sum
from tensorflow import matmul
from tensorflow.python.framework import tensor_shape
from tensorflow_model_optimization.sparsity.keras import PrunableLayer

import numpy as np

@keras_export('keras.layers.Spectral')
class Spectral(Layer, PrunableLayer):
    def __init__(self,
                 units,
                 activation=None,
                 is_base_trainable=True,
                 is_diag_start_trainable=False,
                 is_diag_end_trainable=True,
                 use_bias=False,
                 eigenvalue_mask=None,
                 base_initializer='GlorotUniform',
                 diag_start_initializer='Zeros',
                 diag_end_initializer='Ones',
                 bias_initializer='Zeros',
                 base_regularizer=None,
                 diag_regularizer=None,
                 diag_regularizer_param=5E-4,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 base_constraint=None,
                 diag_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Spectral, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        # Trainable weights
        self.is_base_trainable = is_base_trainable
        self.is_diag_start_trainable = is_diag_start_trainable
        self.is_diag_end_trainable = is_diag_end_trainable
        self.use_bias = use_bias
        # Initializers
        self.base_initializer = initializers.get(base_initializer),
        self.diag_start_initializer = initializers.get(diag_start_initializer),
        self.diag_end_initializer = initializers.get(diag_end_initializer),
        self.bias_initializer = initializers.get(bias_initializer),
        # Regularizer
        if base_regularizer is not None:
            self.base_regularizer = regularizers.get(base_regularizer)
        else:
            self.base_regularizer = None
        if diag_regularizer == 'l2':
            self.diag_regularizer = regularizers.l2(diag_regularizer_param)
        else:
            # Deal with None
            if diag_regularizer is not None:
                self.diag_regularizer = regularizers.get(diag_regularizer)
            else:
                self.diag_regularizer = None
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # Constraint
        self.base_constraint = constraints.get(base_constraint)
        self.diag_constraint = constraints.get(diag_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # Mask
        self.eigenvalue_mask = eigenvalue_mask

    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        # trainable eigenvector elements matrix
        # \phi_ij
        self.base = self.add_weight(
            name='base',
            shape=(last_dim, self.units),
            initializer=self.base_initializer[0],
            regularizer=self.base_regularizer,
            constraint=self.base_constraint,
            dtype=self.dtype,
            trainable=self.is_base_trainable
        )

        # trainable eigenvalues
        # \lambda_i
        self.diag_end = self.add_weight(
            name='diag_end',
            shape=(1, self.units),
            initializer=self.diag_end_initializer[0],
            regularizer=self.diag_regularizer,
            constraint=self.diag_constraint,
            dtype=self.dtype,
            trainable=self.is_diag_end_trainable
        )

        # \lambda_j
        self.diag_start = self.add_weight(
            name='diag_start',
            shape=(last_dim, 1),
            initializer=self.diag_start_initializer[0],
            regularizer=self.diag_regularizer,
            constraint=self.diag_constraint,
            dtype=self.dtype,
            trainable=self.is_diag_start_trainable
        )

        # bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='spectral_bias',
                shape=(self.units,),
                initializer=self.bias_initializer[0],
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, **kwargs):
        if self.eigenvalue_mask is not None:
            diag_end = self.diag_end * self.eigenvalue_mask
        else:
            diag_end = self.diag_end

        kernel = mul(self.base, self.diag_start - diag_end)
        outputs = matmul(a=inputs, b=kernel)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def direct_space(self):
        """
        Returns the weight matrix in the direct space, namely, the classical weights
        :return: units x input_shape tensor
        """
        return mul(self.base, self.diag_start - self.diag_end).numpy().T

    def get_prunable_weights(self):
        # Return a list of weights to be pruned. Usually the main weights of the layer.
        # Adjust this based on your layer's structure.
        return [self.base, self.diag_end]

    def return_base(self):
        """
        Returns the base matrix in the direct space, namely, the classical weights
        """
        c = self.base.shape[0]
        N = reduce_sum(self.base.shape).numpy()
        phi = np.eye(N)
        phi[c:, :c] = self.base.numpy().T
        return phi

    def return_diag(self):
        """
        Returns the eigenvalues as [start, end]. Start are in relation with the first neurons and end with the last
        of the linear transfer between layer k and k+1
        """
        if self.is_diag_start_trainable and self.is_diag_end_trainable:
            return np.concatenate([self.diag_start.numpy().reshape([-1]), self.diag_end.numpy().reshape([-1])], axis=0)
        elif self.is_diag_start_trainable and not self.is_diag_end_trainable:
            return self.diag_start.numpy().reshape([-1])
        elif not self.is_diag_start_trainable and self.is_diag_end_trainable:
            return self.diag_end.numpy().reshape([-1])



    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'is_base_trainable':
                self.is_base_trainable,
            'is_diag_start_trainable':
                self.is_diag_start_trainable,
            'is_diag_end_trainable':
                self.is_diag_end_trainable,
            'use_bias':
                self.use_bias,
            'base_initializer':
                initializers.serialize(self.base_initializer[0]),
            'diag_start_initializer':
                initializers.serialize(self.diag_start_initializer[0]),
            'diag_end_initializer':
                initializers.serialize(self.diag_end_initializer[0]),
            'bias_initializer':
                initializers.serialize(self.bias_initializer[0]),
            'base_regularizer':
                regularizers.serialize(self.base_regularizer),
            'diag_regularizer':
                regularizers.serialize(self.diag_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'base_constraint':
                constraints.serialize(self.base_constraint),
            'diag_constraint':
                constraints.serialize(self.diag_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)