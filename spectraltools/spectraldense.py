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
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
import numpy as np


@keras_export('keras.layers.Spectral')
class Spectral(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 is_base_trainable=True,
                 is_diag_end_trainable=True,
                 is_diag_start_trainable=False,
                 use_bias=False,
                 diag_end_mask=None,
                 diag_start_mask=None,
                 base_initializer='GlorotUniform',
                 diag_start_initializer='Zeros',
                 diag_end_initializer='Ones',
                 bias_initializer='Zeros',
                 base_regularizer=None,
                 diag_regularizer=None,
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
        self.is_diag_end_trainable = is_diag_end_trainable
        self.is_diag_start_trainable = is_diag_start_trainable
        self.use_bias = use_bias
        # Initializers
        self.base_initializer = initializers.get(base_initializer),
        self.diag_start_initializer = initializers.get(diag_start_initializer),
        self.diag_end_initializer = initializers.get(diag_end_initializer),
        self.bias_initializer = initializers.get(bias_initializer),
        # Regularizers
        self.base_regularizer = regularizers.get(base_regularizer)
        self.diag_regularizer = regularizers.get(diag_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # Constraint
        self.base_constraint = constraints.get(base_constraint)
        self.diag_constraint = constraints.get(diag_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        # Mask for pruning the eigenvalues (the linear features)
        self.diag_end_mask = self.add_weight(
            name='diag_end_mask',
            shape=(1, self.units),
            initializer='ones',
            trainable=False,
            dtype=self.dtype
        )

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
            initializer='zeros',
            regularizer=self.diag_regularizer,
            constraint=self.diag_constraint,
            dtype=self.dtype,
            trainable=self.is_diag_start_trainable
        )

        # bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
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

        diag_end = mul(self.diag_end, self.diag_end_mask)
        kernel = mul(self.base, self.diag_start - diag_end)
        
        outputs = matmul(a=inputs, b=kernel)

        if self.use_bias:
            bias = mul(self.bias, self.diag_end_mask)
            outputs = outputs + bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def direct_space(self):
        """
        Returns the weight matrix in the direct space, namely, the classical weights
        :return: units x input_shape tensor
        """
        return mul(self.base, self.diag_start - self.diag_end).numpy().T

    def return_base(self):
        """
        Returns the base matrix in the direct space, namely, the classical weights
        """
        c = self.base.shape[0]
        N = reduce_sum(self.base.shape).numpy()
        phi = np.eye(N)
        phi[c:, :c] = self.base.numpy().T
        return phi

    def conditions(self,
                   cut_off):
        masking_conditions: np.ndarray = abs(self.diag_end.numpy()) >= cut_off
        return masking_conditions.reshape(-1)

    def mask_diag_end(self,
                      cut_off):
        """
        This function sets to zero the diag_end that are below the cut_off changing the diag_end_mask. The mask will be
        initialized to all ones and only the values that are below the cut_off will be set to zero.
        It will be used in the pruning process.
        :param cut_off: The cut_off value
        :return: None
        """
        masking_conditions = self.conditions(cut_off)
        tmp = np.zeros(shape=self.diag_end.shape)
        tmp[0, masking_conditions] = 1
        self.diag_end_mask.assign(tmp)

    def get_eigenvalues(self, masked=False):
        """
        This function returns the eigenvalues of the layer. Check that at least one between is_diag_end_trainable and
        is_diag_start_trainable is True, otherwise returns an error. If is_diag_end_trainable is True, it returns diag_end. If also
        is_diag_start_trainable is True, it returns diag_start and diag_end as a concatenated vector.
        """
        eigenvalues = {'diag_end': self.diag_end.numpy(), 'diag_start': self.diag_start.numpy()}
        if masked:
            eigenvalues['diag_end'] *= self.diag_end_mask
        return eigenvalues

    # Functions for TensorFlow compatibility
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
