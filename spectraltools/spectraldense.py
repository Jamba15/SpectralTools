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
import numpy as np


@keras_export('keras.layers.Spectral')
class Spectral(Layer):
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
        # Regularizers
        self.base_regularizer = regularizers.get(base_regularizer)
        self.diag_regularizer = regularizers.get(diag_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # Constraint
        self.base_constraint = constraints.get(base_constraint)
        self.diag_constraint = constraints.get(diag_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # Mask
        self.diag_end_mask = eigenvalue_mask

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
        if self.diag_end_mask is not None:
            diag_end = self.diag_end * self.diag_end_mask
            bias = self.bias * self.diag_end_mask
        else:
            diag_end = self.diag_end
            bias = self.bias

        kernel = mul(self.base, self.diag_start - diag_end)
        outputs = matmul(a=inputs, b=kernel)

        if self.use_bias:
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

    def conditions(self,
                   cut_off):

        if np.all(self.diag_start.numpy() == 0):
            start_cond: np.ndarray = abs(self.diag_start.numpy()) >= -1
        else:
            start_cond: np.ndarray = abs(self.diag_start.numpy()) >= cut_off

        end_cond: np.ndarray = abs(self.diag_end.numpy()) >= cut_off

        return {"diag_start": start_cond.reshape((-1)),
                "diag_end": end_cond.reshape((-1))}

    def mask_diag_end(self,
                      cut_off):
        """
        This function sets to zero the diag_end that are below the cut_off changing the diag_end_mask. The mask will be
        initialized to all ones and only the values that are below the cut_off will be set to zero.
        It will be used in the pruning process.
        :param cut_off: The cut_off value
        :return: None
        """
        condition_dictionary = self.conditions(cut_off)
        self.diag_end_mask = np.zeros(shape=self.diag_end.shape)
        self.diag_end_mask[0, condition_dictionary["diag_end"]] = 1


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
