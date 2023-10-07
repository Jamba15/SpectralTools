from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as scc
from tensorflow.keras.datasets import mnist
from spectraltools.spectraldense import Spectral

# Import the L2 regularization
from tensorflow.keras.regularizers import l2

# Dataset and model creation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

spectral_configuration = {'activation': 'relu',
                          'use_bias': False,
                          'base_regularizer': l2(1E-3),
                          'diag_regularizer': l2(5E-3)}

inputs = Input(shape=(28, 28,))
x = Flatten()(inputs)
y = Spectral(200,  **spectral_configuration, name='Spec1')(x)
y = Spectral(300,  **spectral_configuration, name='Spec2')(y)
outputs = Dense(10, activation="softmax", name='LastDense')(y)

model = Model(inputs=inputs, outputs=outputs, name="branched")

model.compile(optimizer=Adam(1E-3),
              loss=scc(from_logits=False),
              metrics=["accuracy"])

model.fit(x_train, y_train,
          validation_split=0.2,
          batch_size=300,
          epochs=1,
          verbose=1)

#%%
model.evaluate(x_test, y_test, batch_size=300)
from spectraltools import metric_based_pruning
from spectraltools.spectralprune import original_model, prune_percentile, metric_based_pruning
import numpy as np

# Reset the model to the unpruned state: the mask are all set to None
original_model(model)
print(f'Baseline accuracy: {model.evaluate(x_test, y_test, batch_size=300)[1]:.3f}')
# Cycle through the spectral layers and count the number of active nodes
compile_dictionary = dict(optimizer=Adam(1E-3),
                            loss=scc(from_logits=False),
                            metrics=["accuracy"])
#
# for lay in new.layers:
#     if hasattr(lay, 'diag_end_mask'):
#         print(f'Layer {lay.name} has {np.count_nonzero(lay.diag_end_mask)} active nodes')

new = metric_based_pruning(model,
                     eval_dictionary=dict(x=x_train, y=y_train, batch_size=200),
                     compile_dictionary=compile_dictionary,
                     compare_metric='accuracy',
                     max_delta_percent=3)

new.evaluate(x_test, y_test, batch_size=300)
