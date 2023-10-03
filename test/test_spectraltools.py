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

inputs = Input(shape=(28, 28,))
x = Flatten()(inputs)
y = Spectral(200, activation='relu', name='Spec1', use_bias=False)(x)
y = Spectral(300, activation='relu', name='Dense1', diag_end_initializer='ones')(y)

x = Spectral(200, activation='relu', name='Spec3', use_bias=False, diag_regularizer=l2(5E-3))(x)
x = Spectral(300, activation='relu', name='Spec5')(x)

z = Average()([x, y])
outputs = Dense(10, activation="softmax")(z)

model = Model(inputs=inputs, outputs=outputs, name="branched")
model.compile(optimizer=Adam(1E-3), loss=scc(from_logits=False), metrics=["accuracy"])

model.fit(x_train, y_train, validation_split=0.2, batch_size=300, epochs=2, verbose=1)
model.evaluate(x_test, y_test, batch_size=300)

from spectraltools.spectralprune import metric_based_pruning
metric_based_pruning(model, dict(x=x_train, y=y_train, batch_size=300),
                      compare_metric='loss',
                      max_delta_percent=10,
                      eval_dictionary=None)