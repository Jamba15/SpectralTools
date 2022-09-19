from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as scc
from tensorflow.keras.datasets import mnist
from spectraltools import Spectral, spectral_pruning, spectral_distiller

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

inputs = Input(shape=(28, 28,))
y = Flatten()(inputs)
y = Spectral(200, activation='relu', name='Spec_a', use_bias=True, diag_regularizer='l2')(y)
outputs = Dense(10, activation="softmax")(y)

model = Model(inputs=inputs,
              outputs=outputs,
              name="branched")

model.compile(optimizer=Adam(1E-3), loss=scc(), metrics=["accuracy"])
tr_dict = dict(x=x_train, y=y_train, validation_split=0.2, batch_size=300, epochs=20, verbose=0)

minimal = spectral_distiller(model, fit_dictionary=tr_dict, percentile=98)

minimal.summary()

# fitout = minimal.fit(x_train,
#                      y_train,
#                      validation_split=0.2,
#                      batch_size=300,
#                      epochs=1,
#                      verbose=0)

evalout = minimal.evaluate(x_train,
                           y_train,
                           batch_size=300)

# new = spectral_pruning(model, 50)
# new.evaluate(x_train, y_train, batch_size=300)
# new.summary()
