from keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as scc
from tensorflow.keras.datasets import mnist
from spectraltools import Spectral, spectral_pruning, spectral_pretrain

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.

inputs = Input(shape=(28, 28,))
x = Flatten()(inputs)
y = Spectral(200, activation='relu', name='Spec_a', use_bias=False)(x)
y = Spectral(300, activation='relu', use_bias=False, name='Spec_b')(y)
y = Spectral(100, activation='relu', name='Spec_c')(y)

x = Spectral(200, activation='relu', name='Spec1', use_bias=False)(x)
x = Dense(300, activation='relu', use_bias=False, name='Dense2')(x)
x = Spectral(100, activation='relu', name='Spec3')(x)

z = Average()([x, y])
outputs = Dense(10, activation="softmax")(z)

model = Model(inputs=inputs, outputs=outputs, name="branched")
model.get_layer('Spec3').is_base_trainable = False
model.compile(optimizer=Adam(1E-3), loss=scc(from_logits=False), metrics=["accuracy"])
ev_dict = dict(x=x_train, y=y_train, batch_size=300)
tr_dict = dict(x=x_train, y=y_train, validation_split=0.2, batch_size=300, epochs=1, verbose=0)

minimal = spectral_pretrain(model, fit_dictionary=tr_dict, eval_dictionary=ev_dict, max_delta=40)

minimal.summary()
fitout = model.fit(x_train, y_train, validation_split=0.2, batch_size=300, epochs=1, verbose=0)
evalout = model.evaluate(x_train, y_train, batch_size=300)

#
# new = spectral_pruning(model, 50)
# new.evaluate(x_train, y_train, batch_size=300)
# new.summary()
