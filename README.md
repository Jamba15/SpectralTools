[![PyPI version](https://badge.fury.io/py/spectraltools.svg)](https://badge.fury.io/py/spectraltools)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Python Version](https://img.shields.io/badge/Python-3.10-blue)

# spectraltools
*spectraltools* is a package for spectral training and analysis of fully connected feedforward NN.<br>
According to our test it is well integrated in `Tensorflow 2.10` and older versions up to `Tensorflow 2.3`.

## Installation
Activate the environment where the package is to be installed.<br>
```bash
$ pip install spectraltools
```
# Usage
## Spectral layer
The package contains the spectral fully connected layer that can be imported as follows:
~~~
from spectraltools import Spectral
~~~
It is a representation in the reciprocal space of a fully connected layer.<br>
The layer can be used inside a Tensorflow model and has three main attributes:
~~~python
from spectraltools import Spectral
Spectral(
    units=300,
    activation='relu',
    is_base_trainable=True,
    is_diag_end_trainable=True,
    use_bias=False
)
~~~
In this configuration the layer is a fully connected layer with 300 nodes and ReLU activation function. The layer is equivalent to a dense 
layer with a scalar parameter that multiplies the features. The layer is initialized with a random base and eigenvalues
equal to one (namely the initialization is equivalent to a fully-connected layer).<br>
It implements the operation: <code> output = activation(dot(input, spectral_kernel) + bias) </code> where 
<code>spectral_kernel = dot(base, diag_in) - dot(diag_out, base)</code>.
<code>diag_in</code> and <code>diag_out</code> are the eigenvalues of the adjacency matrix representing the layer and
base are the nontrivial components of its eigenvectors. `bias` is a bias vector created by the layer 
(only applicable if `use_bias` is `True`).<br>
This configuration (where the eigenvectors and the diag_end are trained) is the one suggested and for which the 
pruning function are developed. In the future other configurations support will be added.<br>

### Attributes description
If `is_base_trainable=True` the eigenvectors of the adjacency matrix will be trained. This is equivalent to the
training of all the connections (features). Those are `input_dim x output_dim`
trainable parameters.<br>
`is_input_layer` (default set to `False`) train the first `input_dim` eigenvalues of the matrix and `is_diag_end_trainable` trains the last 
`output_dim` eigenvalues. We recommend to set `is_input_layer` to `True` only for the first layer of the network and 
leave it to False for the other layers. This is because the behaviour of the pruning algorithm has been tested and heuristically proven effective 
only when in this setting.<br>
The total number of trainable parameters is therefore `input_dim x output_dim + input_dim + output_dim`.
If only the eigenvalues are trained the number of free parameters drops but the learning still occurs. A 
suboptimal loss minimum is reached but overfitting is less likely to occur. If also eigenvectors are trained the layer
is, from a training point of view, the same as the Dense.<br>


## Spectral Pruning
The pruning function are tested with **Functional** or **Sequential** models implementing one or more Spectral layers.
Best pruning results are achived when also an L2 regularization is applied to the spectral layer parameters: base and eigenvalues.
There are two ways in which the pruning can be done:
1. **Percentile based Pruning**: the pruning is done according to the eigenvalues distribution of every spectral layer in
the model. The nodes with the smallest eigenvalues magnitude (according to the percentile given) are removed. The percentile of nodes to be removed is passed as
an argument to the function. The compile configuration is needed
It can be called as follows:
```python
from spectraltools import prune_percentile

pruned_model = prune_percentile(model,
                                percentile,
                                percentile_threshold)
```
model: `Sequential` or `Functional` model, employing one or more Spectral layers, that needs to be pruned.
percentile_threshold: the percentile (1-100) of nodes that the model should try to prune. The pruning is done by masking 
the eigenvalues of the spectral layers which is equivalent to set all the corresponding features and biases to 0. <br>
#### Example:
```python
from tensorflow.keras.layers import Dense, Input
from spectraltools import Spectral

inputs = Input(shape=(784,))
x = Dense(100, 
          activation='relu')(inputs)
x = Spectral(80, 
            activation='relu')(x)
```

In this case the prunable nodes will be 80.<br>
The nodes are removed according to the eigenvalues
distribution which has been empirically and heuristically proven to be an indicator of node relevance inside the network.

2. **Metric based Pruning**: the pruning is done according to the impact that the removal of a node has on the loss or another 
metric calculated on the dataset. The nodes with the smallest impact are removed. The impact is calculated by training the model on a validation set.
that is given.
```python
from spectraltools import metric_based_pruning
metric_based_pruning(model,
                     eval_dictionary,
                     compile_dictionary,
                     compare_metric='accuracy',
                     max_delta_percent=10,
                     **kwargs)
```
`model`: the trained model to be pruned.<br>
`eval_dictionary`: the dictionary with the arguments to be passed to the `evaluate` method of the model.<br>
`compile_dictionary`: the dictionary with the arguments to be passed to the `compile` method of the model.<br>
`max_delta_percent`: maximal variation of the given indicator at which break the pruning process.<br>
`compare_metric`: indicator to be used (the corresponding metric name should be used while compiling the model) <br>

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms.

## License

`spectraltools` was created by Lorenzo Giambagli. It is licensed under the terms of the MIT license.

## Credits

`spectraltools` was created with [`cookiecutter`] (https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
