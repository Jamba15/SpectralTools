[![GitHub version](https://badge.fury.io/gh/Jamba15%2FSpectralTools.svg)](https://badge.fury.io/gh/Jamba15%2FSpectralTools)
[![PyPI version](https://badge.fury.io/py/spectraltools.svg)](https://badge.fury.io/py/spectraltools)

# spectraltools
*spectraltools* is a package for spectral training and analysis of fully connected feedforward NN.<br>
According to our test it is well integrated in Tensorflow 2.3 

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
    activation=None,
    is_base_trainable=True,
    is_diag_start_trainable=False,
    is_diag_end_trainable=True,
    use_bias=False
)
~~~
It implements the operation: <code> output = activation(dot(input, spectral_kernel) + bias) </code> where 
<code>spectral_kernel = dot(base, diag_in) - dot(diag_out, base)</code>.
<code>diag_in</code> and <code>diag_out</code> are the eigenvalues of the adjacency matrix representing the layer and
base are the nontrivial components of its eigenvectors. `bias` is a bias vector created by the layer 
(only applicable if `use_bias` is `True`).

### Attributes description
If `is_base_trainable=True` the eigenvectors of the adjacency matrix will be trained. Those are `input_dim x output_dim`
trainable parameters.<br>
is_diag_start_trainable train the first input_dim eigenvalues of the matrix and is_diag_end_trainable trains the last 
output_dim eigenvalues. The total number of trainable parameters is therefore `input_dim x output_dim + input_dim + output_dim`.
If only the eigenvalues are trained the number of free parameters drops but the learning still occurs. A 
suboptimal loss minimum is reached but overfitting is less likely to occur. If also eigenvectors are trained the layer
is, from a training point of view, the same as the Dense.<br>

## Spectral Pruning
The pruning function should work regardless of the model and of hte topology. It can be called as follows:
```python
from spectraltools import spectral_pruning
pruned_model = spectral_pruning(model,
                                percentile)
```
model: `Sequential` or `Functional` model, employing one or more Spectral layers, that needs to be pruned.
percentile: the percentile (1-100) of nodes that the model should try to prune. The prunable nodes are the one with 
trainable eigenvalues.

#### Example:
```python
from tensorflow.keras.layers import Dense, Input
from spectraltools import Spectral

inputs = Input(shape=(784,))
x = Dense(100, 
          activation='relu')(inputs)
x = Spectral(80, 
            is_diag_start_trainable=True,
            is_diag_end_trainable=True,
            activation='relu')(x)
```

In this case the prunable nodes will be 100 (`is_diag_start_trainable=True`) and 80 (`is_diag_end_trainable=True`)<br>
If 2 spectral layers are one next to each other the "end" eigenvalues of the preceding and the "start" fo the following
are both taken into account.
The function removes, if it can, a certain percentile of the nodes in every spectral layer. At the moment it only prunes 
if the Spectral layer is followed or follows a Dense or Spectral layer. The nodes are removed according to the eigenvalues
distribution which has been empirically and heuristically proven to be an indicator of node relevance inside the network.
If two or more Spectral layers inbounds on the same layer, their eigenvalues, and therefore their nodes, will NOT be pruned.

### Spectral Pre-train

This funtion aims at finding the must efficient subnetwork due to random initialization with a pre-training of only 
the eigenvalues inside every spectral layer in the network. *At the moment* all the parameters of the others layers will
Not be modified and therefore every `trainable=True` weight will be trained.<br>
The function trains only the eigenvalues of every spectral layers according to the fit_dictionary passed. 
After that an increasing percentile of the nodes is pruned, until the accuracy or the loss has dropped (or risen) 
by a max_delta percent.

```python
from spectraltools import spectral_pretrain

pruned_model = spectral_pretrain(model, 
                                 fit_dictionary, 
                                 eval_dictionary,
                                 max_delta, 
                                 compare_with='acc' )
```
`model`: the untrained model to be pruned
`fit_dictionary`: the dictionary with the arguments to be passed to the `fit` method of the model.
`eval_dictionary`: the dictionary with the arguments to be passed to the `fit` method of the model.
`max_delta`: maximal variation of the given indicator at which break the pruning process
`compare_with`: indicator to be used: `'loss'` or `'acc'`


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`spectraltools` was created by Lorenzo Giambagli. It is licensed under the terms of the MIT license.

## Credits

`spectraltools` was created with [`cookiecutter`] (https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
