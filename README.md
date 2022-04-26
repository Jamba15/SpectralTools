# spectraltools
*spectraltools* is a package for spectral training and analysis of fully connected feedforward NN.<br>
According to our test it is well integrated in Tensorflow 2.3 

## Installation
Activate the environment where the package is to be installed.<br>
After downloading the repo, go with the terminal in the folder with "setup.py" and run:
```bash
$ pip install .
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
The function removes, if it can, a certain percentile of the nodes in the spectral layer. At the moment it only prunes 
if the Spectral layer is followed or follows a Dense or Spectral layer. The nodes are removed acording to the eigenvalues
distribution which has been empirically and heuristically proven to be an indicator of node relevance inside the network.
If 2 or more Spectral layers inbounds on 
the same layer their eigenvalues, and therefore their nodes, will NOT be pruned.


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`spectraltools` was created by Lorenzo Giambagli. It is licensed under the terms of the MIT license.

## Credits

`spectraltools` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
