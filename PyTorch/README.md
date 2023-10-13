## Experimental PyTorch implementation

Following the structure in the TensorFlow existing code the PyTorch implementation is in the folder `PyTorch`. 
The main file is `spectraldense.py` containing the class `Spectral`. Different from the TF implementation, this
class only implements the gating of the weights (i.e. the eigenvectors in the spectral formalism) by the eigenvalues.

This allows to use the spectral reparametrization in any model by using the functions in `utils.py`:

- The function `spectral_linear` takes in input a model and reparametrize all the linear layers in the model with the spectral reparametrization.
- The function `spectral_conv2d` takes in input a model and reparametrize all the convolutional layers in the model with the spectral reparametrization.
- The function `spectral_all` takes in input a model and applies the two nethods above in succession.

One can then train the model as usual.

This reparametrization can be then used to effectively prune the model with the function contained in `spectralprune.py` which 
takes in input a model and a pruning rate and globally prunes the eigenvalues that are below a certain percentile.