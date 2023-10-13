import torch
import torch.nn as nn

class Spectral(nn.Module):
    """Spectral layer base model
    Base Spectral layer model as presented in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.054312,
    implemented using PyTorch-based code. Here only the multiplication by eigenvalues is implemented.
    ----------
    dim:
        Input size (equal to output size)
    eigval_grad:
        If True, the eigenvalues are trainable
    device:
        Device for training
    dtype:
        Type for the training parameters
    Example
    -------
    model = torch.nn.Sequential(
                            torch.nn.Linear(784, 20),
                            Spectral(20),
                            torch.nn.Sigmoid(),
                            )
    """

    __constants__ = ['dim']
    __name___=['Spectral']
    dim: int

    def __init__(self,
                 dim: int,
                 eigval_grad: bool = True,
                 device=None,
                 dtype=torch.float,
     ):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Spectral, self).__init__()
        self.dim = dim
        self.eigval_grad = eigval_grad

        # Build the model
        # Eigenvalues
        self.eigvals = nn.Parameter(torch.empty(self.dim, **factory_kwargs), requires_grad=self.eigval_grad)

        # Initialize the layer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #nn.init.normal_(self.eigvals)
        nn.init.ones_(self.eigvals)

    def forward(self, x):
        return torch.mul(x, self.eigval)

    def extra_repr(self) -> str:
        s='{dim}, eigval_grad={eigval_grad}'
        return s.format(**self.__dict__)