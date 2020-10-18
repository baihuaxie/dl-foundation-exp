"""
    Implement line search routines
"""

import torch

class ArmijoLineSearch(object):
    """
    Armijo line search
    """
    def __init__(self, rho=0.8, c=1e-4, loss_fn=None):
        """
        Constructor
        """
        self._rho = rho
        self._c = c
        self.loss_fn = loss_fn

    def search(self, func=None, func_grad=None, input=None, label=None,
               output=None):
        """
        Search routine for Armijo algorithm

        Args:
            func: (nn.Module) current network; to produce forward-pass 
                  on updated input
            func_grad: (tensor) loss gradient w.r.t. input tensor
            input: (tensor) current input tensor
            label: (tensor) labels for computing loss
            output: (tensor) current output tensor
        """
        # initial stepsize
        alpha = 1
        rho = self._rho
        c = self._c

        # loop for Armijo condition
        while self.loss_fn(func(input + alpha*func_grad), label) \
              > output - c*alpha*torch.dot(func_grad, func_grad):
            alpha *= rho

        return alpha
