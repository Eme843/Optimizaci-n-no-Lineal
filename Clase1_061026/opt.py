""" opt.py  """

import numpy as np
import scipy as sp


class opt_unconstrained:

    def __init__(self, f, grad, x0, nombre=""):
        self._f = f
        self._grad = grad
        #self._hess = hess
        self.x0 = np.array(x0, dtype=float)
        self.n = self.x0.size
        self.nombre = nombre
        self.reset()

    def reset(self):
        """ Reinicia las variabñes de conteo de evaluaciones de f y grad_f """
        self.nf = 0  # evaluaciones de f
        self.ng = 0  # evaluaciones de grad_f

    def f(self, x):
        self.nf += 1
        return self._f(x)

    def grad(self, x):
        self.ng += 1
        return self._grad(x)
