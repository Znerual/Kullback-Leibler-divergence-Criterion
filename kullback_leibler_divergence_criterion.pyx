
# Author: Laurenz Ruzicka
# Base on the work from Evgeni Dubov <evgeni.dubov@gmail.com>
#
# License: MIT
#sklearn imports
from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t
#from sklearn.tree._utils cimport log

#default imports
import numpy as np
cdef double INFINITY = np.inf

#math imports
from libc.math cimport sqrt, pow, log
from libc.math cimport abs


cdef class KullbackLeiblerCriterion(ClassificationCriterion):
    cdef double node_impurity(self) nogil:
        
        cdef double* sum_total = self.sum_total
        cdef double kule    = 0.0
        cdef double rho

        # kule
        rho   = sum_total[1]/self.weighted_n_node_samples
        if rho==1:
            kule  = -INFINITY
        elif rho>0:
            kule  = 2.0*rho - rho*log(rho/(1-rho))
        else:
            kule = 0

            # This sum is gloablly relevant! It moves the array Pointer to the next entry
        sum_total += self.sum_stride

        return kule / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double kule_left = 0.0
        cdef double kule_right = 0.0
        cdef double rho_left
        cdef double rho_right
        cdef SIZE_t c



        # kule
        rho_left = sum_left[1]/self.weighted_n_left
        
        if rho_left == 1:
            kule_left = -INFINITY
        elif rho_left > 0:
            kule_left = 2.0*rho_left - rho_left*log(rho_left/(1-rho_left))
        else:
            kule_left = 0.

        rho_right  = sum_right[1]/self.weighted_n_right
        if rho_right == 1:
            kule_right = -INFINITY
        elif rho_right > 0:
            kule_right = 2.0*rho_right - rho_right*log(rho_right/(1-rho_right))
        else:
            kule_right = 0.

        # Careful! This is a global sum! Can only do once and only at the end of this loop.
        sum_left += self.sum_stride
        sum_right += self.sum_stride

        impurity_left[0]  = kule_left / self.n_outputs
        impurity_right[0] = kule_right / self.n_outputs
