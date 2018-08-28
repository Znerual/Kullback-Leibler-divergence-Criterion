
# Author: Laurenz Ruzicka
# Base on the work from Evgeni Dubov <evgeni.dubov@gmail.com>
#
# License: MIT

from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t

import numpy as np
cdef double INFINITY = np.inf

from libc.math cimport sqrt, pow
from libc.math cimport abs


cdef class KullbackLeibnerCriterion(ClassificationCriterion):

    cdef double node_impurity(self) nogil:
      cdef SIZE_t* n_classes = self.n_classes
      cdef double* sum_total = self.sum_total
      cdef double kule = 0.0
      cdef double count_k
      cdef double tmp_div
      cdef SIZE_t c
      with gil:
        assert self.n_outputs == 1, "Only one Output with Kullback-Leibner Criterion"
      for c in range(n_classes[0]):
          count_k = sum_total[c]
          tmp_div = count_k / (1.0 - count_k)
          kule += count_k * np.log([tmp_div])
      sum_total += self.sum_stride

      return kule

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:

      cdef SIZE_t* n_classes = self.n_classes
      cdef double* sum_left = self.sum_left
      cdef double* sum_right = self.sum_right
      cdef double kule_left = 0.0
      cdef double kule_right = 0.0
      cdef double tmp_div
      cdef double count_k
      cdef SIZE_t c

      with gil:
        assert self.n_outputs == 1, "Only one Output with Kullback-Leibner Criterion"


        tmp_div = 0.0

        for c in range(n_classes[0]):
            count_k = sum_left[c]
            tmp_div = count_k / (1.0 - count_k)
            kule_left += count_k * np.log([tmp_div])

            count_k = sum_right[c]
            tmp_div = count_k / (1.0 - count_k)
            kule_right += count_k * np.log([tmp_div])

        sum_left += self.sum_stride
        sum_right += self.sum_stride

        impurity_left[0] = kule_left
        impurity_right[0] = kule_right
