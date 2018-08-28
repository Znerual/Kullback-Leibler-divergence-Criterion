
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

cdef class GiniB(ClassificationCriterion):
    r"""Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end] using the Gini criterion."""


        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        #with gil:
          #print "Node impurity 632"
        for k in range(self.n_outputs):
            sq_count = 0.0
            for c in range(n_classes[k]):
                #print "Klasse c(" + str(c) + ")"
                count_k = sum_total[c]
                #print "count_k = sum_total(" + str(count_k) + ") an der Stelle c"
                sq_count += count_k * count_k
            #print "gini(" + str(gini) + ") +=  1.0 - sq_count(" + str(sq_count) + ") / self.weighted_n_node_samples **2 (" + str(self.weighted_n_node_samples *
            #                          self.weighted_n_node_samples) + ")"
            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)



            sum_total += self.sum_stride
          #print "Gini/self.n_outputs final: " + str(gini /self.n_outputs)
        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : DTYPE_t
            The memory address to save the impurity of the left node to
        impurity_right : DTYPE_t
            The memory address to save the impurity of the right node to
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class KullbackLeibnerCriterion(ClassificationCriterion):

  cdef double node_impurity(self) nogil:
    cdef SIZE_t* n_classes = self.n_classes
    cdef double* sum_total = self.sum_total
    cdef double kule = 0.0
    cdef double count_k
    cdef double log_arg
    cdef SIZE_t c
    with gil:
      assert self.n_outputs == 1, "Only one Output with Kullback-Leibner Criterion"
      for c in range(n_classes[0]):
          print "Klasse c(" + str(c) +")"
          count_k = sum_total[c]
          print "count_k(" + str(count_k) +")"
          #assert ((1.0 - count_k) != 0), "Division by zero,46"
          log_arg = (count_k) / (1.0 - count_k)
          print "log_arg(" + str(log_arg) + ")"
          print "Kule(" +str(kule)+ ") + count_k * np.log([tmp_div])(" + str(np.log([log_arg])) + ")"
          kule += (count_k / 2.0) - (count_k / 4.0 * np.log([log_arg]))

      sum_total += self.sum_stride
      print "Final kule(" + str(kule) + ")"

    return kule

  cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:

    cdef SIZE_t* n_classes = self.n_classes
    cdef double* sum_left = self.sum_left
    cdef double* sum_right = self.sum_right
    cdef double kule_left = 0.0
    cdef double kule_right = 0.0
    cdef double log_arg
    cdef double count_k
    cdef SIZE_t c

    with gil:
      assert self.n_outputs == 1, "Only one Output with Kullback-Leibner Criterion"


      tmp_div = 0.0

      for c in range(n_classes[0]):
          count_k = sum_left[c]
          log_arg = (count_k) / (1.0 - count_k)
          with gil:
            kule_left += (count_k / 2.0) - (count_k / 4.0 * np.log([log_arg]))

          count_k = sum_right[c]
          log_arg = (count_k) / (1.0 - count_k)
          with gil:
            kule_right += (count_k / 2.0) - (count_k / 4.0 * np.log([log_arg]))

      sum_left += self.sum_stride
      sum_right += self.sum_stride

      impurity_left[0] = kule_left
      impurity_right[0] = kule_right
