
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

from sklearn.tree._utils cimport log

choice = 'kule'

cdef class KullbackLeibnerCriterion(ClassificationCriterion):

    cdef double node_impurity(self) nogil:

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double kule    = 0.0
        cdef double entropy = 0.0
        cdef double gini    = 0.0
        cdef double hellinger = 0.0
        cdef double rho
        cdef double rho_0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        with gil:
          assert self.n_outputs == 1,    "Only one output with Kullback-Leibner Criterion"
          assert self.n_classes[0] == 2, "Only two classes with Kullback-Leibner Criterion"

        for k in range(self.n_outputs):
            # Gini
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

            # Entropy
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

            # kule 
            rho   = sum_total[1]/self.weighted_n_node_samples
            rho_0 = sum_total[0]/self.weighted_n_node_samples # for debugging
            if rho==1:
                kule  = -INFINITY
            elif rho>0:
                kule  = 0.5*rho - 0.25*rho*log(rho/(1-rho))
            else:
                rho=0 

            # Hellinger
            for c in range(n_classes[k]):
                hellinger += 1.0

            # This sum is gloablly relevant!
            sum_total += self.sum_stride


        with gil:

            print "node_impurity: gini %6.4f entropy %6.4f kule %6.4f hellinger %6.4f" %( gini, entropy, kule, hellinger )
            print "  sum_total[0] %6.4f sum_total[1] %6.4f" %( sum_total[0], sum_total[1] )
            print "  weighted_n_node_samples %6.4f" %( self.weighted_n_node_samples )
            print "  rho %6.4f rho_0 %6.4f" %( rho, rho_0 )

            if choice == 'gini':
                return gini / self.n_outputs
            elif choice == 'kule': 
                return kule / self.n_outputs
            elif choice == 'entropy':
                return entropy / self.n_outputs
            elif choice == 'hellinger':
                return hellinger / self.n_outputs
    

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double kule_left = 0.0
        cdef double kule_right = 0.0
        cdef double kule = 0.0
        cdef double rho = 0.0
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double hellinger_left = 0.0
        cdef double hellinger_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double rho_left
        cdef double rho_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        with gil:
          assert self.n_outputs == 1,    "Only one output with Kullback-Leibner Criterion"
          assert self.n_classes[0] == 2, "Only two classes with Kullback-Leibner Criterion"


        for k in range(self.n_outputs):
            # Gini
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

            # Entropy
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

            # kule
            rho_left = sum_left[1]/self.weighted_n_left
            if rho_left == 1:
                kule_left = - INFINITY
            elif rho_left > 0:
                kule_left = 0.5*rho_left - 0.25*rho_left*log(rho_left/(1-rho_left))
            else:
                kule_left = 0.

            rho_right  = sum_right[1]/self.weighted_n_right
            if rho_right == 1:
                kule_right = - INFINITY
            elif rho_right > 0:
                kule_right = 0.5*rho_right - 0.25*rho_right*log(rho_right/(1-rho_right))
            else:
                kule_right = 0.

            # for debugging: compute the mother impurity
            rho  = (sum_left[1] + sum_right[1])/(self.weighted_n_left + self.weighted_n_right)
            if rho == 1:
                kule = - INFINITY
            elif rho > 0:
                kule = 0.5*rho - 0.25*rho*log(rho/(1-rho))
            else:
                kule = 0.

            # Hellinger
            # stop splitting in case reached pure node with 0 samples of second class
            if sum_left[1] + sum_right[1] == 0:
                impurity_left[0] = -INFINITY
                impurity_right[0] = -INFINITY
            else:

                if(sum_left[0] + sum_right[0] > 0):
                    count_k1 = sqrt(sum_left[0] / (sum_left[0] + sum_right[0]))
                if(sum_left[1] + sum_right[1] > 0):
                    count_k2 = sqrt(sum_left[1] / (sum_left[1] + sum_right[1]))

                hellinger_left += pow((count_k1  - count_k2),2)

                if(sum_left[0] + sum_right[0] > 0):
                    count_k1 = sqrt(sum_right[0] / (sum_left[0] + sum_right[0]))
                if(sum_left[1] + sum_right[1] > 0):
                    count_k2 = sqrt(sum_right[1] / (sum_left[1] + sum_right[1]))

                hellinger_right += pow((count_k1  - count_k2),2)

            # Careful! This is a global sum! Can only do once and only at the end of this loop.
            sum_left += self.sum_stride
            sum_right += self.sum_stride

        with gil:
            DeltaKule =  - self.weighted_n_left*kule_left - self.weighted_n_right*kule_right + (self.weighted_n_left+self.weighted_n_right)*kule 
            #print "children_impurity: kule_left %6.4f kule_right %6.4f kule_tot %6.4f DeltaKule %6.4f"%( kule_left, kule_right, kule, DeltaKule) 
            if choice == 'gini':
                impurity_left[0] = gini_left / self.n_outputs
                impurity_right[0] = gini_right / self.n_outputs
            elif choice == 'entropy':
                impurity_left[0] = entropy_left / self.n_outputs
                impurity_right[0] = entropy_right / self.n_outputs
            elif choice == 'kule': 
                impurity_left[0]  = kule_left / self.n_outputs
                impurity_right[0] = kule_right / self.n_outputs
            elif choice == 'hellinger':
                impurity_left[0]  = hellinger_left  / self.n_outputs
                impurity_right[0] = hellinger_right / self.n_outputs
            sum_left += self.sum_stride
            sum_right += self.sum_stride


