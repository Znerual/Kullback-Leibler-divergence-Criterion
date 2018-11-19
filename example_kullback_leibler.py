# Standard imports 
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs

#generated data
from gen_datasets import *
#from ttz_dataset import ttz_dataset

#generate the dataset
X,y,w, w_min = gauss_easy(n_samples_bsm=4000, n_samples_sm= 4000)

from kullback_leibler_divergence_criterion import KullbackLeiblerCriterion
kldc = KullbackLeiblerCriterion(1, np.array([2], dtype='int64'))

#Create the tree
dt = DecisionTreeClassifier(max_depth=2, criterion='gini')

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(dt, algorithm="SAMME", n_estimators=200)
bdt.fit(X, y, w)

#calculate the reached score
score = bdt.score(X,y,w)
print('distance score: ', score)

#set the plot settings
plot_colors = "br"
plot_step = 0.02
class_names = "AB"
plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)

#generate the 2D Matrix for plotting the decision funciton on
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

#map the dec function
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#draw the countour and color it
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

#fruther plot settings
plt.axis("tight")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('P')
plt.ylabel('Cos(Theta*)')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X[np.where(w> w_min)])

#get the range
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)

#loop over the two hypothesis (SM, BSM)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y[np.where(w> w_min)] == i], #select only events with weights bigger than EPSILON
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')

#further plot settings
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score \n ' + str(score))
plt.title('Decision Scores')
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)

#save the plot
plt.savefig('test.png')
