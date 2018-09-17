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

# TTXPheno
from TTXPheno.Tools.user import plot_directory 

X,y,w, w_min = gauss_easy(n_samples_bsm=400, n_samples_sm= 300)
#X,y,w,w_min = ttz_dataset()
#print X[:2]
#print y[:2]
#print w[:2]
#print w_min

#assert False, "End of test reached"

from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

#Create the tree
dt = DecisionTreeClassifier(max_depth=2, criterion=kldc)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(dt,
                         algorithm="SAMME",
                     n_estimators=200)
bdt.fit(X, y, w)

#from sklearn.ensemble import RandomForestClassifier
#bdt = RandomForestClassifier(criterion=kldc, max_depth=2, n_estimators=100)
#bdt.fit(X, y)
score = bdt.score(X,y,w)
print('distance score: ', score)

#assert False, "End of test"

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

# Plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.intersect1d(np.where(y == i),np.where(w > w_min))
    #idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n,
                marker=".")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('P')
plt.ylabel('Cos(Theta*)')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X[np.where(w> w_min)])

plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y[np.where(w> w_min)] == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score \n ' + str(score))
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)


plt.savefig(os.path.join( plot_directory, 'Kullback-Leibner-Plots','gauss-comparison.png'))
#plt.show()
