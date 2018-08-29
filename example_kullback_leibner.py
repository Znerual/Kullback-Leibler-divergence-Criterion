import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs


#Dataset creation
X1,y1 = make_gaussian_quantiles(mean=(2, 2), cov=1.5, n_samples=800, n_features=2, n_classes=1, random_state=1)
X2,y2 = make_gaussian_quantiles(cov=1.2, n_samples=600, n_features=2, n_classes=1, random_state=1)

X12 = np.concatenate((X1,X2))
X = np.concatenate((X12,X12))
n_samle_size = len(X12)

# Generating labels
y1 = np.zeros(n_samle_size)
y2 = np.ones(n_samle_size)
y = np.concatenate((y1,y2))

#Generating weights
n_alt_hyp_size = len(X1) #BSM
n_hyp_size = len(X2) #SM

#in case of y=0, SM
w_alt_hyp = np.zeros(n_alt_hyp_size)
w_alt_hyp += (0.01 / n_alt_hyp_size)
w_hyp = np.ones(n_hyp_size)
w_hyp /= n_hyp_size
w0 = np.concatenate((w_alt_hyp, w_hyp))

#in case of y=1, BSM
w_alt_hyp = np.ones(n_alt_hyp_size)
w_alt_hyp /= n_alt_hyp_size
w_hyp = np.zeros(n_hyp_size)
w_hyp += (0.01 / n_hyp_size)
w1 = np.concatenate((w_alt_hyp, w_hyp))

#final weights
w = np.concatenate((w0,w1))

from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

#dt = DecisionTreeClassifier(max_depth=1, criterion=bdt)
dt = DecisionTreeClassifier(max_depth=1, criterion=kldc)
#dt = DecisionTreeClassifier(max_depth=1, criterion=hdc)
#dt = DecisionTreeClassifier(max_depth=1, criterion='gini')
#dt = DecisionTreeClassifier(max_depth=1, criterion='entropy')
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(dt,
                         algorithm="SAMME",
                     n_estimators=100)
bdt.fit(X, y, w)

#from sklearn.ensemble import RandomForestClassifier
#bdt = RandomForestClassifier(criterion=kldc, max_depth=2, n_estimators=100)
#bdt.fit(X, y)

print('distance score: ', bdt.score(X, y))

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
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolor='k',
                label="Class %s" % n,
                marker=".")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X)
#twoclass_output = bdt.decision_path(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
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
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
