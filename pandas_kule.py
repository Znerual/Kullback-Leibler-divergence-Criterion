# Standard imports 
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs

# TTXPheno
from TTXPheno.Tools.user import plot_directory, tmp_directory

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Use the small dataset')
argParser.add_argument('--version', action='store',default='v1')
argParser.add_argument('--data', action='store',default='data.h5')
args = argParser.parse_args()

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)


from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

if args.small:
    args.version += '_small'
#find directory
input_directory = os.path.join(tmp_directory, args.version)
#Create the tree
dt = DecisionTreeClassifier(max_depth=2, criterion=kldc)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(dt,
                         algorithm="SAMME",
                     n_estimators=200)

#read data from file
df = pd.read_hdf(os.path.join(input_directory, args.data))

X1 = np.array(df[['genZ_pt/F','genZ_cosThetaStar/F']])
X = np.concatenate((X1,X1))

y0 = np.zeros(len(X1))
y1 = np.ones(len(X1))
y = np.concatenate((y0,y1))

w0 = np.array(df['sm_weight'])
w1 = np.array(df['bsm_weight'])
w = np.concatenate((w0,w1))

#train
bdt.fit(X, y, w)

#from sklearn.ensemble import RandomForestClassifier
#bdt = RandomForestClassifier(criterion=kldc, max_depth=2, n_estimators=100)
#bdt.fit(X, y)

print('distance score: ', bdt.score(X, y))

plot_colors = "br"
plot_step = 0.5
class_names = ["SM","BSM"]

plt.figure(figsize=(10, 5))

# Plot the decision boundaries
#plt.subplot(121)
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                     np.arange(y_min, y_max, plot_step))
#
#Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#plt.axis("tight")

# Plot the training points
#for i, n, c in zip(range(2), class_names, plot_colors):
#    idx = np.intersect1d(np.where(y == i),np.where(w > w_min))
#    #idx = np.where(y == i)
#    plt.scatter(X[idx, 0], X[idx, 1],
#                c=c, cmap=plt.cm.Paired,
#                s=20, edgecolor='k',
#                label="Class %s" % n,
#                marker=".")
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#plt.legend(loc='upper right')
#plt.xlabel('P')
#plt.ylabel('Cos(Theta*)')
#plt.title('Decision Boundary')

# Plot the two-class decision scores
#twoclass_output = bdt.decision_function(X[np.where(w> w_min)])

#Plot the Histogramm for the number of Events over genZ_p..
plot_weights = [w0,w1]
plt.subplot(121)
plot_range = (np.amin(X), np.amax(X))
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(df['genZ_pt/F'],
        bins=50, 
        range=plot_range, 
        weights=plot_weights[i],
        facecolor=c,
        label='%s Data' % n,
        alpha=.5,
        edgecolor='k',
        log=True)
plt.ylabel('Number of Events')
plt.xlabel('p_T(Z) (GeV)')
plt.title('Weighted p_T(Z)')
plt.legend(loc='upper right')

#Plot the decision diagram
twoclass_output = bdt.decision_function(X)

plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    #plt.hist(twoclass_output[y[np.where(w> w_min)] == i],
    plt.hist(twoclass_output,
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

#get the output directory
output_directory = os.path.join(plot_directory,'Kullback-Leibner-Plots', argParser.prog.split('.')[0])
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#save the plot
plt.savefig(os.path.join( output_directory,'pandas-ttz' + args.version + '.png'))
