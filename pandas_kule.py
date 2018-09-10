# Standard imports 
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs
from sklearn.externals import joblib
# TTXPheno
from TTXPheno.Tools.user import plot_directory, tmp_directory

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Use the small dataset')
argParser.add_argument('--data', action='store',default='data.h5')
argParser.add_argument('--data_version', action='store',default='v2',help='Version of the data to be used')
argParser.add_argument('--log_plot', action='store_true',help='Use a logarithmic plot')
argParser.add_argument('--save', action='store_true', help='Write the trained BDT to a file')
argParser.add_argument('--criterion', action='store', default='kule', nargs='?', choices=['gini', 'kule', 'entropy'] , help="Select the Criterion to be used")
argParser.add_argument('--export', action='store_true', help="Export the trainded tree as graphviz dot")
argParser.add_argument('--no_plot', action='store_true', help="Don't generate a plot")
argParser.add_argument('--max_depth', action='store', default=2, type=int,nargs='?',  help="The max depth argument, which is given to the DecisionTreeClassifier")
argParser.add_argument('--n_estimators', action='store', default=200, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")

args = argParser.parse_args()

#Set the version of the script
vversion = 'v7'

#set criterion, you can choose from (gini, kule, entropy, hellinger)

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)


from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

version = vversion
if args.small:
    args.data_version += '_small'
    version += '_small'
if args.log_plot:
    version += '_log'
version += '_' + args.criterion
#find directory
input_directory = os.path.join(tmp_directory, args.data_version)
logger.debug('Import data from %s', input_directory)

#Create the tree
if args.criterion == 'kule':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion=kldc)
elif args.criterion == 'gini':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='gini')
elif args.criterion == 'entropy':    
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='entropy')
else:
    assert False, "You choose the wrong Classifier"
# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(dt,
                         algorithm= args.boost_algorithm,
                     n_estimators= args.n_estimators)

#read data from file
df = pd.read_hdf(os.path.join(input_directory, args.data))

X1 = np.array(df[['genZ_pt/F','genZ_cosThetaStar/F']])
X = np.concatenate((X1,X1))

y0 = np.zeros(len(X1))
y1 = np.ones(len(X1))
y = np.concatenate((y0,y1))

w0 = np.array(df['sm_weight'])
sm_weight_mean = np.mean(w0)
w0 /= sm_weight_mean
w1 = np.array(df['bsm_weight'])
bsm_weight_mean = np.mean(w1)
w1 /= bsm_weight_mean
w = np.concatenate((w0,w1))

weight_mean = np.mean([sm_weight_mean, bsm_weight_mean])
weight_mean_array = np.full([len(w0)], weight_mean)

logger.info('Mean of sm_weights: %f, mean of bsm_weights: %f',sm_weight_mean, bsm_weight_mean  )

#train
bdt.fit(X, y, w)



#get the directory
output_dir = os.path.join(tmp_directory, args.data_version)
if not os.path.exists(output_dir):
    os.makedirs( output_dir)

#save to file
if args.save:
    logger.info("Save the trained plot to %s, it uses the %s criterion", output_dir, args.criterion)
    joblib.dump(bdt,  os.path.join(output_dir,"bdt-trainingdata-" + version)) 
    logger.info("Dumped the tree to %s",  os.path.join(output_dir,"bdt-trainingdata-" + version))

#export the tree
if args.export:
    export_graphviz(bdt, out_file= os.path.join(output_dir, version + "-tree.dot"))
    logger.info("Exported the tree as .dot to %s", os.path.join(output_dir, version + "-tree.dot")) 

print('distance score: ', bdt.score(X, y))

reached_score = bdt.score(X,y,w)
#calculate the score
logger.info('Reached a score of %f',  reached_score)

if args.no_plot:
    raise SystemExit

#get the output directory
output_directory = os.path.join(plot_directory,'Kullback-Leibner-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)

plot_colors = "brk"
plot_step = 0.5
class_names = ["SM","BSM","Event"]

plt.figure(figsize=(12, 13))

#Plot the decision boundaries
plt.subplot(224)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

#Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])

#fill the plot with gradient color
Z = bdt.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.coolwarm)
#cs = plt.contour(xx, yy, Z, cmap=plt.cm.coolwarm)
#cs = plt.contour(xx, yy, Z, linewidths=0.75, colors='k') #draw edges
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
plt.axis("tight")
#Plot the training points
for i, n, c in zip(range(2), class_names[:2], plot_colors[:2]):
    #idx = np.intersect1d(np.where(y == i),np.where(w > w_min))
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.coolwarm,
                s=20, edgecolor='k',
                label="Class %s" % n,
                marker=".")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('p_T(T) (GeV)')
plt.ylabel('Cos(Theta*)')
plt.title('Decision Boundary')

# Plot the two-class decision scores
#twoclass_output = bdt.decision_function(X[np.where(w> w_min)])

#Plot the Histogramm for the number of Events over genZ_p..
plot_weights = [w0,w1, weight_mean_array]

plt.subplot(221)
plot_range = (df['genZ_pt/F'].min(), df['genZ_pt/F'].max())
for i, n, c in zip(range(3), class_names, plot_colors):
    plt.hist(df['genZ_pt/F'],
        bins=50, 
        range=plot_range, 
        weights=plot_weights[i],
        facecolor=c,
        label='%s Data' % n,
        alpha=.5,
        edgecolor='k',
        log=args.log_plot)
plt.ylabel('Number of Events')
plt.xlabel('p_T(Z) (GeV)')
plt.title('Weighted p_T(Z)')
plt.legend(loc='upper right')

#prints the predicted value
#print bdt.decision_function([[400.0,0.0]])

#prints the predicted class
#print bdt.predict([[400.0,0.0]])

#Plot the Feature Importance..

#Plot the decision diagram
score  = bdt.decision_function(X)
#now, we weight our score
#plot_range = (min(np.amin(score[:len(score)/2]*w0),np.amin(score[len(score)/2:]*w1)  ), 
 #       max(np.amax(score[:len(score)/2]*w0),np.amax(score[len(score)/2:]*w1   )))
plot_range = (score.min(), score.max())
plt.subplot(222)
for i, n, c in zip(range(2), class_names[:2], plot_colors[:2]):
    #plt.hist(twoclass_output[y[np.where(w> w_min)] == i],
    plt.hist(score[i*len(X)/2:(i+1)*len(X)/2] ,
             bins=10,
             range=plot_range,
             facecolor=c,
             weights=plot_weights[i],
             label='Class %s' % n,
            # label='Score',
             alpha=.5,
             edgecolor='k')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2 * 1.2))
    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score \n ' + str(reached_score))
    plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)


#Draw the same plots with Theta Star

plt.subplot(223)
plot_range = (df['genZ_cosThetaStar/F'].min(), df['genZ_cosThetaStar/F'].max())
for i, n, c in zip(range(3), class_names, plot_colors):
    plt.hist(df['genZ_cosThetaStar/F'],
        bins=50, 
        range=plot_range, 
        weights=plot_weights[i],
        facecolor=c,
        label='%s Data' % n,
        alpha=.5,
        edgecolor='k',
        log=args.log_plot)
plt.ylabel('Number of Events')
plt.xlabel('cos(Theta)')
plt.title('Weighted cos(Theta)')
plt.legend(loc='upper right')


#save the plot
plt.savefig(os.path.join( output_directory,  'pandas-ttz-' + version + '.png'))
