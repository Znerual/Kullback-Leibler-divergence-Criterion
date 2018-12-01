# Standard imports 
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import ROOT
import time

# sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# TTXPheno
from TTXPheno.Tools.user import plot_directory, tmp_directory

#Kullback Leibler Divergenz
from criterion import KullbackLeibler, Gini

#start the timer
start = time.time()

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Use the small dataset')
argParser.add_argument('--data', action='store',default='data.h5')
argParser.add_argument('--data_version', action='store',default='v2',help='Version of the data to be used')
argParser.add_argument('--max_depth', action='store', default=2, type=int,nargs='?',  help="The max depth argument, which is given to the DecisionTreeClassifier")
argParser.add_argument('--est_num', action='store', default=40, type=int,nargs='?',  help="The number of steps between the start and end of n_estimators")
argParser.add_argument('--bin_num', action='store', default=12, type=int,nargs='?',  help="The number of bins for the histograms ")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")
argParser.add_argument('--swap_hypothesis', action='store_true', help="Chance the Target Labels of SM and BSM Hypothesis")
argParser.add_argument('--random_state', action='store', default=0, type=int,nargs='?',  help="The random state, which is given to the train_test_split method")
args = argParser.parse_args()

#Set the version of the script
vversion = 'v3'

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)

#Kule algorithm
from kullback_leibler_divergence_criterion import KullbackLeiblerCriterion
kldc = KullbackLeiblerCriterion(1, np.array([2], dtype='int64'))

#setting up the file save name
version = vversion
if args.small:
    args.data_version += '_small'
    version += '_small'
if args.swap_hypothesis:
    version += '_swap'
version += '_binNum' + str(args.bin_num) +  '_maxDepth' + str(args.max_depth) +  '_EstNum' + str(args.est_num) + '_BoostAlg'  + str(args.boost_algorithm) + '_RandState' + str(args.random_state)

#find directory
input_directory = os.path.join(tmp_directory, args.data_version)
logger.debug('Import data from %s', input_directory)

#read data from file
df = pd.read_hdf(os.path.join(input_directory, args.data))
X1 = np.array(df[['genZ_pt/F','genZ_cosThetaStar/F']])
X = np.concatenate((X1,X1))
    #generate targets
y0 = np.zeros(len(X1))
y1 = np.ones(len(X1))
y = np.concatenate((y0,y1))
    #read weights, normalize weights, used the mean to scale it before
w0 = np.array(df['sm_weight'])
sm_weight_sum = np.sum(w0)
w0 /= sm_weight_sum
w1 = np.array(df['bsm_weight'])
bsm_weight_sum = np.sum(w1)
w1 /= bsm_weight_sum
if args.swap_hypothesis:
    w = np.concatenate((w1,w0))
else:
    w = np.concatenate((w0,w1))

logger.info('Sum of sm_weights: %f, Sum of bsm_weights: %f',sm_weight_sum, bsm_weight_sum  )

#split the data into validation and trainings set
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w,test_size= 0.5, random_state=0)

#caluclate the sum of weights
k_w_test_sum_sm = 0.0
k_w_test_sum_bsm = 0.0
k_w_train_sum_sm = 0.0
k_w_train_sum_bsm = 0.0

#caluclate the sum of weights
g_w_test_sum_sm = 0.0
g_w_test_sum_bsm = 0.0
g_w_train_sum_sm = 0.0
g_w_train_sum_bsm = 0.0

#Create the tree
dt_k = DecisionTreeClassifier(max_depth= args.max_depth, criterion=kldc)
dt_g = DecisionTreeClassifier(max_depth= args.max_depth, criterion='gini')

# Create and fit an AdaBoosted decision tree
bdt_k = AdaBoostClassifier(dt_k, algorithm= args.boost_algorithm,n_estimators= args.est_num)
bdt_k.fit(X_train, y_train, w_train)

# Create and fit an AdaBoosted decision tree
bdt_g = AdaBoostClassifier(dt_g, algorithm= args.boost_algorithm,n_estimators= args.est_num)
bdt_g.fit(X_train, y_train, w_train)

#setup the decision functions, which will be used by the histograms as well
k_test_decision_function = bdt_k.decision_function(X_test)
k_train_decision_function = bdt_k.decision_function(X_train)

#setup the decision functions, which will be used by the histograms as well
g_test_decision_function = bdt_g.decision_function(X_test)
g_train_decision_function = bdt_g.decision_function(X_train)

ende_training = time.time()
logger.info('Time to train the tree ' +  '{:5.3f}s'.format(ende_training-start))

#get the directory for data
output_dir = os.path.join(tmp_directory, args.data_version)
if not os.path.exists(output_dir):
    os.makedirs( output_dir)

#get the output directory for plots
output_directory = os.path.join(plot_directory,'Kullback-Leibler-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)

#setup the historgramms
k_h_dis_train_SM = ROOT.TH1D("kdis_train_sm", "Discriminator", args.bin_num, -1, 1)
k_h_dis_train_BSM = ROOT.TH1D("kdis_train_bsm", "Discriminator", args.bin_num, -1, 1)
k_h_dis_test_SM = ROOT.TH1D("kdis_test_sm", "Discriminator", args.bin_num, -1, 1)
k_h_dis_test_BSM = ROOT.TH1D("kdis_test_bsm", "Discriminator", args.bin_num, -1, 1)

#setup the historgramms
g_h_dis_train_SM = ROOT.TH1D("gdis_train_sm", "Discriminator", args.bin_num, -1, 1)
g_h_dis_train_BSM = ROOT.TH1D("gdis_train_bsm", "Discriminator", args.bin_num, -1, 1)
g_h_dis_test_SM = ROOT.TH1D("gdis_test_sm", "Discriminator", args.bin_num, -1, 1)
g_h_dis_test_BSM = ROOT.TH1D("gdis_test_bsm", "Discriminator", args.bin_num, -1, 1)

#set the error calculationsmethod
ROOT.TH1.SetDefaultSumw2()

logger.info('Zeit bis vor der Loop: ' + '{:5.3f}s'.format(time.time()-start))

#loop over the feature vektor to fill the histogramms (test data)
for i in xrange(len(X_test)):
    if y_test[i] == 0:
        k_h_dis_test_SM.Fill(k_test_decision_function[i],w_test[i])
        g_h_dis_test_SM.Fill(g_test_decision_function[i],w_test[i])
    else:
        k_h_dis_test_BSM.Fill(k_test_decision_function[i],w_test[i])
        g_h_dis_test_BSM.Fill(g_test_decision_function[i],w_test[i])

#fill with trainings data
for i in xrange(len(X_train)): 
    if y_train[i] == 0:
        k_h_dis_train_SM.Fill(k_train_decision_function[i],w_train[i])
        g_h_dis_train_SM.Fill(g_train_decision_function[i],w_train[i])
    else:
        k_h_dis_train_BSM.Fill(k_train_decision_function[i],w_train[i])
        g_h_dis_train_BSM.Fill(g_train_decision_function[i],w_train[i])

logger.info('Zeit bis vor nach der Loop: ' + '{:5.3f}s'.format(time.time()-start))

#calcuate the yields after fitting
k_w_train_sum_sm = k_h_dis_train_SM.Integral()
k_w_train_sum_bsm = k_h_dis_train_BSM.Integral()
k_w_test_sum_sm = k_h_dis_test_SM.Integral()
k_w_test_sum_bsm = k_h_dis_test_BSM.Integral()

#calcuate the yields after fitting
g_w_train_sum_sm = g_h_dis_train_SM.Integral()
g_w_train_sum_bsm = g_h_dis_train_BSM.Integral()
g_w_test_sum_sm = g_h_dis_test_SM.Integral()
g_w_test_sum_bsm = g_h_dis_test_BSM.Integral()

#normalize the histograms
k_h_dis_train_SM.Scale(1/k_w_train_sum_sm)
k_h_dis_train_BSM.Scale(1/k_w_train_sum_bsm)
k_h_dis_test_SM.Scale(1/k_w_test_sum_sm)
k_h_dis_test_BSM.Scale(1/k_w_test_sum_bsm)

#normalize the histograms
g_h_dis_train_SM.Scale(1/g_w_train_sum_sm)
g_h_dis_train_BSM.Scale(1/g_w_train_sum_bsm)
g_h_dis_test_SM.Scale(1/g_w_test_sum_sm)
g_h_dis_test_BSM.Scale(1/g_w_test_sum_bsm)

#pyplot settings
class_names = ["Kule Training", "Gini Training"]
plot_colors = ["#000cff","#ff0000"]
plt.figure(figsize=(12,12))
plt.title("Kullback Leibler divergence for a left sided cut")

#initialice the Entropy classes
kl = KullbackLeibler(logger)

#Generate the X values
X_disc = range(0,k_h_dis_test_SM.GetNbinsX())
kule_values = []
gini_values = []

#Fill the kule and gini values
for x in X_disc:
    kule_test, k_error_test = kl.kule_div(k_h_dis_test_SM, k_h_dis_test_BSM, x)
    gini_test, g_error_test = kl.kule_div(g_h_dis_test_SM, g_h_dis_test_BSM, x)
    kule_values.append(kule_test)
    gini_values.append(gini_test)
    
plot_range = (min(min(kule_values), min(gini_values), max(max(kule_values), max(gini_values))))

#show the cut plot
plt.plot(X_disc, kule_values, label=class_names[0])
plt.plot(X_disc, gini_values, label=class_names[1])
plt.legend(loc='upper right')
plt.ylim(plot_range)
plt.xlim((0,k_h_dis_train_SM.GetNbinsX() +1))
plt.xlabel('Cut value')
plt.ylabel('Kullback Leibler divergence')
plt.axis("tight")

#save the matlib plot
plt.savefig(os.path.join( output_directory, 'cut' + version + '.png'))

#stop the timer
ende = time.time()
logger.info('Ende des Runs, Laufzeit: ' + '{:5.3f}s'.format(ende-start))
