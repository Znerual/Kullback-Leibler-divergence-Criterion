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

#Kullback Leibner Divergenz
from kullback_leibner import KullbackLeibner

#start the timer
start = time.time()

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Use the small dataset')
argParser.add_argument('--data', action='store',default='data.h5')
argParser.add_argument('--data_version', action='store',default='v2',help='Version of the data to be used')
argParser.add_argument('--no_plot', action='store_true', help="Don't generate a plot")
argParser.add_argument('--n_est', action='store', default=58, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier over an interval from n_est_start to n_est_end")
argParser.add_argument('--num', action='store', default=1, type=int,nargs='?',  help="Number to not override the last plot")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")
argParser.add_argument('--swap_hypothesis', action='store_true', help="Chance the Target Labels of SM and BSM Hypothesis")
argParser.add_argument('--random_state', action='store', default=0, type=int,nargs='?',  help="The random state, which is given to the train_test_split method")
argParser.add_argument('--inverse_kule', action='store_true',help='caculate the kule div inverted')
args = argParser.parse_args()

#Set the version of the script
vversion = 'v1'

#set criterion, you can choose from (gini, kule, entropy, hellinger)

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)

#Kule algorithm
from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

#setting up the file save name
version = vversion
if args.small:
    args.data_version += '_small'
    version += '_small'
if args.swap_hypothesis:
    version += '_swap'
version += '_' + str(args.num) +  '_EstNum' + str(args.n_est) + '_RandState' + str(args.random_state)

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
    #read weights
#normalize weights, used the mean to scale it before
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


#split the data into validation and trainings set
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w,test_size= 0.5, random_state=0)

#Create the tree
dtkule = DecisionTreeClassifier(max_depth= 2, criterion=kldc)
dtgini = DecisionTreeClassifier(max_depth= 2, criterion='gini')
dtentropy = DecisionTreeClassifier(max_depth=2, criterion='entropy')

bdtkule = AdaBoostClassifier(dtkule, algorithm= args.boost_algorithm,n_estimators= args.n_est)
bdtgini = AdaBoostClassifier(dtgini, algorithm= args.boost_algorithm,n_estimators= args.n_est)
bdtentropy = AdaBoostClassifier(dtentropy, algorithm= args.boost_algorithm,n_estimators= args.n_est)

bdtkule.fit(X_train, y_train, w_train)
bdtgini.fit(X_train, y_train, w_train)
bdtentropy.fit(X_train, y_train, w_train)

train_score_kule = bdtkule.score(X_train, y_train, w_train)
test_score_kule = bdtkule.score(X_test, y_test, w_test)

train_score_gini = bdtgini.score(X_train, y_train, w_train)
test_score_gini = bdtgini.score(X_test, y_test, w_test)

train_score_entropy = bdtentropy.score(X_train, y_train, w_train)
test_score_entropy = bdtentropy.score(X_test, y_test, w_test)

#stop the time to train the tree
ende_training = time.time()

logger.info('Time to train the tree ' +  '{:5.3f}s'.format(ende_training-start))

#end if no_plot argument was choosen
if args.no_plot:
    raise SystemExit

#get the output directory for plots
output_directory = os.path.join(plot_directory,'Kullback-Leibner-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)



#setup the historgramms
kule_h_dis_train_SM = ROOT.TH1D("k_dis_train_sm", "Discriminator", 12, -1, 1)
kule_h_dis_train_BSM = ROOT.TH1D("k_dis_train_bsm", "Discriminator", 12, -1, 1)
kule_h_dis_test_SM = ROOT.TH1D("k_dis_test_sm", "Discriminator", 12, -1, 1)
kule_h_dis_test_BSM = ROOT.TH1D("k_dis_test_bsm", "Discriminator", 12, -1, 1)

gini_h_dis_train_SM = ROOT.TH1D("g_dis_train_sm", "Discriminator", 12, -1, 1)
gini_h_dis_train_BSM = ROOT.TH1D("g_dis_train_bsm", "Discriminator", 12, -1, 1)
gini_h_dis_test_SM = ROOT.TH1D("g_dis_test_sm", "Discriminator", 12, -1, 1)
gini_h_dis_test_BSM = ROOT.TH1D("g_dis_test_bsm", "Discriminator", 12, -1, 1)

entropy_h_dis_train_SM = ROOT.TH1D("e_dis_train_sm", "Discriminator", 12, -1, 1)
entropy_h_dis_train_BSM = ROOT.TH1D("e_dis_train_bsm", "Discriminator", 12, -1, 1)
entropy_h_dis_test_SM = ROOT.TH1D("e_dis_test_sm", "Discriminator", 12, -1, 1)
entropy_h_dis_test_BSM = ROOT.TH1D("e_dis_test_bsm", "Discriminator", 12, -1, 1)


#set the error calculationsmethod
ROOT.TH1.SetDefaultSumw2()

kule_test_dec_func = bdtkule.decision_function(X_test)
gini_test_dec_func = bdtgini.decision_function(X_test)
entropy_test_dec_func = bdtentropy.decision_function(X_test)
kule_train_dec_func = bdtkule.decision_function(X_train)
gini_train_dec_func = bdtgini.decision_function(X_train)
entropy_train_dec_func = bdtentropy.decision_function(X_train)
#fill the histograms
for i in xrange(len(X_test)):
    if y_test[i] == 0:
        kule_h_dis_test_SM.Fill(kule_test_dec_func[i],w_test[i])
        gini_h_dis_test_SM.Fill(gini_test_dec_func[i],w_test[i])
        entropy_h_dis_test_SM.Fill(entropy_test_dec_func[i],w_test[i])
    else:
        kule_h_dis_test_BSM.Fill(kule_test_dec_func[i],w_test[i])
        gini_h_dis_test_BSM.Fill(gini_test_dec_func[i],w_test[i])
        entropy_h_dis_test_BSM.Fill(entropy_test_dec_func[i],w_test[i])

#fill with trainings data
for i in xrange(len(X_train)): 
    if y_train[i] == 0:
        kule_h_dis_train_SM.Fill(kule_train_dec_func[i],w_train[i])
        gini_h_dis_train_SM.Fill(gini_train_dec_func[i],w_train[i])
        entropy_h_dis_train_SM.Fill(entropy_train_dec_func[i],w_train[i])
    else:
        kule_h_dis_train_BSM.Fill(kule_train_dec_func[i],w_train[i])
        gini_h_dis_train_BSM.Fill(gini_train_dec_func[i],w_train[i])
        entropy_h_dis_train_BSM.Fill(entropy_train_dec_func[i],w_train[i])

logger.info('Zeit bis vor nach der Loop: ' + '{:5.3f}s'.format(time.time()-start))


#calcuate the yields after fitting
kule_w_train_sum_sm =kule_h_dis_train_SM.Integral()
kule_w_train_sum_bsm = kule_h_dis_train_BSM.Integral()
kule_w_test_sum_sm = kule_h_dis_test_SM.Integral()
kule_w_test_sum_bsm = kule_h_dis_test_BSM.Integral()

gini_w_train_sum_sm = gini_h_dis_train_SM.Integral()
gini_w_train_sum_bsm = gini_h_dis_train_BSM.Integral()
gini_w_test_sum_sm = gini_h_dis_test_SM.Integral()
gini_w_test_sum_bsm = gini_h_dis_test_BSM.Integral()

entropy_w_train_sum_sm = entropy_h_dis_train_SM.Integral()
entropy_w_train_sum_bsm = entropy_h_dis_train_BSM.Integral()
entropy_w_test_sum_sm = entropy_h_dis_test_SM.Integral()
entropy_w_test_sum_bsm = entropy_h_dis_test_BSM.Integral()

#normalize the histograms
kule_h_dis_train_SM.Scale(1/kule_w_train_sum_sm)
kule_h_dis_train_BSM.Scale(1/kule_w_train_sum_bsm)
kule_h_dis_test_SM.Scale(1/kule_w_test_sum_sm)
kule_h_dis_test_BSM.Scale(1/kule_w_test_sum_bsm)

gini_h_dis_train_SM.Scale(1/gini_w_train_sum_sm)
gini_h_dis_train_BSM.Scale(1/gini_w_train_sum_bsm)
gini_h_dis_test_SM.Scale(1/gini_w_test_sum_sm)
gini_h_dis_test_BSM.Scale(1/gini_w_test_sum_bsm)

entropy_h_dis_train_SM.Scale(1/entropy_w_train_sum_sm)
entropy_h_dis_train_BSM.Scale(1/entropy_w_train_sum_bsm)
entropy_h_dis_test_SM.Scale(1/entropy_w_test_sum_sm)
entropy_h_dis_test_BSM.Scale(1/entropy_w_test_sum_bsm)
dataTrain = []
#Berechne die Kule Div
kl = KullbackLeibner(logger)
kule_test, error_test = kl.kule_div(kule_h_dis_test_SM, kule_h_dis_test_BSM)
kule_train, error_train = kl.kule_div(kule_h_dis_train_SM, kule_h_dis_train_BSM)
inv_kule_test, inv_error_test = kl.kule_div(kule_h_dis_test_BSM, kule_h_dis_test_SM)
inv_kule_train, inv_error_train = kl.kule_div(kule_h_dis_train_BSM, kule_h_dis_train_SM)
dataTrain.append(("Kule Train", kule_train,error_train, inv_kule_train, inv_error_train))
dataTrain.append(("Kule Test", kule_test,error_test, inv_kule_test, inv_error_test))

   
kule_test, error_test = kl.kule_div(gini_h_dis_test_SM, gini_h_dis_test_BSM)
kule_train, error_train = kl.kule_div(gini_h_dis_train_SM, gini_h_dis_train_BSM)
inv_kule_test, inv_error_test = kl.kule_div(gini_h_dis_test_BSM, gini_h_dis_test_SM)
inv_kule_train, inv_error_train = kl.kule_div(gini_h_dis_train_BSM, gini_h_dis_train_SM)
dataTrain.append(("Kule Train", kule_train,error_train, inv_kule_train, inv_error_train))
dataTrain.append(("Kule Test", kule_test,error_test, inv_kule_test, inv_error_test))
    
kule_test, error_test = kl.kule_div(entropy_h_dis_test_SM, entropy_h_dis_test_BSM)
kule_train, error_train = kl.kule_div(entropy_h_dis_train_SM, entropy_h_dis_train_BSM)
inv_kule_test, inv_error_test = kl.kule_div(entropy_h_dis_test_BSM, entropy_h_dis_test_SM)
inv_kule_train, inv_error_train = kl.kule_div(entropy_h_dis_train_BSM, entropy_h_dis_train_SM)
dataTrain.append(("Kule Train", kule_train,error_train, inv_kule_train, inv_error_train))
dataTrain.append(("Kule Test", kule_test,error_test, inv_kule_test, inv_error_test))
    


klabel = "Kullback Leibner"
if args.swap_hypothesis:
    klabel += " Labels changed"
labels = ["Name", klabel, "Error", klabel + " Inverted", "Error"]
plt.figure(figsize=(18,7))

plt.axis('tight')
plt.axis('off')
dtable = plt.table(cellText=dataTrain,colLabels=labels,loc='center')
dtable.set_fontsize(14)
dtable.scale(1.2,2)
plt.savefig(os.path.join( output_directory, 'kule-table' + version + '.png'))


#stop the timer
ende = time.time()
logger.info('Ende des Runs, Laufzeit: ' + '{:5.3f}s'.format(ende-start))
