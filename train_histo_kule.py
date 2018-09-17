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
argParser.add_argument('--log_plot', action='store_true',help='Use a logarithmic plot')
argParser.add_argument('--save', action='store_true', help='Write the trained BDT to a file')
argParser.add_argument('--criterion', action='store', default='kule', nargs='?', choices=['gini', 'kule', 'entropy'] , help="Select the Criterion to be used")
argParser.add_argument('--export', action='store_true', help="Export the trainded tree as graphviz dot")
argParser.add_argument('--no_plot', action='store_true', help="Don't generate a plot")
argParser.add_argument('--max_depth', action='store', default=2, type=int,nargs='?',  help="The max depth argument, which is given to the DecisionTreeClassifier")
argParser.add_argument('--n_est_start', action='store', default=100, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier over an interval from n_est_start to n_est_end")
argParser.add_argument('--n_est_end', action='store', default=2000, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier")
argParser.add_argument('--est_num', action='store', default=40, type=int,nargs='?',  help="The number of steps between the start and end of n_estimators")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")
argParser.add_argument('--swap_hypothesis', action='store_true', help="Chance the Target Labels of SM and BSM Hypothesis")
argParser.add_argument('--random_state', action='store', default=0, type=int,nargs='?',  help="The random state, which is given to the train_test_split method")

args = argParser.parse_args()

#Set the version of the script
vversion = 'v10'

#set criterion, you can choose from (gini, kule, entropy, hellinger)

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)

#Kule algorithm
from kullback_leibner_divergence_criterion import KullbackLeibnerCriterion
kldc = KullbackLeibnerCriterion(1, np.array([2], dtype='int64'))

#setting up the file save name
version = vversion
version += '_' + args.criterion
if args.small:
    args.data_version += '_small'
    version += '_small'
if args.log_plot:
    version += '_log'
if args.swap_hypothesis:
    version += '_swap'
version += '_maxDepth' + str(args.max_depth) + '_estStart' + str( args.n_est_start) + '_estEnd' + str(args.n_est_end) + '_EstNum' + str(args.est_num) + '_BoostAlg'  + str(args.boost_algorithm) + '_RandState' + str(args.random_state)

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
w0 = np.array(df['sm_weight'])
sm_weight_mean = np.mean(w0)
w0 /= sm_weight_mean
w1 = np.array(df['bsm_weight'])
bsm_weight_mean = np.mean(w1)
w1 /= bsm_weight_mean
if args.swap_hypothesis:
    w = np.concatenate((w1,w0))
else:
    w = np.concatenate((w0,w1))
    #calculate weight mean and stretch it to an array
weight_mean = np.mean([sm_weight_mean, bsm_weight_mean])
weight_mean_array = np.full([len(w0)], weight_mean)

logger.info('Mean of sm_weights: %f, mean of bsm_weights: %f',sm_weight_mean, bsm_weight_mean  )

#split the data into validation and trainings set
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w,test_size= 0.5, random_state=0)

#Create the tree
if args.criterion == 'kule':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion=kldc)
elif args.criterion == 'gini':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='gini')
elif args.criterion == 'entropy':    
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='entropy')
else:
    assert False, "You choose the wrong Classifier"

parameters = np.linspace(args.n_est_start, args.n_est_end, num=args.est_num, dtype=np.int32) 
train_scores = []
test_scores = []

for para in parameters:
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(dt, algorithm= args.boost_algorithm,n_estimators= para)
    bdt.fit(X_train, y_train, w_train)
    train_score = bdt.score(X_train, y_train, w_train)
    test_score = bdt.score(X_test, y_test, w_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    logger.info("Parameter: %i, Train-Score: %f, Test-Score: %f", para, train_score, test_score)

i_para_optimal = np.argmax(test_scores)
para_optim = parameters[i_para_optimal]
logger.info("The optimal Parameter was: %i", para_optim)
#stop the time to train the tree
ende_training = time.time()

bdt = AdaBoostClassifier(dt, alorithm = args.boost_algorithm, n_estimators=para_optim)
bdt.fit(X_train, y_train, w_train)

logger.info('Time to train the tree ' +  '{:5.3f}s'.format(ende_training-start))





#get the directory for data
output_dir = os.path.join(tmp_directory, args.data_version)
if not os.path.exists(output_dir):
    os.makedirs( output_dir)

#save the data to file
if args.save:
    logger.info("Save the trained plot to %s, it uses the %s criterion", output_dir, args.criterion)
    joblib.dump(bdt,  os.path.join(output_dir,"bdt-trainingdata-" + version)) 
    logger.info("Dumped the tree to %s",  os.path.join(output_dir,"bdt-trainingdata-" + version))

#export the tree TODO! NOT WORKING!!
if args.export:
    export_graphviz(bdt, out_file= os.path.join(output_dir, version + "-tree.dot"))
    logger.info("Exported the tree as .dot to %s", os.path.join(output_dir, version + "-tree.dot")) 



#end if no_plot argument was choosen
if args.no_plot:
    raise SystemExit

#get the output directory for plots
output_directory = os.path.join(plot_directory,'Kullback-Leibner-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)


#show the performance plot
plt.semilogx(parameter, train_score, label='Train')
plt.semilogx(parameter, test_score, label='Test')
plt.vlines(para_optim, plt.ylim()[0], np.max(test_score), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')

#save the matlib plot
plt.savefig(os.path.join( output_directory, 'training_performance' + version + '.png'))

#setup the historgramms
h_dis_train_SM = ROOT.TH1F("dis_train_sm", "Discriminator", 25, -1, 1)
h_dis_train_BSM = ROOT.TH1F("dis_train_bsm", "Discriminator", 25, -1, 1)
h_dis_test_SM = ROOT.TH1F("dis_test_sm", "Discriminator", 25, -1, 1)
h_dis_test_BSM = ROOT.TH1F("dis_test_bsm", "Discriminator", 25, -1, 1)

#set the error calculationsmethod
ROOT.TH1.SetDefaultSumw2()

#set colors
h_dis_train_SM.SetLineColor(ROOT.kBlue+2)
h_dis_train_BSM.SetLineColor(ROOT.kRed+2)
h_dis_test_SM.SetLineColor(ROOT.kBlue-5)
h_dis_test_BSM.SetLineColor(ROOT.kRed-4)

#set line width
h_dis_train_SM.SetLineWidth(3)
h_dis_train_BSM.SetLineWidth(3)
h_dis_test_SM.SetLineWidth(3)
h_dis_test_BSM.SetLineWidth(3)


#set colors
h_dis_train_SM.SetFillColor(ROOT.kBlue+3)
h_dis_train_BSM.SetFillColor(ROOT.kRed+3)
h_dis_test_SM.SetFillColor(ROOT.kBlue-8)
h_dis_test_BSM.SetFillColor(ROOT.kRed-9)

#set fill styles
h_dis_train_SM.SetFillStyle(0)
h_dis_train_BSM.SetFillStyle(0)
h_dis_test_SM.SetFillStyle(0)
h_dis_test_BSM.SetFillStyle(0)

#hide statbox
ROOT.gStyle.SetOptStat(0)

logger.info('Zeit bis vor der Loop: ' + '{:5.3f}s'.format(time.time()-start))

test_decision_function = bdt.decision_function(X_test)
train_decision_function = bdt.decision_function(X_train)

#loop over the feature vektor to fill the histogramms (test data)

for i in xrange(len(X_test)):
    if y_test[i] == 0:
        h_dis_test_SM.Fill(test_decision_function[i],w_test[i])
    else:
        h_dis_test_BSM.Fill(test_decision_function[i],w_test[i])

#fill with trainings data
for i in xrange(len(X_test)): 
    if y_test[i] == 0:
        h_dis_train_SM.Fill(train_decision_function[i],w_train[i])
    else:
        h_dis_train_BSM.Fill(train_decision_function[i],w_train[i])

logger.info('Zeit bis vor nach der Loop: ' + '{:5.3f}s'.format(time.time()-start))

#normalize the histograms
h_dis_train_SM.Scale(1/h_dis_train_SM.Integral())
h_dis_train_BSM.Scale(1/h_dis_train_BSM.Integral())
h_dis_test_SM.Scale(1/h_dis_test_SM.Integral())
h_dis_test_BSM.Scale(1/h_dis_test_BSM.Integral())

#Berechne die Kule Div
kl = KullbackLeibner(logger)
kule_test, error_test = kl.kule_div(h_dis_test_SM, h_dis_test_BSM)
kule_train, error_train = kl.kule_div(h_dis_train_SM, h_dis_train_BSM)
logger.info('Kullback-Leibner Divergenz:\nTraining: %f and error: %f \nTesting: %f and error: %f',kule_train,error_train, kule_test, error_test)

#Plot the diagramms
c = ROOT.TCanvas("Discriminator", "", 2880, 1620)
h_dis_train_SM.Draw("h e")
h_dis_train_BSM.Draw("h same e")
h_dis_test_SM.Draw("h same e")
h_dis_test_BSM.Draw("h same e")


#add a legend
leg = ROOT.TLegend(0.65, 0.9, 0.9, 0.65)
leg.AddEntry(h_dis_train_SM,"SM Training")
leg.AddEntry(h_dis_train_BSM,"BSM Training")
leg.AddEntry(h_dis_test_SM,"SM Testing")
leg.AddEntry(h_dis_test_BSM,"BSM Testing")
leg.AddEntry(0, "Kul.div. test: " + str(kule_test) + " train: " + str(kule_train), "")
leg.AddEntry(0, "Kul.div. error  test: " + str(error_test) + " train: " + str(error_train), "")

leg.Draw()

#label the axes
h_dis_train_SM.GetXaxis().SetTitle("Discriminator")
h_dis_train_SM.GetYaxis().SetTitle("Normalized number of events")

#Save to file
c.Print(os.path.join( output_directory, 'ttz-kule' + version + '.png'))



#stop the timer
ende = time.time()
logger.info('Ende des Runs, Laufzeit: ' + '{:5.3f}s'.format(ende-start))
