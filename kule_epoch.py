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
argParser.add_argument('--log_plot', action='store_true',help='Use a logarithmic plot')
argParser.add_argument('--no_plot', action='store_true', help="Don't generate a plot")
argParser.add_argument('--max_depth', action='store', default=2, type=int,nargs='?',  help="The max depth argument, which is given to the DecisionTreeClassifier")
argParser.add_argument('--n_est_start', action='store', default=100, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier over an interval from n_est_start to n_est_end")
argParser.add_argument('--n_est_end', action='store', default=2000, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier")
argParser.add_argument('--est_num', action='store', default=40, type=int,nargs='?',  help="The number of steps between the start and end of n_estimators")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")
argParser.add_argument('--swap_hypothesis', action='store_true', help="Chance the Target Labels of SM and BSM Hypothesis")
argParser.add_argument('--random_state', action='store', default=0, type=int,nargs='?',  help="The random state, which is given to the train_test_split method")
argParser.add_argument('--ptz_only', action='store_true', help='Only use the pTZ feature for training')

args = argParser.parse_args()

#Set the version of the script
vversion = 'v12'

#set criterion, you can choose from (gini, kule, entropy, hellinger)

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
if args.log_plot:
    version += '_log'
if args.swap_hypothesis:
    version += '_swap'
if args.ptz_only:
    version +='_ptconly'
version += '_maxDepth' + str(args.max_depth) + '_estStart' + str( args.n_est_start) + '_estEnd' + str(args.n_est_end) + '_EstNum' + str(args.est_num) + '_BoostAlg'  + str(args.boost_algorithm) + '_RandState' + str(args.random_state)



def plot(epoch, test, train, error_test, error_train, choice):
    class_names = ["Gini", "Kule" , "Entropy"]
    plot_colors = ["#000cff","#ff0000", "#9ba0ff" , "#ff8d8d"]
    plt.figure(figsize=(12,12))
    plot_step = 0.25
    #pyplot settings
    if choice == 'test':
        for n in class_names:
            plt.plot(epoch, test[n](0),'b', label='Trained with ' + n + ' gini index')  #0 is the gini 
            plt.plot(epoch, test[n](1),'r', label='Trained with ' + n + 'kullback-leibler divergence')  #1 is the kule div
    elif choice == 'train':
        for n in class_names:
            plt.plot(epoch, train[n](0),'b', label='Trained with ' + n + ' gini index')  #0 is the gini 
            plt.plot(epoch, train[n](1),'r', label='Trained with ' + n + 'kullback-leibler divergence')  #1 is the kule div
    elif choice == 'error_test':
        for n in class_names:
            plt.plot(epoch, error_test[n](0),'b', label='Trained with ' + n + ' gini index')  #0 is the gini 
            plt.plot(epoch, error_test[n](1),'r', label='Trained with ' + n + 'kullback-leibler divergence')  #1 is the kule div
    
    elif choice == 'error_train':
        for n in class_names:
            plt.plot(epoch, error_train[n](0),'b', label='Trained with ' + n + ' gini index')  #0 is the gini 
            plt.plot(epoch, error_train[n](1),'r', label='Trained with ' + n + 'kullback-leibler divergence')  #1 is the kule div

    else:
        return
    plt.legend(loc='lower left')
   # plt.ylim(plot_range)
    plt.xlabel('Epoch')
    plt.ylabel('Criterion')

    #save the matlib plot
    plt.savefig(os.path.join( output_directory, 'Epoch' + version + '.png'))

def get_histograms(X_test, X_train, y_test, y_train, w_test, w_train, test_decision_function, train_decision_function):
    
    #setup the historgramms
    h_dis_train_SM = ROOT.TH1D("dis_train_sm", "Discriminator", 12, -1, 1)
    h_dis_train_BSM = ROOT.TH1D("dis_train_bsm", "Discriminator", 12, -1, 1)
    h_dis_test_SM = ROOT.TH1D("dis_test_sm", "Discriminator", 12, -1, 1)
    h_dis_test_BSM = ROOT.TH1D("dis_test_bsm", "Discriminator", 12, -1, 1)

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

    #set line style
    h_dis_train_SM.SetLineStyle(7)
    h_dis_train_BSM.SetLineStyle(7)
    h_dis_test_SM.SetLineStyle(1)
    h_dis_test_BSM.SetLineStyle(1)



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


    #loop over the feature vektor to fill the histogramms (test data)

    for i in xrange(len(X_test)):
        if y_test[i] == 0:
            h_dis_test_SM.Fill(test_decision_function[i],w_test[i])
        else:
            h_dis_test_BSM.Fill(test_decision_function[i],w_test[i])

    #fill with trainings data
    for i in xrange(len(X_train)): 
        if y_train[i] == 0:
            h_dis_train_SM.Fill(train_decision_function[i],w_train[i])
        else:
            h_dis_train_BSM.Fill(train_decision_function[i],w_train[i])

    logger.info('Zeit bis vor nach der Loop: ' + '{:5.3f}s'.format(time.time()-start))

    w_test_sum_sm = 0.0
    w_test_sum_bsm = 0.0
    w_train_sum_sm = 0.0
    w_train_sum_bsm = 0.0

    #calcuate the yields after fitting
    w_train_sum_sm = h_dis_train_SM.Integral()
    w_train_sum_bsm = h_dis_train_BSM.Integral()
    w_test_sum_sm = h_dis_test_SM.Integral()
    w_test_sum_bsm = h_dis_test_BSM.Integral()


    logger.info("Yields after fitting: Training SM %f, Train BSM %f, Testing SM %f, Testing BSM %f",w_train_sum_sm, w_train_sum_bsm, w_test_sum_sm, w_test_sum_bsm )

    #normalize the histograms
    h_dis_train_SM.Scale(1/w_train_sum_sm)
    h_dis_train_BSM.Scale(1/w_train_sum_bsm)
    h_dis_test_SM.Scale(1/w_test_sum_sm)
    h_dis_test_BSM.Scale(1/w_test_sum_bsm)
    
    return (h_dis_train_SM, h_dis_train_BSM, h_dis_test_SM, h_dis_test_BSM)


#find directory
input_directory = os.path.join(tmp_directory, args.data_version)
logger.debug('Import data from %s', input_directory)


#read data from file
df = pd.read_hdf(os.path.join(input_directory, args.data))
if args.ptz_only:
    X1 = np.array(df['genZ_pt/F'])
else:
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

    
logger.info('Sum of sm_weights: %f, Sum of bsm_weights: %f',sm_weight_sum, bsm_weight_sum  )

#split the data into validation and trainings set
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X,y,w,test_size= 0.5, random_state=0)



criterion_list = ['gini', 'entropy', kldc]
name_list = ['Gini' , 'Entropy' , 'Kule']
test = {}
train = {}
error_test = {}
error_train = {}
kl = KullbackLeibler(logger)
gi = Gini(logger)
if args.ptz_only:
    X_train = np.reshape(X_train, (-1,1))
    X_test = np.reshape(X_test, (-1,1))

for crit, name in zip(criterion_list, name_list):
#Create the tree
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion=crit)
    parameters = np.linspace(args.n_est_start, args.n_est_end, num=args.est_num, dtype=np.int32) 

    for para in parameters:
        # Create and fit an AdaBoosted decision tree for kule criterion
        bdt = AdaBoostClassifier(dt, algorithm= args.boost_algorithm,n_estimators= para)
        bdt.fit(X_train, y_train, w_train)
        #get the decision functions from the kule tree
        test_dec_fct = bdt.decision_function(X_test)
        train_dec_fct = bdt.decision_function(X_train)
        #get the histograms for kule
        h_dis_train_SM, h_dis_train_BSM, h_dis_test_SM, h_dis_test_BSM = get_histograms(X_test, X_train, y_test, y_train, w_test, w_train, test_dec_fct, train_dec_fct)
        #Berechne die Kule Div und den Gini
        kule_test, k_error_test = kl.kule_div(h_dis_test_SM, h_dis_test_BSM)
        kule_train, k_error_train = kl.kule_div(h_dis_train_SM, h_dis_train_BSM)
        gini_test, g_error_test = gi.gini(h_dis_test_SM, h_dis_test_BSM)
        gini_train, g_error_train = gi.gini(h_dis_train_SM, h_dis_train_BSM)
        h_dis_train_SM.Delete()
        h_dis_train_BSM.Delete()
        h_dis_test_SM.Delete()
        h_dis_test_BSM.Delete()        
        if name in test:
            test[name].append((gini_test, kule_test))
            train[name].append((gini_train, kule_train))
            error_test[name].append((g_error_test, k_error_test))
            error_train[name].append((k_error_train, k_error_train))
        else:
            test[name] = [(gini_test, kule_test)]
            train[name] = [(gini_train, kule_train)]
            error_test[name] = [(g_error_test, k_error_test)]
            error_train[name] = [(g_error_train, k_error_train)]
    logger.info('Kullback-Leibler Divergenz:\nTraining: %f and error: %f \nTesting: %f and error: %f',kule_train,error_train, kule_test, error_test)
    
#stop the time to train the tree
ende_training = time.time()
logger.info('Time to train the tree ' +  '{:5.3f}s'.format(ende_training-start))



#get the output directory for plots
output_directory = os.path.join(plot_directory,'Kullback-Leibler-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)


#end if no_plot argument was choosen
if not args.no_plot:
    plot()


#stop the timer
ende = time.time()
logger.info('Ende des Runs, Laufzeit: ' + '{:5.3f}s'.format(ende-start))
