######### This script is used by the gif-helper script, to generate a series
######### of single image plots, which can be assembles to a gif, showing the 
######### trainings evolution of our BDT



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
from criterion import KullbackLeibler

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
argParser.add_argument('--n_est', action='store', default=100, type=int,nargs='?',  help="The n_estimators argument, which is given to the AdaBoostClassifier")
argParser.add_argument('--boost_algorithm', action='store', default='SAMME', nargs='?', choices=['SAMME', 'SAMME.R'], help="Choose a boosting algorithm for the AdaBoostClassifier")
argParser.add_argument('--swap_hypothesis', action='store_true', help="Chance the Target Labels of SM and BSM Hypothesis")
argParser.add_argument('--random_state', action='store', default=0, type=int,nargs='?',  help="The random state, which is given to the train_test_split method")
argParser.add_argument('--ptz_only', action='store_true', help='Only use the pTZ feature for training')

args = argParser.parse_args()

#Set the version of the script
vversion = 'v1'

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)

#Kule algorithm
from kullback_leibler_divergence_criterion import KullbackLeiblerCriterion
kldc = KullbackLeiblerCriterion(1, np.array([2], dtype='int64'))

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
if args.ptz_only:
    version +='_ptconly'
version += '_maxDepth' + str(args.max_depth) +  '_n_est' + str(args.n_est) + '_BoostAlg'  + str(args.boost_algorithm) + '_RandState' + str(args.random_state)

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
w_test_sum_sm = 0.0
w_test_sum_bsm = 0.0
w_train_sum_sm = 0.0
w_train_sum_bsm = 0.0

#going through all weights and selecting them with their label
for i  in xrange(len(w_train)):
    if y_train[i] == 0:
        w_train_sum_sm += w_train[i]
    else:
        w_train_sum_bsm += w_train[i]

for i  in xrange(len(w_test)):
    if y_test[i] == 0:
        w_test_sum_sm += w_test[i]
    else:
        w_test_sum_bsm += w_test[i]

logger.info('Yields, Training SM: %f, Training BSM: %f, Testing SM: %f, Testing BSM %f', w_train_sum_sm, w_train_sum_bsm, w_test_sum_sm, w_test_sum_bsm)

#Create the tree
if args.criterion == 'kule':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion=kldc)
elif args.criterion == 'gini':
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='gini')
elif args.criterion == 'entropy':    
    dt = DecisionTreeClassifier(max_depth= args.max_depth, criterion='entropy')
else:
    assert False, "You choose the wrong Classifier"

#reshape the data only the ptz feature should be used
if args.ptz_only:
    X_train = np.reshape(X_train, (-1,1))
    X_test = np.reshape(X_test, (-1,1))

#train the tree
bdt = AdaBoostClassifier(dt, algorithm = args.boost_algorithm, n_estimators=args.n_est)
bdt.fit(X_train, y_train, w_train)
ende_training = time.time()

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
output_directory = os.path.join(plot_directory,'Kullback-Leibler-Plots',  argParser.prog.split('.')[0], vversion)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logger.info('Save to %s directory', output_directory)

#pyplot settings
class_names = ["SM Test", "BSM Test" , "SM Train", "BSM Train"]
plot_colors = ["#000cff","#ff0000", "#9ba0ff" , "#ff8d8d"]
plt.figure(figsize=(18,8)).suptitle("Decision Boundaries for test- (top) and trainings-dataset (bottom) \n n: " + str(args.n_est), fontsize=18)
plot_step = 0.075

#setup the decision functions, which will also be used by the histograms
test_decision_function = bdt.decision_function(X_test)
train_decision_function = bdt.decision_function(X_train)

#show the decision shape for test data
plt.subplot(2,1,1)
if args.ptz_only:
    #generate the 1 values which will be used to generate the cut values
    x_min, x_max = X_test.min() - 1, X_test.max() + 1
    xx = np.arange(x_min, x_max, plot_step)
    xx = np.reshape(xx, (-1,1))
    
    #map the decision function
    Z = bdt.decision_function(xx)
    Z = np.reshape(Z, (-1,1))

    #get the limits and plot the function
    y_min, y_max = Z.min() - 1, Z.max() + 1
    plt.plot(xx, Z) 
else:
    #generate the 2D Matrix, which will be used to generate the values
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 0.1 , X_test[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    #map the decision function
    Z = bdt.decision_function(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape)
    
    #fill the plot with color and draw the contour
    plt.pcolormesh(xx,yy,Z, cmap=plt.cm.coolwarm)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm) #plt.contour(.., linewidth=0.

#further plot settings
plt.axis("tight")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
if args.ptz_only:
    plt.xlabel("p_T(Z) GeV")
    plt.ylabel("Decision Function")
else:
    plt.xlabel('p_T(Z) (GeV)')
    plt.ylabel('Cos(Theta*)')


#Plot the decision boundaries and generate the decision data for training data
plt.subplot(2,1,2)
if args.ptz_only:
    #generate the 1 values which will be used to generate the cut values
    x_min, x_max = X_train.min() - 1, X_test.max() + 1
    xx = np.arange(x_min, x_max, plot_step)
    xx = np.reshape(xx,(-1,1))
    
    #map the decision function
    Z = bdt.decision_function(xx)
    Z = np.reshape(Z, (-1,1))
    
    #get the limits and plot the function
    y_min, y_max = Z.min() - 1, Z.max() + 1
    plt.plot(xx, Z) 
else:
    #generate the 2D Matrix, which will be used to generate the values
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))

    #map the decision function
    Z = bdt.decision_function(np.c_[xx.ravel(), yy.ravel()]) #use bdt.predict for no
    Z = Z.reshape(xx.shape)
    
    #fill the plot with color and draw the contour
    plt.pcolormesh(xx,yy,Z, cmap=plt.cm.coolwarm)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm) #plt.contour(.., linewidth=0.

#further plot settings
plt.axis("tight")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
if args.ptz_only:
    plt.xlabel("p_T(Z) GeV")
    plt.ylabel("Decision Function")
else:
    plt.xlabel('p_T(T) (GeV)')
    plt.ylabel('Cos(Theta*)')

#generate a number of the file and save the matlib plot
number = '%03d' % args.n_est
plt.savefig(os.path.join( output_directory, args.criterion, number + '.png'))

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

#setup the delta weights for comparison (only for DEBUGGING)
w_delta_test_sum_sm = w_test_sum_sm
w_delta_test_sum_bsm = w_test_sum_bsm
w_delta_train_sum_sm = w_train_sum_sm 
w_delta_train_sum_bsm = w_train_sum_bsm

#calcuate the yields after fitting
w_train_sum_sm = h_dis_train_SM.Integral()
w_train_sum_bsm = h_dis_train_BSM.Integral()
w_test_sum_sm = h_dis_test_SM.Integral()
w_test_sum_bsm = h_dis_test_BSM.Integral()

#calculate the delta weights
w_delta_test_sum_sm -= w_test_sum_sm
w_delta_test_sum_bsm -= w_test_sum_bsm
w_delta_train_sum_sm -= w_train_sum_sm 
w_delta_train_sum_bsm -= w_train_sum_bsm

logger.info("Yields after fitting: Training SM %f, Train BSM %f, Testing SM %f, Testing BSM %f",w_train_sum_sm, w_train_sum_bsm, w_test_sum_sm, w_test_sum_bsm )
logger.info("Difference of the weights before and after fitting: Test SM %f, Test BSM %f, Train SM %f, Train BSM %f", w_delta_test_sum_sm, w_delta_test_sum_bsm, w_delta_train_sum_sm, w_delta_train_sum_bsm)

#normalize the histograms
h_dis_train_SM.Scale(1/w_train_sum_sm)
h_dis_train_BSM.Scale(1/w_train_sum_bsm)
h_dis_test_SM.Scale(1/w_test_sum_sm)
h_dis_test_BSM.Scale(1/w_test_sum_bsm)

#Berechne die Kule Div
kl = KullbackLeibler(logger)
kule_test, error_test = kl.kule_div(h_dis_test_SM, h_dis_test_BSM)
kule_train, error_train = kl.kule_div(h_dis_train_SM, h_dis_train_BSM)
logger.info('Kullback-Leibler Divergenz:\nTraining: %f and error: %f \nTesting: %f and error: %f',kule_train,error_train, kule_test, error_test)

#Plot the diagramms
c = ROOT.TCanvas("Discriminator", "", 2880, 1620)
h_dis_train_SM.Draw("h e")
h_dis_train_BSM.Draw("h same e")
h_dis_test_SM.Draw("h same e")
h_dis_test_BSM.Draw("h same e")

#get the maximum value
max_value_hist = max(h_dis_train_SM.GetMaximum(), h_dis_train_BSM.GetMaximum(), h_dis_test_SM.GetMaximum(), h_dis_test_BSM.GetMaximum())
max_value_margin = max_value_hist / 10.0
#scale the plots
h_dis_train_SM.SetMaximum(max_value_hist + max_value_margin)
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
