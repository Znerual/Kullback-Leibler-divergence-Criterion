#standard imports
import ROOT
import numpy as np
import pandas as pd
from RootTools.core.standard import *
# load datasets
from TTXPheno.samples.benchmarks import *

# weights
from TTXPheno.Tools.WeightInfo import WeightInfo
sample = fwlite_ttZ_ll_LO_order2_15weights_ref_CMS 

# Directories
from TTXPheno.Tools.user import plot_directory, tmp_directory

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Run only a subset')
argParser.add_argument('--chunksize', action='store',  default='100000', type=int)
args = argParser.parse_args()

#version
version = 'v3'

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)

#Output directory and makeing the dataset small
if args.small:
    version += '_small'
    sample.reduceFiles( to = 10 )

output_dir = os.path.join(tmp_directory, version)
if not os.path.exists(output_dir):
    os.makedirs( output_dir )

# lumi
lumi = 150

# Polynomial parametrization
w = WeightInfo(sample.reweight_pkl)
w.set_order(2)

# function that evaluates the weight
sm_weight_func = w.get_weight_func()
bsm_weight_func = w.get_weight_func(ctGI=10)

#set the selection string for the data
selectionString = "Sum$(genLep_pt>10&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)==3&&Sum$(genLep_pt>20&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)>=2&&Sum$(genLep_pt>40&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)>=1&&abs(genZ_mass-91.2)<=10&&Sum$(genJet_pt>30&&abs(genJet_eta)<2.4)>=3&&Sum$(genJet_pt>30&&genJet_matchBParton>=1&&abs(genJet_eta)<2.4)>=1&&genZ_pt>=0"
sample.setSelectionString( selectionString ) 

# Define  variables to be read from the root file
file_read_variables = [ "genZ_pt/F", "genZ_eta/F", "genZ_phi/F", "genZ_mass/F", "genZ_cosThetaStar/F", "ref_lumiweight1fb/F" ]
read_variables = map( TreeVariable.fromString, file_read_variables)
read_variables.append( VectorTreeVariable.fromString('p[C/F]', nMax=2000) )

#connect the name with the event
training_variables = {
    'genZ_pt/F' : lambda event: event.genZ_pt,
    'genZ_cosThetaStar/F' : lambda event: event.genZ_cosThetaStar }

spectator_variables = {
    'genZ_eta/F' : lambda event: event.genZ_eta,
    'genzZ_phi/F' : lambda event: event.genZ_phi,
    'genZ_mass/F' : lambda event: event.genZ_mass,
    'ref_lumiweight1fb/F' : lambda event: event.ref_lumiweight1fb }


#setup the data dictionary
datadict = {key : [] for key in  training_variables.keys() }
datadict['sm_weight'] = []
datadict['bsm_weight'] = []

#setup the .h5 file
df = pd.DataFrame(datadict)
df.to_hdf(os.path.join( output_dir, 'data.h5'), key = 'df', format='table', mode='w')

#setup for chunking
i = 0
lastsavedindex = 0

#start the ROOT main feature loop
reader = sample.treeReader( variables = read_variables )
reader.start()

#setup the histograms
ptz_sm = ROOT.TH1F('ptZ_sm'+version, 'ptZ_sm'+version , 100,0,1000)
ptz_bsm = ROOT.TH1F('ptZ_bsm'+version , 'ptZ_bsm'+version, 100,0,1000)
cosTheta_sm = ROOT.TH1F('cosThetaStar_sm'+version, 'cosThetaStar_sm'+version, 100, -1,1 )
cosTheta_bsm = ROOT.TH1F('cosThetaStar_bsm'+version, 'cosThetaStar_bsm'+version, 100, -1,1 )
while reader.run():
    #caluculate the prefactor
    if args.small:
        prefac = lumi*reader.event.ref_lumiweight1fb/sample.reduce_files_factor
    else:
        prefac = lumi*reader.event.ref_lumiweight1fb
        
    #calculate the weights and skip negativs
    sm_weight = prefac * sm_weight_func(reader.event, sample) 
    bsm_weight = prefac * bsm_weight_func(reader.event, sample) 
    if sm_weight < 0 or bsm_weight < 0: continue
    
    #assert, throw  when the weight gets 0
    assert (sm_weight > 0 and bsm_weight > 0),"Weight ist null in ttz_dataset"  
    
    #add events and weight to the dictionary
    for key, lambda_function in training_variables.iteritems():
        datadict[key].append( lambda_function(reader.event) )

    datadict['sm_weight'].append(sm_weight)
    datadict['bsm_weight'].append(bsm_weight) 
    
    ptz_sm.Fill( reader.event.genZ_pt, sm_weight )       
    ptz_bsm.Fill( reader.event.genZ_pt, bsm_weight )

    cosTheta_sm.Fill( reader.event.genZ_cosThetaStar, sm_weight)
    cosTheta_bsm.Fill( reader.event.genZ_cosThetaStar, bsm_weight)       
    
    i += 1
    #chunking
    if not i % args.chunksize:
        #save a chunk
        df = pd.DataFrame(datadict)
        df.index = pd.RangeIndex(start=lastsavedindex, stop=i,step=1)
        df.to_hdf( os.path.join(output_dir, 'data.h5'), key='df', format='table', append='True', mode='a') 
       
        #emptying the datadic 
        datadict = {key : [] for key in  training_variables.keys() }
        datadict['sm_weight'] = []
        datadict['bsm_weight'] = []
        lastsavedindex = i
     
#save the rest
df = pd.DataFrame(datadict)
df.index = pd.RangeIndex(start=lastsavedindex, stop=i,step=1)
df.to_hdf( os.path.join(output_dir, 'data.h5'), key='df', format='table', append='True', mode='a') 

#emptying the datadic 
datadict = {key : [] for key in  training_variables.keys() }
datadict['sm_weight'] = []
datadict['bsm_weight'] = []
lastsavedindex = i

logger.info( "Written drectory %s", output_dir)

#setting up the plot colors
ptz_sm.style = styles.lineStyle( ROOT.kBlue)
ptz_bsm.style = styles.lineStyle( ROOT.kRed)
cosTheta_sm.style = styles.lineStyle( ROOT.kBlue)
cosTheta_bsm.style = styles.lineStyle( ROOT.kRed)

#Outputdirectory
output_directory = os.path.join(plot_directory, 'Kullback-Leibler-Plots', argParser.prog.split('.')[0],version)

#draw the plots
plot = Plot.fromHisto("ptz"+version,
                [[ptz_sm],[ptz_bsm]],
                texX = "p_{T}(Z) (GeV)",
            )

plotting.draw( plot,
    plot_directory = output_directory, 
    logX = False, logY = True, sorting = False, 
    copyIndexPHP = False,
    scaling = {1:0},
)

plot = Plot.fromHisto("CosTheta"+version,
                [[cosTheta_sm],[cosTheta_bsm]],
                texX = "cos(\Theta)",
            )

plotting.draw( plot,
    plot_directory = output_directory, 
    logX = False, logY = False, sorting = False, 
    copyIndexPHP = False,
    scaling = {1:0},
)
