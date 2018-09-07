# load datasets
from TTXPheno.samples.benchmarks         import *

# weights
from TTXPheno.Tools.WeightInfo           import WeightInfo
sample = fwlite_ttZ_ll_LO_order2_15weights_ref_CMS 

# RootTools
from RootTools.core.standard import *

#numpy
import numpy as np

#pandas
import pandas as pd

#Arguments
import argparse
argParser = argparse.ArgumentParser(description = "Argument Parser")
argParser.add_argument('--logLevel', action='store', default='INFO', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], help="Log level for logging")
argParser.add_argument('--small', action='store_true', help='Run only a subset')
argParser.add_argument('--version', action='store', default='v1')
argParser.add_argument('--chunksize', action='store',  default='100000', type=int)
args = argParser.parse_args()

#Logger
import RootTools.core.logger as Logger
logger = Logger.get_logger(args.logLevel, logFile = None)


#Output directory and makeing the dataset small
if args.small:
    args.version += '_small'
    sample.reduceFiles( to = 1 )

output_dir = os.path.join('ttz_data', args.version)
if not os.path.exists(output_dir):
    os.makedirs( output_dir )

# lumi
lumi = 150

# Polynomial parametrization
w = WeightInfo(sample.reweight_pkl)
w.set_order(2)

# function that evaluates the weight of the SM hypothesis
sm_weight = w.get_weight_func()
# function that evaluates the weight of the BSM hypothesis
bsm_weight = w.get_weight_func(ctZ = 4)

selectionString = "Sum$(genLep_pt>10&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)==3&&Sum$(genLep_pt>20&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)>=2&&Sum$(genLep_pt>40&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)>=1&&abs(genZ_mass-91.2)<=10&&Sum$(genJet_pt>30&&abs(genJet_eta)<2.4)>=3&&Sum$(genJet_pt>30&&genJet_matchBParton>=1&&abs(genJet_eta)<2.4)>=1&&genZ_pt>=0"
sample.setSelectionString( selectionString ) 

# Define variables
file_read_variables = [ "genZ_pt/F", "genZ_eta/F", "genZ_phi/F", "genZ_mass/F", "genZ_cosThetaStar/F", "ref_lumiweight1fb/F" ]
read_variables = map( TreeVariable.fromString, file_read_variables)
read_variables.append( VectorTreeVariable.fromString('p[C/F]', nMax=2000) )

training_variables = {
    'genZ_pt/F' : lambda event: event.genZ_pt,
    'genZ_cosThetaStar/F' : lambda event: event.genZ_cosThetaStar }

spectator_variables = {
    'genZ_eta/F' : lambda event: event.genZ_eta,
    'genzZ_phi/F' : lambda event: event.genZ_phi,
    'genZ_mass/F' : lambda event: event.genZ_mass,
    'ref_lumiweight1fb/F' : lambda event: event.ref_lumiweight1fb }


#setup the .h5 file
datadict = {key : [] for key in  training_variables.keys() + spectator_variables.keys() }
datadict['sm_weight'] = []
datadict['bsm_weight'] = []
#datadict['target'] = []

df = pd.DataFrame(datadict)
df.to_hdf(os.path.join( output_dir, 'data.h5'), key = 'df', format='table', mode='w')

i = 0
lastsavedindex = 0

reader = sample.treeReader( variables = read_variables )
reader.start()

    #y = 0, sm
while reader.run():
    prefac = lumi*reader.event.ref_lumiweight1fb/sample.reduce_files_factor
    
    #calculate the weights and skip negativs
    sm_tmp_weight = prefac * sm_weight(reader.event, sample) 
    bsm_tmp_weight = prefac * bsm_weight(reader.event, sample) 
    if sm_tmp_weight < 0 or bsm_tmp_weight < 0: continue
    #assert when the weight gets 0
    assert (sm_tmp_weight > 0 and bsm_tmp_weight > 0),"Weight ist null in ttz_dataset"  
    
    #add events and weight to the dictionary
    for dictionary in training_variables, spectator_variables:
        for key, lambda_function in dictionary.iteritems():
                datadict[key].append( lambda_function(reader.event) )
    datadict['sm_weight'].append(sm_tmp_weight)
    datadict['bsm_weight'].append(bsm_tmp_weight) 
        
    #add  target to the list
    #datadict['target'].append(0)
    #datadict['target'].append(1)
    
    i += 1
    #chunking
    if not i % args.chunksize:
        df = pd.DataFrame(datadict)
        df.index = pd.RangeIndex(start=lastsavedindex, stop=i,step=1)
        df.to_hdf( os.path.join(output_dir, 'data.h5'), key='df', format='table', append='True', mode='a') 
        #emptying the datadic 
        datadict = {key : [] for key in  training_variables.keys() + spectator_variables.keys() }
        datadict['sm_weight'] = []
        datadict['bsm_weight'] = []
        #datadict['target'] = []
        lastsavedindex = i
     

df = pd.DataFrame(datadict)
df.index = pd.RangeIndex(start=lastsavedindex, stop=i,step=1)
df.to_hdf( os.path.join(output_dir, 'data.h5'), key='df', format='table', append='True', mode='a') 
#emptying the datadic 
datadict = {key : [] for key in  training_variables.keys() + spectator_variables.keys() }
datadict['sm_weight'] = []
datadict['bsm_weight'] = []
#datadict['target'] = []
lastsavedindex = i

logger.info( "Written drectory %s", output_dir)
