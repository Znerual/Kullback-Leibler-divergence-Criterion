# load datasets
from TTXPheno.samples.benchmarks         import *

# weights
from TTXPheno.Tools.WeightInfo           import WeightInfo
sample = fwlite_ttZ_ll_LO_order2_15weights_ref_CMS 

# RootTools
from RootTools.core.standard import *

#numpy
import numpy as np

def ttz_dataset():
    # make small
    sample.reduceFiles( to = 1 )

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

    #print sample.files

    # Define variables
    read_variables = map( TreeVariable.fromString, [
    #    "GenMet_pt/F", "GenMet_phi/F", 
    #    "nGenJet/I", "GenJet[pt/F,eta/F,phi/F,matchBParton/I]", 
    #    "nGenLep/I", "GenLep[pt/F,eta/F,phi/F,pdgId/I,motherPdgId/I]", 
    #    "ntop/I", "top[pt/F,eta/F,phi/F]", 
        "genZ_pt/F", "genZ_eta/F", "genZ_phi/F", "genZ_mass/F", "genZ_cosThetaStar/F", "ref_lumiweight1fb/F"
    ])
    read_variables.append( VectorTreeVariable.fromString('p[C/F]', nMax=2000) )

    reader = sample.treeReader( variables = read_variables )
    reader.start()

    X = []
    y = []
    w = []
    #y = 0, sm

    while reader.run():
        prefac = lumi*reader.event.ref_lumiweight1fb/sample.reduce_files_factor
        
        #calculate the weights and skip negativs
        sm_tmp_weight = prefac * sm_weight(reader.event, sample) 
        bsm_tmp_weight = prefac * bsm_weight(reader.event, sample) 
        if sm_tmp_weight < 0 or bsm_tmp_weight < 0: continue
        #assert when the weight gets 0
        assert (sm_tmp_weight > 0 and bsm_tmp_weight > 0),"Weight ist null in ttz_dataset"  
        w.append(sm_tmp_weight)
        w.append(bsm_tmp_weight)
        
        #add events and target to the list
        X.append([reader.event.genZ_pt, reader.event.genZ_cosThetaStar])
        y.append(0)
        X.append([reader.event.genZ_pt, reader.event.genZ_cosThetaStar])
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    w_min = np.amin(w)

    return X,y,w, w_min
