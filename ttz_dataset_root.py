
#load datasets
from TTXPheno.samples.benchmarks         import *
# weights
from TTXPheno.Tools.WeightInfo           import WeightInfo
sample = fwlite_ttZ_ll_LO_order2_15weights_ref_CMS
# RootTools
from RootTools.core.standard import *

# Root_numpy
from root_numpy import root2array

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

selectionString = "Sum$(genLep_pt>10&&(abs(genLep_pdgId)==11||abs(genLep_pdgId)==13)&&abs(genLep_eta)<2.5)==3&&Sum$(genLep_pt>20&&(abs(genLep_sample.setSelectionString( selectionString ))))"

print sample.files

X = root2array(sample.files, branches=["genZ_pt/F", "genZ_eta/F", "genZ_phi/F", "genZ_mass/F", "genZ_cosThetaStar/F", "ref_lumiweight1fb/F"
],selection=selectionString)

print X
