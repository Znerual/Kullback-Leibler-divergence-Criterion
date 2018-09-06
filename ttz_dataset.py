# load datasets
from TTXPheno.samples.benchmarks         import *
# weights
from TTXPheno.Tools.WeightInfo           import WeightInfo
sample = fwlite_ttZ_ll_LO_order2_15weights_ref_CMS 
# RootTools
from RootTools.core.standard import *

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

print sample.files

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

while reader.run():
    #print reader.event.genZ_pt, reader.event.genZ_cosThetaStar
    prefac = lumi*reader.event.ref_lumiweight1fb/sample.reduce_files_factor
    print "sm", prefac*sm_weight(reader.event, sample), "bsm", prefac*bsm_weight(reader.event, sample)
     
