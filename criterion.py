#default imports
import ROOT
import numpy as np

class KullbackLeibler:
    def __init__(self, logger):
        self.logger = logger
    
    def kule_div(self, sm, bsm):
        assert sm.GetNbinsX() == bsm.GetNbinsX(), "Different Bin counts, in KullbackLeibner.py"
        kule = 0
        varianz = 0
        for i in range(1,bsm.GetNbinsX()+1):
            p = bsm.GetBinContent(i)
            q = sm.GetBinContent(i)
            w_bsm = bsm.GetBinError(i)
            w_sm = sm.GetBinError(i)
            if not (p ==  0.0 and q == 0.0):
                with np.errstate(divide='raise'):
                    try:
                        lg = np.log(p/q)
                        kule += p * lg
                        varianz += np.square(w_bsm) * (np.square(1.0 + lg)) + np.square(p * w_sm/ q) #nicht error nennen, varianz
                    except ZeroDivisionError:                    
                        self.logger.warning("Divided by zero np.log(%f/%f), at bin %i with bin SM Error %f and BSM error %f", p,q, i, w_sm, w_bsm) 
                    except FloatingPointError:
                        self.logger.warning("Floating Point Error np.log(%f/%f) at bin %i with error bins SM: %f and BSM: %f", p,q, i, w_sm, w_bsm) 
                     
        return kule, np.sqrt(varianz)#return std (wurzel)

        
