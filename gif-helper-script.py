###### generate a series of 50 images, from n= 0 to n=49, where n is the
###### number of weak learners, used for boosting

import os
for i in range(1,50):
    os.system("python cut_evolution.py --n_est " + str(i)) 
   # os.system("python cut_evolution.py --criterion gini --n_est " + str(i)) 
   # os.system("python cut_evolution.py --criterion entropy --n_est " + str(i)) 
  
