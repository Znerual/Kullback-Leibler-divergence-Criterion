import os
for i in range(1,50):
    os.system("python epoch_train.py --n_est " + str(i)) 
    os.system("python epoch_train.py --criterion gini --n_est " + str(i)) 
    os.system("python epoch_train.py --criterion entropy --n_est " + str(i)) 
  
