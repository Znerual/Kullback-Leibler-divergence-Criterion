import numpy as np

from sklearn.datasets import make_gaussian_quantiles

def gauss_easy(n_samples_bsm, n_samples_sm):
    #Dataset creation
    X1,y1 = make_gaussian_quantiles(mean=(2, 0), cov=1.5, n_samples=n_samples_bsm, n_features=2, n_classes=1, random_state=1)
    X2,y2 = make_gaussian_quantiles(mean=(-2,0), cov=1.5, n_samples=n_samples_sm, n_features=2, n_classes=1, random_state=1)
    X12 = np.concatenate((X1,X2))
    X = np.concatenate((X12,X12))
    n_samle_size = len(X12)

    # Generating labels
    y1 = np.zeros(n_samle_size)
    y2 = np.ones(n_samle_size)
    y = np.concatenate((y1,y2))

    #Generating weights
    n_alt_hyp_size = len(X1) #BSM
    n_hyp_size = len(X2) #SM
    EPSILON = 0.01

    #in case of y=0, SM
    w_alt_hyp = np.zeros(n_alt_hyp_size)
    w_alt_hyp += (EPSILON / n_alt_hyp_size)
    w_hyp = np.ones(n_hyp_size)
    w_hyp /= n_hyp_size
    w0 = np.concatenate((w_alt_hyp, w_hyp))

    #in case of y=1, BSM
    w_alt_hyp = np.ones(n_alt_hyp_size)
    w_alt_hyp /= n_alt_hyp_size
    w_hyp = np.zeros(n_hyp_size)
    w_hyp += (EPSILON / n_hyp_size)
    w1 = np.concatenate((w_alt_hyp, w_hyp))

    #final weights
    w = np.concatenate((w0,w1))

    w_min = min((2.0 * EPSILON) / n_hyp_size, (2.0 * EPSILON) / n_alt_hyp_size)

    return X, y, w, w_min

def gauss_hard(n_samples_bsm, n_samples_sm):
    #Dataset creation
    X1,y1 = make_gaussian_quantiles(mean=(2, 0), cov=1.5, n_samples=n_samples_bsm, n_features=2, n_classes=1, random_state=1)
    X2,y2 = make_gaussian_quantiles( cov=1.2, n_samples=n_samples_sm, n_features=2, n_classes=1, random_state=1)

    X12 = np.concatenate((X1,-X2 + 1))
    X = np.concatenate((X12,X12))
    n_samle_size = len(X12)

    # Generating labels
    y1 = np.zeros(n_samle_size)
    y2 = np.ones(n_samle_size)
    y = np.concatenate((y1,y2))

    #Generating weights
    n_alt_hyp_size = len(X1) #BSM
    n_hyp_size = len(X2) #SM
    EPSILON = 0.01

    #in case of y=0, SM
    w_alt_hyp = np.zeros(n_alt_hyp_size)
    w_alt_hyp += (EPSILON / n_alt_hyp_size)
    w_hyp = np.ones(n_hyp_size)
    w_hyp /= n_hyp_size
    w0 = np.concatenate((w_alt_hyp, w_hyp))

    #in case of y=1, BSM
    w_alt_hyp = np.ones(n_alt_hyp_size)
    w_alt_hyp /= n_alt_hyp_size
    w_hyp = np.zeros(n_hyp_size)
    w_hyp += (EPSILON / n_hyp_size)
    w1 = np.concatenate((w_alt_hyp, w_hyp))

    #final weights
    w = np.concatenate((w0,w1))

    w_min = min((2.0 * EPSILON) / n_hyp_size, (2.0 * EPSILON) / n_alt_hyp_size)

    return X, y, w, w_min

def exponential_easy(n_samples_bsm, n_samples_sm):
    assert False, "exponential function is not ready to use, crashes the whole pc"
    lambda_sm = 2.0
    lambda_bsm = 1.5

    X1 = make_exponential_lambda(lambda_bsm, n_samples_bsm)
    X2 = make_exponential_lambda(lambda_sm, n_samples_sm)

    X12 = np.concatenate((X1,X2))
    X = np.concatenate((X12,X12))
    n_samle_size = len(X12)

    # Generating labels
    y1 = np.zeros(n_samle_size)
    y2 = np.ones(n_samle_size)
    y = np.concatenate((y1,y2))

    #Generating weights
    #EPSILON = 0.00001 / (n_samples_sm + n_samples_bsm)
    #in case of y=0, SM
    #alt_hyp = bsm
    w_alt_hyp = np.zeros(len(X1))
    w_hyp = np.zeros(len(X2))
    for i in range(0,len(X1)):
        w = np.sqrt(calc_exponential_lambda(lambda_bsm, X1[i][0]) * calc_exponential_lambda(lambda_bsm, X1[i][1]))
#Wurzel"!""
        w_alt_hyp[i] = w / (lambda_bsm) # * weight_increase)
    for i in range(0, len(X2) -1):
        w = np.sqrt(calc_exponential_lambda(lambda_sm, X2[i][0]) * calc_exponential_lambda(lambda_sm, X2[i][1]))
        w_hyp[i] = w / (lambda_sm) # * weight_increase)
    #w_alt_hyp /= lambda_bsm
    #w_hyp /= lambda_sm
    w0 = np.concatenate((w_alt_hyp, w_hyp))

    #in case of y=1, BSM
    #alt_hyp = sm
    w_alt_hyp = np.zeros(len(X2))
    w_hyp = np.zeros(len(X1))

    for i in range(0,len(X2) -1):
        w = np.sqrt(calc_exponential_lambda(lambda_sm, X2[i][0] ) * calc_exponential_lambda(lambda_sm, X2[i][1] ))
        print "calc e l" + str(calc_exponential_lambda(lambda_sm, X2[i][0]))
        w_alt_hyp[i] = w / (lambda_sm) # * weight_increase)
    for i in range(0, len(X1) -1):
        w = np.sqrt(calc_exponential_lambda(lambda_bsm, X1[i][0] ) * calc_exponential_lambda(lambda_bsm, X1[i][1] ))
        w_hyp[i] = w / (lambda_bsm) # * weight_increase)
    #w_alt_hyp /= lambda_sm
    #w_hyp /= lambda_bsm

    w1 = np.concatenate((w_alt_hyp, w_hyp))

    #final weights
    w = np.concatenate((w0,w1))

    w_min = np.amin(w[np.where(w > 0.0)]) #+ EPSILON
    EPSILON = w_min / 100.0
    w[np.where(w == 0.0)] += EPSILON

    print "X"
    print X
    print "y"
    print y
    print "w"
    print w

    return X, y, w, w_min

def make_exponential_lambda(l, n_samples):
    return np.random.exponential(l, (n_samples,2))
def calc_exponential_lambda(l,x):
    return (l * np.exp(-l * x))
#exponential_easy(4,4)
