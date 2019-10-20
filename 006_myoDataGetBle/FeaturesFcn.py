#coding:UTF-8
import numpy as np


''' As following, lots of features extraction function have been implemented.
   Please implement more features incrementally.
   
   Please read Features.ipynb for equations, explanations, conventions, and plotting&testing scripts.

'''

# FeaturesFcn
'''
'MAV': fMAV, 
'SSC': fSSC,
'ZC' : fZC, 
'WL': fWL,
'Skewness': fSkewness,
'RMS': fRMS,
'HP': fHP,
'SampEn': fSampEn,
'HIST':fHIST
'''

def fMAV(x):
    # print("the type of x : " , type(x))
    # print("the shape of x :" , x.shape)
    # print("x:" , x)
    fe = np.mean(np.abs(x), axis=1, keepdims=True)
    # print("the shape of fe : " , fe.shape)
    return fe.T

def fSSC(x, threshold=1e-10):
    Nch, LW = x.shape
    xRightShiftOne = np.hstack( (x[:, 0:1], x[:, 0:LW-1]) )  # circlly append the former value
    xLeftShiftOne = np.hstack( (x[:, 1:], x[:, LW-1:LW]) )               
    leftPart = x - xRightShiftOne
    rightPart = x - xLeftShiftOne
    multiply = leftPart * rightPart
#     threshold = 1e-12 # --------------- key parameter
    conditionValue = multiply > threshold
    fe = np.mean( conditionValue, axis=1, keepdims=True )
    return fe.T, multiply # return [multiply] for debugging...

def fZC(x, threshold=1e-6):
    Nch, LW = x.shape
    xRightShiftOne = np.hstack( (x[:, 0:1], x[:, 0:LW-1]) ) # circlly append the former value
    
    # conditions if they have different signs
    diffSigns = (-1 * np.sign(xRightShiftOne * x)) >0
    
    # surpass the threshold
    threshCondition = np.abs(xRightShiftOne - x) >= threshold
    
    fe = np.mean( diffSigns * threshCondition, axis=1, keepdims=True)
    
    return fe.T, diffSigns, threshCondition

def fWL(x):
    xCutHeadOne = x[:, 1:]
    xCutTailOne = x[:, :-1]
    diff = np.abs(xCutHeadOne-xCutTailOne)
    fe = np.mean(diff, axis=1, keepdims=True)
    return fe.T, diff


def fSkewness(x):
    Nch, LW = x.shape
    sigma = np.std(x, axis=1, keepdims=True)
    Sigma = np.matlib.repmat(sigma, 1, LW)
    average = np.mean(x, axis=1, keepdims=True)
    Average = np.matlib.repmat(average, 1, LW)
    thirdMoment = ((x-Average)/Sigma)**3
    fe = np.mean( thirdMoment, axis=1, keepdims=True )
    
    return fe.T, thirdMoment


def fRMS(x):
    fe = np.sqrt( np.mean( np.square(x), axis=1, keepdims=True ) )
    return fe.T


def fHP(x):
    def Activity(x):
        return np.var(x, axis=1, keepdims=True)  # a column vector with Nch x 1
    
    def Mobility(x):
        xDiff = np.diff(x, axis=1) # difference between adjacent columns
        return np.sqrt( Activity(xDiff)/Activity(x) ) # a column vector with Nch x 1
    
    def Complexity(x):
        xDiff = np.diff(x, axis=1) # difference between adjacent columns
        return Mobility(xDiff)/Mobility(x)  # a column vector with Nch x 1
    
    activity = Activity(x)
    mobility = Mobility(x)
    complexity = Complexity(x)
    return np.hstack( (activity.T, mobility.T, complexity.T)) # a row vector with 1 x (Nchx3)
    
def fSampEn(X, m=2, tau=0.2):
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result
    
    def _phi(row, m):
        r = tau * np.std(row)
        x = [[row[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) for x_i in x]
        return sum(C) - 1.0
    
    def _SampEnRow(row, m):
        return np.reshape( -np.log( _phi(row, m+1) / _phi(row, m)  ), (1, 1) )
    
    Nch, N = X.shape

    fe = np.zeros( (1,1))
    for ch in range(Nch):
        fe = np.hstack( (fe, _SampEnRow(X[ch, :], m) ) )
    return fe[:, 1:]


def fHIST(x):
    Nch, LW = x.shape
    fe = np.zeros( shape=(1,1) )
    for ch in range(Nch):
        d = x[ch, :] # a row vector
        std = np.std(d)
        hist, edges = np.histogram(d, bins=np.arange(-10*std, 11*std, std))
                      # must have 20 bins
                      # every bin range is [std] value, e.g.
                      # -10std -> -9std -> -8std -> ... -> 9std -> 10std
        hist = np.reshape(hist, (1, -1) )
        fe = np.hstack( (fe, hist) )
    return fe[:, 1:]/LW


def extractSlidingWindow(data, features = {'Names':['MAV'], 
                                           'LW': 512,
                                           'LI': 64}):
    # 'Names':['MAV','SSC','ZC','WL','Skewness','RMS','HP','SampEn','HIST']
    feStr = features['Names']
    LW = features['LW']
    LI = features['LI']
    
    Nch, L = data.shape
    Nw = int( (L-LW)/LI + 1)
    Nfe = len(feStr)
    
    fcn = {'MAV': fMAV, 
           'SSC': fSSC,
           'ZC' : fZC, 
           'WL': fWL,
           'Skewness': fSkewness,
           'RMS': fRMS,
           'HP': fHP,
           'SampEn': fSampEn,
           'HIST':fHIST}
    
    feMatrix = np.zeros( shape=(1,1))
    
    for nw in range(Nw):
        dataWindow = data[:, nw*LI:nw*LI+LW]
        feRow = np.zeros( shape=(1,1))
        for nfe in range(Nfe):
            festr = feStr[nfe]
            fhandle = fcn[festr]
            output = fhandle(dataWindow) # return a row vector
            
            if len(output) == 1:
                fe = output
            else:
                fe = output[0]
            
            feRow = np.hstack( (feRow, fe) )
        if nw == 0:
            feMatrix = feRow[:, 1:]
        else:
            feMatrix =  np.vstack( (feMatrix, feRow[:, 1:]) )
    return feMatrix
