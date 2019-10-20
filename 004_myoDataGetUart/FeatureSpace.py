# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from FeaturesFcn import *

# import pywt
import time

class FeatureSpace(object):
    def __init__(self, 
                 rawDict, 
                 moveNames,  # [0, 1, 2, 3, 4, ..., 51], for 52 movements
                 ChList=list(range(8)),
                 features={'Names': 'RawImage',
                           'LW': 200,
                           'LI': 100
                          },
                 trainPercent=[0.7, 0.2, 0.1], 
                 one_hot = True
                ):
        self.rawDict = rawDict
        self.moveNames = moveNames
        self.ChList = ChList
        self.features = features
        self.trainPercent = trainPercent
        self.one_hot = one_hot

        self.nextBatchIndex = 0


        self.TrainTestValidateXY()

    def RawImage(self):
        LW = self.features['LW']
        LI = self.features['LI']
        ChList = self.ChList
        print('RawImage ...')

        Nch = len(ChList)
        rawImageX = np.zeros( (1, Nch, LW, 1) )
        rawImageY = np.ones( shape=(1, 1), dtype=np.int8 )
        for i in range(len(self.moveNames)):
            mvName = self.moveNames[i]
            # - construct (N, length=LW, height=Nch, depth=1) RawImage
            # - ->      ->(N, height=Nch, length=LW, depth=1) RawImage
            sampleMatrix =self.rawDict[mvName][ChList, :]
            N = int( (sampleMatrix.shape[1]-LW)/LI + 1 )
            rawImageXi = np.zeros( (N, Nch, LW, 1) )
            rawImageYi = np.ones( shape=(N, 1), dtype=np.int8 ) * i
                         # - only right for NinaPro
            for k in range(N):
                rawImageXi[k, 0:Nch+1, 0:LW+1, 0] = sampleMatrix[ChList, k*LI:k*LI+LW]

            rawImageX = np.vstack( (rawImageX, rawImageXi) )
            rawImageY = np.vstack( (rawImageY, rawImageYi) )

        if self.one_hot:
            rawImageY = OneHot(rawImageY)

        return rawImageX[1:, :, :, :], rawImageY[1:, :]


    def FFTImage(self):
        LW = self.features['LW']
        LI = self.features['LI']
        ChList = self.ChList
        print('FFTImage ...')

        Fs = 2000 # 2KHz sampling frequency for Delsys wireless sensors.
        fstep = 10 # not spectrogram with every frequency value is necessary.
        length = int(Fs/2/fstep)
        segmentL = 600 # 300ms with Fs=2KHz
        height = int( (segmentL-LW)/LI + 1)
        Nch = len(ChList)
        depth = Nch
        fftImageX = np.zeros( (1, height, length, depth) )
        fftImageY = np.zeros( shape=(1,1), dtype=np.int8 )

        startT = time.clock()
        for i in range(len(self.moveNames)):
            mvName = self.moveNames[i]
            sampleMatrix = self.rawDict[mvName][ChList, :]
            M = int( sampleMatrix.shape[1]/segmentL )

            fftImageXi = np.zeros( (M, height, length, depth) )
            fftImageYi = np.ones( shape=(M,1), dtype=np.int8) * i

            for m in range(M):
                for ch in ChList:
                    sequenceCh = sampleMatrix[ch, m*segmentL:(m+1)*segmentL]
                    for nw in range(height):
                        sequence = sequenceCh[nw*LI:nw*LI+LW]
                        freq, spectr = fftSpect(sequence, Fs)  # 100 points
                        # spectr = spectr[0:int(Fs/2):fstep]

                        fftImageXi[m, nw, :, ch] = spectr
            endT = time.clock()
            # print('Time used %d for moveName %d' % (endT - startT, int(mvName) ) )
            startT = time.clock()
            fftImageX = np.vstack( (fftImageX, fftImageXi) )
            fftImageY = np.vstack( (fftImageY, fftImageYi) )

        if self.one_hot:
            fftImageY = OneHot(fftImageY)

        return fftImageX[1:, :, :, :], fftImageY[1:, :]

    def WTImage(self):
        LW = self.features['LW']
        LI = self.features['LI']
        ChList = self.ChList
        print('WTImage ...')

        Fs = 2000 # 2KHz sampling frequency for Delsys wireless sensors.
        waveletNames = ['gaus1'] # more like, ['gaus1', 'gaus2'], and so on. 

        frequencies = np.arange(1, 300, 5) # define target frequencies or scales
        length = LW
        height = len(frequencies)
        depth = len(ChList)
        wtImageX = np.zeros( (1, length, height, depth))
        wtImageY = np.ones( shape=(1,1), dtype=np.int8)

        for i in range(len(self.moveNames)):
            mvName = self.moveNames[i]
            sampleMatrix = self.rawDict[mvName][ChList, :]
            M = int( (sampleMatrix.shape[1]-LW)/LI + 1)
            
            wtImageXi = np.zeros( (M, length, height, depth) )
            wtImageYi = np.ones( shape=(M, 1), dtype=np.int8) * i

            startT = time.clock()
            for m in range(M):
                for ch in range(len(ChList)):
                    sequenceCh = sampleMatrix[ch, m*LI:m*LI+LW]
                    #- frequencies * scales = C = Fc*Fs
                    #frequencies = np.arange(1, 300)  #-define above
                    spect = wtSpect(sequenceCh, frequencies, waveletNames[0], Fs)

                    wtImageXi[m, :, 0:height+1, ch] = spect

                       
            endT = time.clock()
            #print('Time used %f for moveName %d' % (endT - startT, mvName ) )
            startT = time.clock()

            wtImageX = np.vstack( (wtImageX, wtImageXi) )
            wtImageY = np.vstack( (wtImageY, wtImageYi) )
        if self.one_hot:
            wtImageY = OneHot(wtImageY)

        return wtImageX[1:, :, :, :], wtImageY[1:, :]


    def AWTImage(self):
        LW = self.features['LW']
        LI = self.features['LI']
        ChList = self.ChList
        print('AWTImage ...')

        pass

    def FeatureEng(self):
        
        ChList = self.ChList
        # print('FeatureEng ...')

        for i in range(len(self.moveNames)): 
            mvName = self.moveNames[i]
            sequenceMatrix = self.rawDict[mvName][ChList, :]
            # -- there might be a bug if only one element in ChList
            # -- a bug because [sequenceMatrix] dimension is (L,), like np.squeeze().
            featureEngXi = extractSlidingWindow(sequenceMatrix, features=self.features)
            m, Nfeatures = featureEngXi.shape
            featureEngYi = np.ones( shape=(m, 1), dtype=np.int8) * i

            # force to transform into
            # (M, length=Nfeatures, height=1, depth=1)
            featureEngXi = np.reshape( featureEngXi, (m, Nfeatures, 1, 1) )
            if i==0:
                featureEngX = featureEngXi
                featureEngY = featureEngYi
            else:
                featureEngX = np.vstack( (featureEngX, featureEngXi) )
                featureEngY = np.vstack( (featureEngY, featureEngYi) )
        if self.one_hot:
            featureEngY = OneHot(featureEngY)
        return featureEngX, featureEngY
            
    def TrainTestValidateXY(self):
        if self.features['Names'] == 'RawImage':
            #features = {'Names':'RawImages',
            #            'LW':200,
            #            'LI':100}
            ImageX, ImageY = self.RawImage()
        elif self.features['Names'] == 'FFTImage':
            #features = {'Names':'FFTImage',
            #            'LW':200,
            #            'LI':100}
            ImageX, ImageY = self.FFTImage()
        elif self.features['Names'] == 'WTImage':
            #features = {'Names':'WTImage',
            #            'LW':200,
            #            'LI':100}
            ImageX, ImageY = self.WTImage()
        elif self.features['Names'] == 'AWTImage':
            #features = {'Names':'AWTImage',
            #            'LW':200,
            #            'LI':100}
            ImageX, ImageY = self.AWTImage()
        else:
            #features = {'Names':['MAV', 'ZC'],
            #            'LW':200,
            #            'LI':100}
            ImageX, ImageY = self.FeatureEng()

        m = ImageX.shape[0]
        shuffleIndex = np.arange(m) # total samples
        np.random.shuffle(shuffleIndex)
        shuffleIndex = np.reshape(shuffleIndex, (-1,1) )
        nP = [int(p*m) for p in self.trainPercent]
        trainIndex = shuffleIndex[0:nP[0], 0]
        testIndex = shuffleIndex[nP[0]:nP[0]+nP[1], 0]
        validateIndex = shuffleIndex[nP[0]+nP[1]:, 0]

        # - train
        self.trainImageX = ImageX[trainIndex, :, :, :]
        self.trainImageY = ImageY[trainIndex, :]
        # - test
        self.testImageX = ImageX[testIndex, :, :, :]
        self.testImageY = ImageY[testIndex, :]
        # - validate
        self.validateImageX = ImageX[validateIndex, :, :, :]
        self.validateImageY = ImageY[validateIndex, :]

        # print('Extraction and Split done!')

    def next_batchN(self, nextN):
        m = self.trainImageX.shape[0]
        if (self.nextBatchIndex + nextN) < m:
            sliceIndex = list(range(self.nextBatchIndex, self.nextBatchIndex + nextN))
            self.nextBatchIndex += nextN

        elif (self.nextBatchIndex < m) and (self.nextBatchIndex + nextN > m):
            sliceIndex = list(range(self.nextBatchIndex, m))
            self.nextBatchIndex += nextN

        else:
            self.nextBatchIndex = 0
            sliceIndex = list(range(0, nextN))
        
        return self.trainImageX[sliceIndex, :, :, :], self.trainImageY[sliceIndex, :]
       

def fftSpect(y, Fs):
    Y = np.fft.fft(y) / len(y)
    n = len(y)
    half = int(n/2)
    freq = np.arange(n)/n * Fs
    freq = freq[range(half)]

    Y = Y[range(half)]

    return freq, abs(Y)


#-Wavelet Transformation
#-Input:
#       y, 1xLW, numpy array
#       frequencies, (Nscales,), numpy array
#       strWavelet, wavelet name of string type, like 'gaus1'
#-Output:
#       cwtmatrix, (Nscales, LW), numpy array
# def wtSpect(y, frequencies, strWavelet, Fs):
#    #-Attention:
#    #-frequencies * scales = C = Fc*Fs
#    Fc = pywt.central_frequency(strWavelet) 
#    scales = Fc*Fs/frequencies
#    [cwtmatrix, f] = pywt.cwt(y, scales, strWavelet, Fs);
   
#    return cwtmatrix.T # numpy array of (height, length)


def OneHot(label):
    if label.shape[1] == 1: # column
        m = label.shape[0]
        nL = len(np.unique(label))
        oneHot = np.zeros( (m, nL) )
        oneHot[np.arange(m), np.squeeze(label) ] = 1
        return oneHot


if __name__ == '__main__':
    #ninapro = FeatureSpace()
    print('Test FeatureSpace successfully')
