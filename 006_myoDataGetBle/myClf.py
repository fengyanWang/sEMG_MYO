# -*- coding: utf-8 -*-

'''
 * @file:		myClf.py
 * @brief:		Training and testing of off-line emg data model	
 * @details:	It can read the off-line emg data, use overlapping Windows to enhance the 
 				original data, then extract relevant features, and finally use the sklearn
 				machine learning package to train relevant models
 * @author:		VincentWang
 * @data:		2018.06.14
 * @version:	v_1
 * @par Copyright(c):	None
 * @par history:		None	
 '''

from __future__ import print_function

import os
import pickle
import sys
import time
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.lda import LDA

from sklearn.externals import joblib

from FeatureSpace import FeatureSpace


'''
* @details			:This class can be used for data reading,feature extraction, model training,
*           	 	model evaluation and other operations
* @param model		:Objects of the incoming model, such as SVM, lda, etc
* @param winWidth	:Data processing sliding window width
* @param slidingLen	:The step size of the data processing sliding window
* @param featureList:Feature list
* @return			:None
* @see  			:None
* @note				:Since the relative path is used, there is no need to consider the data storage path. The environment is 
*         		 	 raspberry pie 3b + python2.7 + sklearn 16
'''
class myModel(object):

	def __init__(self , model , 
			    features={'Names': ['MAV', 'RMS','ZC'],'LW': 150,'LI': 1}, 
			    trainPercent=[0.7, 0.2, 0.1]):
		
		'''about the model'''
		self.mModel = model #使用的模型类别：lda , svm , lr ..... 
		self.mFeatures = features
		self.mTrainPercent = trainPercent #数据的划分
		'''the init'''
		self.setFilePath() #初始化路径相关
	
	#读取数据
	def readDataFile(self):
		
		with open(self.emgDataFilePath, 'rb' ) as fp: #now , im only read csv data file
			self.emgData = pd.read_csv(fp) #得到训练数据

		self.actionNames = self.emgData['action'].unique() #得到动作类别的数组

	#The feature space is divided into data sets
	def getTrainData(self , sample):

		nDimensions = sample.trainImageX.shape[1]
		#训练集
		self.trainX = np.reshape(sample.trainImageX, (-1, nDimensions))
		self.trainY = np.squeeze(sample.trainImageY)
		# print("the trainX is :" , self.trainX.shape)
		#测试集
		self.testX = np.reshape(sample.testImageX, (-1, nDimensions))
		self.testY = np.squeeze(sample.testImageY)
		#评估集
		self.validateX = np.reshape(sample.validateImageX, (-1, nDimensions))
		self.validateY = np.squeeze(sample.validateImageY)

	#Model training
	def trainModel(self ):
		print("start train the model....")
		sTime = time.time()
		self.mModel.fit(self.trainX , self.trainY)
		eTime = time.time()
		print("the time of train is :" , eTime - sTime)
		print("train the model over!")

	#evaluate the model
	def evaluateModel(self):

		print('the score of train :',self.mModel.score(self.trainX, self.trainY))
		print('the score of test :',self.mModel.score(self.testX, self.testY))
		print('the score of validate :',self.mModel.score(self.validateX, self.validateY))

	#Set the path of the file
	def setFilePath(self):

		self.currentFilePath = os.getcwd()
		self.emgDataPath = self.currentFilePath + "/data/emgData/"
		self.imuDataPath = self.currentFilePath + "/data/imuData/"

		if os.path.exists(self.emgDataPath) == False:
			os.makedirs(self.emgDataPath)
		if os.path.exists(self.imuDataPath) == False:
			os.makedirs(self.imuDataPath)

	def getFilePath(self , filePath = {"emg": None ,"acc": None , "gyro" : None , "quat" : None}):

		self.emgDataFilePath 	= filePath["emg"]
		self.accDataFilePath 	= filePath["acc"]
		self.gyroDataFilePath 	= filePath["gyro"]
		self.quatDataFilePath 	= filePath["quat"]

	
'''
* @details		:This class can be used for data reading, sliding window processing, 
*          	 	 feature extraction, model training, model evaluation, model storage, model reading and other operations
* @param mModel	:Objects of the incoming model, such as SVM, lda, etc
* @return		:None
* @see  		:The most important document is:FeatureSpace
* @note			:Since the relative path is used, there is no need to consider the data storage path. The environment is 
*         		 raspberry pie 3b + python2.7 + sklearn 16
'''
class analysisData(object):

	def __init__(self , mModel , modelName = "myModel"):

		self.model = mModel
		self.modelFileName = modelName + ".pkl" #modelDataFile

		self.modelFilePath = ""

		self.setModelFilePath()

	#得到特征矩阵
	def getFeature(self):
		#Splicing all the action data into a large dictionary
		self.model.readDataFile()
		emgDict = dict()
		for i in self.model.actionNames:
			temp = self.model.emgData[self.model.emgData['action']==i]
			emgDict[i] = temp[['ch0' , 'ch1' ,'ch2' , 'ch3' ,'ch4' , 'ch5' ,'ch6' , 'ch7' ]].values.T
		self.Sample = FeatureSpace(rawDict = emgDict, 
					  moveNames = self.model.actionNames, #动作类别
                      ChList = [0,1,2,3,4,5,6,7],  #传感器的通道数目
                      features = self.model.mFeatures,   #定义的窗滑动的步长
                      one_hot = False   #是否进行onehot处理
                     )
		self.model.getTrainData(self.Sample)

	def setModelFilePath(self ):
		
		self.currentFilePath = os.getcwd()
		self.modelPath = self.currentFilePath + '/data/model/'
		if os.path.exists(self.modelPath) == False:
			os.makedirs(self.modelPath)

	#set The Model File Name
	def setTheModelFileName(self , mModelFileName):

		self.modelFileName = mModelFileName

	#save model
	def saveModel(self):
		
		joblib.dump(self.model , self.modelPath + self.modelFileName)
		self.modelFilePath = self.modelPath + self.modelFileName
		print("save model done!!!")

	#load the model
	def loadModel(self):

		self.model = joblib.load(self.modelPath + self.modelFileName)
		print("load model done!!!")
		
	#start train model
	def fit(self):
		self.getFeature()
		self.model.trainModel()

	#evaluate the model
	def evaluateModel(self):
		self.model.evaluateModel()


if __name__ == "__main__":

	mModel = myModel(LDA() , features={'Names': ['MAV', 'RMS','ZC'],
                           'LW': 80,
                           'LI': 2
                          })
	myClf = analysisData(mModel)

	'''train model'''
	myClf.fit()
	myClf.evaluateModel()
	# # '''save model'''
	myClf.saveModel()
	 
	'''load model'''
	# myClf.loadModel()
	# myClf.evaluateModel()
	# print("actionNames:" , myClf.model.actionNames)
	# print("mFeatureList:" , myClf.model.mFeatures)
	# print("mWinWidth:" , myClf.model.mWinWidth)
	# print("mSlidingLen:" , myClf.model.mSlidingLen)
	




# C_range = np.logspace(-5,5,11)
# gamma_range = np.logspace(-30,1,32)
# param_grid = dict(gamma=gamma_range,C=C_range)
# cv = StratifiedShuffleSplit(trainY, n_iter=20,test_size=0.2,random_state=42)
# grid = GridSearchCV(SVC(),param_grid=param_grid,cv=cv)
# grid.fit(trainX,trainY)
# print("The best parameters are %s with a score of %0.2f" % (grid.best_params_,grid.best_score_))




