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
import platform


import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

	def __init__(self , model ,  dataFileName , dataFilePath  , winWidth = 19 , slidingLen = 2 , featureList = ['MAV', 'RMS','ZC'] , trainPercent = [0.7, 0.2, 0.1]):
		
		'''about the model'''
		self.mModel = model #使用的模型类别：lda , svm , lr ..... 
		self.mWinWidth = winWidth #数据处理的滑动窗窗宽
		self.mSlidingLen = slidingLen #数据处理滑动窗的步长
		self.mFeatureList = featureList #特征列表
		self.mTrainPercent = trainPercent #数据的划分

		self.optimizationFlag = False

		self.dataFilePath = dataFilePath
		self.dataFileName = dataFileName

		'''the init'''
		self.getSystermType()
		# self.setFilePath() #初始化路径相关
		

	#读取数据
	def readDataFile(self):
		with open(self.dataFilePath, 'rb' ) as fp: #now , im only read csv data file
			self.emgData = pd.read_csv(fp) #得到训练数据

		self.actionNames = self.emgData['label'].unique() #得到动作类别的数组

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
		sTime = time.time()
		self.mModel.fit(self.trainX , self.trainY)
		eTime = time.time()
		if self.optimizationFlag == True:
			pass
		else:
			print("the time of train is :" , eTime - sTime)
			print("train the model over!")

	#evaluate the model
	def evaluateModel(self):

		self.trainScore = self.mModel.score(self.trainX, self.trainY)
		self.testScore = self.mModel.score(self.testX, self.testY)
		self.validateScore = self.mModel.score(self.validateX, self.validateY)

		if self.optimizationFlag == True:
			pass
		else:
			print('the score of train :', self.mModel.score(self.trainX, self.trainY))
			print('the score of test :', self.mModel.score(self.testX, self.testY))
			print('the score of validate :', self.mModel.score(self.validateX, self.validateY))

	#Set the path of the file
	def setFilePath(self):

		self.currentFilePath = os.getcwd()
		print("self.currentFilePath :" , self.currentFilePath )
		if self.systermType == "Windows":
			self.emgDataPath = self.currentFilePath  + "//"
		elif self.systermType == "Linux":
			self.emgDataPath = self.currentFilePath  +  "/"
		print("self.emgDataPath :" , self.emgDataPath )

		if os.path.exists(self.emgDataPath) == False:
			os.makedirs(self.emgDataPath)
		# if os.path.exists(self.imuDataPath) == False:
		# 	os.makedirs(self.imuDataPath)
	
	def getSystermType(self):

		self.systermType = platform.system()

	
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

	def __init__(self , mModel):


		self.model = mModel
		self.modelFileName = str(self.model.dataFileName) + "-model.pkl" #modelDataFile

		'''Optimization model'''
		self.trainScoreDict = dict()
		self.testScoreDict = dict()
		self.validScoreDict = dict()

		self.countNum = 0

		self.setModelFilePath()

	#得到特征矩阵
	def getFeature(self):
		#Splicing all the action data into a large dictionary
		self.model.readDataFile()
		emgDict = dict()
		for i in self.model.actionNames:
			temp = self.model.emgData[self.model.emgData['label']==i]
			emgDict[i] = temp[['ch0' , 'ch1' ,'ch2' , 'ch3' ,'ch4' , 'ch5' ,'ch6' , 'ch7' ]].values.T
		self.Sample = FeatureSpace(rawDict = emgDict, 
					  moveNames = self.model.actionNames, #动作类别
					  ChList = [0,1,2,3,4,5,6,7],  #传感器的通道数目
					  features = {'Names': self.model.mFeatureList,  #定义的特征
								  'LW': self.model.mWinWidth,  #定义的窗宽
								  'LI': self.model.mSlidingLen},   #定义的窗滑动的步长
					  one_hot = False   #是否进行onehot处理
					 )
		self.model.getTrainData(self.Sample)

	def setModelFilePath(self ):
		
		self.currentFilePath = os.getcwd()		
		if self.model.systermType == "Windows":
			self.modelPath = self.currentFilePath + '//data//model//'
		elif self.model.systermType == "Linux":
			self.modelPath = self.currentFilePath + '/data/model/'
		if os.path.exists(self.modelPath) == False:
			os.makedirs(self.modelPath)

	#set The Model File Name
	def setTheModelFileName(self , mModelFileName):

		self.modelFileName = mModelFileName

	#save model
	def saveModel(self):	
		print("self.model.mWinWidth :" , self.model.mWinWidth)
		print("self.model.mSlidingLen :" , self.model.mSlidingLen)
		self.modelFilePath = self.modelPath + self.modelFileName
		joblib.dump(self.model , self.modelPath + self.modelFileName)
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

	#滑动窗和滑动步长的参数选择函数
	def selectParameter(self , winWidthList  , slidingLenList ):

		totalNum = len(winWidthList) * len(slidingLenList)
		
		self.model.optimizationFlag = True

		for winWidth in winWidthList:

			for slidingLen in slidingLenList:
				self.countNum += 1
				self.model.mWinWidth = winWidth
				self.model.mSlidingLen = slidingLen
				self.getFeature()
				self.model.trainModel()
				self.model.evaluateModel()
				self.trainScoreDict[str(winWidth)  + '&' + str(slidingLen)] = self.model.trainScore
				self.testScoreDict[str(winWidth)  + '&' + str(slidingLen)] = self.model.testScore
				self.validScoreDict[str(winWidth)  + '&' + str(slidingLen)] = self.model.validateScore

				print("Done:>>" + str(int(self.countNum * 1.0  / totalNum * 100)) + "%")

		testAverageScore = self.testScoreDict[str(max(self.testScoreDict, key=self.testScoreDict.get))] * 2.0 / 3 + self.validScoreDict[str(max(self.testScoreDict, key=self.testScoreDict.get))] * 1.0 / 3
		validAverageScore = self.testScoreDict[str(max(self.validScoreDict, key=self.validScoreDict.get))] * 2.0 / 3 + self.validScoreDict[str(max(self.validScoreDict, key=self.validScoreDict.get))] * 1.0 / 3 
		
		if testAverageScore > validAverageScore :
			tempStr = max(self.testScoreDict, key=self.testScoreDict.get)
			print("the best testScore is :"  + str(max(self.testScoreDict, key=self.testScoreDict.get)) + 
					"--->" + str(self.trainScoreDict[str(max(self.testScoreDict, key=self.testScoreDict.get))]) + 
					"--->" + str(self.testScoreDict[str(max(self.testScoreDict, key=self.testScoreDict.get))]) +
					"--->" + str(self.validScoreDict[str(max(self.testScoreDict, key=self.testScoreDict.get))]))
			print("the average score is :" ,testAverageScore )
		else:
			tempStr = max(self.validScoreDict, key=self.validScoreDict.get)
			print("the best validate is :"  + str(max(self.validScoreDict, key=self.validScoreDict.get)) + 
					"--->" + str(self.trainScoreDict[str(max(self.validScoreDict, key=self.validScoreDict.get))])+
					"--->" + str(self.testScoreDict[str(max(self.validScoreDict, key=self.validScoreDict.get))])+
					"--->" + str(self.validScoreDict[str(max(self.validScoreDict, key=self.validScoreDict.get))]))
			print("the average score is :" ,validAverageScore )

		tempList = tempStr.split("&")
		self.model.mWinWidth = int(tempList[0])
		self.model.mSlidingLen = int(tempList[1])

		if self.model.systermType == "Windows":

			plt.figure(1)
			plt.title('Result Analysis')
			plt.plot(list(self.trainScoreDict.values()), color='green', label='trainScore')
			plt.plot(list(self.testScoreDict.values()), color='red', label='testScore')
			plt.plot(list(self.validScoreDict.values()), color='skyblue', label='validate')
			plt.legend() # 显示图例
			plt.xlabel('num')
			plt.ylabel('score')
			plt.show()


		
if __name__ == "__main__":
	pass

	# mWinWidthList = [x for x in range(1, 30) ]
	# mSlidingLenList = [x for x in range(1, 5)]

	# mModel = myModel(LDA() , featureList = ['MAV' , 'RMS' , "ZC" ])
	# myClf = analysisData(mModel)
	# '''train model'''
	# # myClf.fit()
	# # myClf.evaluateModel()
	# # '''save model'''
	# # myClf.saveModel()

	# '''Optimization model'''
	# myClf.selectParameter(mWinWidthList , mSlidingLenList)

	

	'''load model'''
	# myClf.loadModel()
	# myClf.evaluateModel()
	# print("actionNames:" , myClf.model.actionNames)
	# print("mFeatureList:" , myClf.model.mFeatureList)
	# print("mWinWidth:" , myClf.model.mWinWidth)
	# print("mSlidingLen:" , myClf.model.mSlidingLen)
	


