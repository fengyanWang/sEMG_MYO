#coding:UTF-8
from __future__ import print_function
import open_myo as myo
from open_myo import EmgMode ,  ImuMode
from sklearn.lda import LDA
import pandas as pd
import os
import time

from saveMyoData import saveMyoData
from myClf import myModel , analysisData
from onLineClf import onLineClf

class myoMain(object):

	def __init__(self):

		self.model = LDA()
		self.emgMode = myo.EmgMode.RAW
		self.imuMode = myo.ImuMode.RAW
		self.features = {'Names': ['MAV', 'RMS','ZC'],'LW': 150,'LI': 1}
		self.trainPercent = [0.7, 0.2, 0.1]
		self.dataMode = ["emg" , "imu"]

		self.emgDataFilePath 	= None
		self.accDataFilePath 	= None
		self.gyroDataFilePath 	= None
		self.quatDataFilePath 	= None
		self.modelFilePath = None

		self.fileDict = {}
		self.modelFileDict = {}
		self.filePathName = "wfyFilePath"
		self.modelName = "wfyModel"

		self.runCount = 0

	def setFilePath(self):
		
		currentFilePath = os.getcwd()
		self.filePath = currentFilePath + '/data/filePath'
		#判断文件夹是否存在，不存在则创建
		if os.path.exists(self.filePath) == False:
			os.makedirs(self.filePath)

	def start(self):

		self.setFilePath()
		self.saveMyoDataModel = saveMyoData(mEmgMode = self.emgMode , mImuMode = self.imuMode)
		self.dataModel = myModel(model = self.model , features = self.features   ,trainPercent = self.trainPercent )
		self.myClfModel = analysisData(mModel = self.dataModel , modelName = "wfyModel")
		
	#actionType = {"saveData" , "saveModel" , "loadModel" , "onlineClf"}
	def run(self):

		while True:

			actionType = raw_input("Please enter the action to be executed!: ")
			

			if actionType == "saveData":

				self.runCount = 1

				self.saveMyoDataModel.addTimeStamp()
				self.saveMyoDataModel.start()
				self.saveMyoDataModel.run()

				self.emgDataFilePath = self.saveMyoDataModel.emgDataFilePath
				self.accDataFilePath = self.saveMyoDataModel.accDataFilePath
				self.gyroDataFilePath = self.saveMyoDataModel.gyroDataFilePath
				self.quatDataFilePath = self.saveMyoDataModel.quatDataFilePath

				self.fileDict["emgDataFile"] 	= 	self.emgDataFilePath
				self.fileDict["accDataFile"] 	= 	self.accDataFilePath
				self.fileDict["gyroDataFile"] 	= 	self.gyroDataFilePath
				self.fileDict["quatDataFile"]	= 	self.quatDataFilePath
				
				tempFileDataFrame = pd.DataFrame(self.fileDict ,index=[0] ,  columns = ["emgDataFile"  , 
											"accDataFile" , 'gyroDataFile' ,'quatDataFile'])
				with open(self.filePath + '/' + self.filePathName + ".csv", 'a+') as fp:
					tempFileDataFrame.to_csv(fp)

			elif actionType =="saveModel":

				if self.runCount == 0 :
					timeStr = time.time()
					self.fileDataFrame = pd.read_csv(self.filePath + '/' + self.filePathName + ".csv")

					self.emgDataFilePath 	=  	 str(self.fileDataFrame["emgDataFile"].values[0])
					self.accDataFilePath 	=	 str(self.fileDataFrame["accDataFile"].values[0])
					self.gyroDataFilePath 	= 	 str(self.fileDataFrame["gyroDataFile"].values[0])
					self.quatDataFilePath 	= 	 str(self.fileDataFrame["quatDataFile"].values[0])
					
					self.dataModel.getFilePath(filePath = {"emg": self.emgDataFilePath ,"acc": self.accDataFilePath ,
										"gyro" : self.gyroDataFilePath , "quat" : self.quatDataFilePath})
				else:
					self.dataModel.getFilePath(filePath = {"emg": self.emgDataFilePath ,"acc": self.accDataFilePath ,
										"gyro" : self.gyroDataFilePath , "quat" : self.quatDataFilePath})

				if self.emgDataFilePath == None:
					print("EMG files do not exist, save data in leisure!")

				else:

					self.myClfModel.fit()
					self.myClfModel.evaluateModel()
					self.myClfModel.saveModel() #save model
					self.modelFilePath = self.myClfModel.modelFilePath
					self.modelFileDict["modelFile"] = self.modelFilePath
					tempFileDataFrame = pd.DataFrame(self.modelFileDict , index=[0] ,  columns = ["modelFile"])
					with open(self.filePath + '/' + "modelFile" + ".csv", 'w') as fp:
						tempFileDataFrame.to_csv(fp)
					self.runCount = 2

			elif actionType =="loadModel":
				self.myClfModel.loadModel()
				self.myClfModel.evaluateModel()
				print("actionNames:" , self.myClfModel.model.actionNames)
				print("mFeatureList:" , self.myClfModel.model.mFeatures)

			elif actionType == "onlineClf":
				self.myOnLineClfModel = onLineClf(self.dataMode)
				if self.runCount == 2:
					self.myOnLineClfModel.getModelFilePath(modelFilePath)
				else:
					tempDataFrame = pd.read_csv(self.filePath + '/' + "modelFile" + ".csv")
					self.modelFilePath = str(tempDataFrame["modelFile"].values[0])
					self.myOnLineClfModel.getModelFilePath(modelFilePath)
				print("self.modelFilePath :" , self.myOnLineClfModel.modelFilePath)
				self.myOnLineClfModel.start()
				self.myOnLineClfModel.run()
				
			elif actionType == "exit":
				print("------------------------------ exit--------------------------------------------")
				break
			else:
				print('>>Your input is wrong, please reenter it! ')
				print('>>The support commands are: {"saveData", "saveModel", "loadModel", "onlineClf" , "exit"}')
				continue

if __name__ == "__main__":

	mMyo = myoMain()
	mMyo.start()
	mMyo.run()

