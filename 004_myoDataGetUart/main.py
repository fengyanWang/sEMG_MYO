#coding:UTF-8
from __future__ import print_function
import os
import platform
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from saveData import saveMyoData
from offlineClf import analysisData , myModel
from onlineClf import onLineClf

class myoMain(object):

	def __init__(self):

		self.emgDataFileExitFlag = False
		self.modelFilePathExitFlag = False

		self.emgDataExitFlag = False
		self.modelPathExitFlag = False

		self.selectModelRunFlag = False
		self.saveModelRunFlag = False
		self.saveDataRunFlag = False

		self.fileName = None
		self.emgFileNameList = []
		self.modelFileNameList = []

		self.mSaveMyoData = saveMyoData()	

	def getFileExit(self , dataType = ["emg" , "model"]):

		currentFilePath = os.getcwd()
		self.systermType = platform.system()

		if self.systermType == "Windows":
			if "emg" in dataType:
				self.emgDataPath = currentFilePath  +'//data//emgData'
			if "model" in dataType:
				self.modelPath = currentFilePath  + '//data//model' 
		elif self.systermType == "Linux":
			if "emg" in dataType:
				self.emgDataPath = currentFilePath  +'/data/emgData'
			if "model" in dataType:
				self.modelPath = currentFilePath  + '/data/model'
		
		if "emg" in dataType:
			if os.path.exists(self.emgDataPath) == True:
				self.emgDataFileExitFlag = True
				for _, _, files in os.walk(self.emgDataPath):
					print("file:" , files)
					self.emgFileNameList.append(files)
				self.fileName = self.emgFileNameList[0]
				if self.systermType == "Windows":
					self.emgDataFilePath = self.emgDataPath + "//" + str(self.fileName[0])
				elif self.systermType == "Linux":
					self.emgDataFilePath = self.emgDataPath + "/" + str(self.fileName[0]) 
				if os.path.getsize(self.emgDataFilePath) != 0 :
					self.emgDataExitFlag = True

		if "model" in dataType:
			if os.path.exists(self.modelPath) == True:
				self.modelFilePathExitFlag = True
				for _, _, files in os.walk(self.modelPath):
					self.modelFileNameList.append(files)
				self.fileName = self.modelFileNameList[0]
				if self.systermType == "Windows":
					self.modelDataFilePath = self.modelPath + "//" + str(self.fileName[0]) 
				elif self.systermType == "Linux":
					self.modelDataFilePath = self.modelPath + "/" + str(self.fileName[0]) 
				if os.path.getsize(self.modelDataFilePath) != 0 :
					self.modelPathExitFlag = True

		
	def run(self):
		
		while True:

			actionType = raw_input("Please enter the action to be executed!>>")
			
			if actionType == "saveData":
				self.saveDataRunFlag = True
				self.mSaveMyoData.start()
				try:
					self.mSaveMyoData.run()	
				except KeyboardInterrupt:
					self.mSaveMyoData.mThread.stopThread()

			elif actionType == "selectModel":

				self.selectModelRunFlag = True

				if self.saveDataRunFlag == True:

					self.emgDataFilePath = self.mSaveMyoData.emgDataFilePath
					self.fileName = self.mSaveMyoData.fileName
					self.emgDataFileExitFlag = True
					self.emgDataExitFlag = True
				else:
					self.getFileExit(dataType = ["emg"])

				if  self.emgDataFileExitFlag == True and self.emgDataExitFlag == True: 
					self.mModel = myModel(model = LDA()  , dataFileName = self.fileName ,  dataFilePath = self.emgDataFilePath , featureList = ['MAV' , 'RMS' , "ZC" ])
					self.myClf = analysisData(self.mModel)
					mWinWidthList = [x for x in range(1, 30) ]
					mSlidingLenList = [x for x in range(1, 5)]
					self.myClf.selectParameter(mWinWidthList , mSlidingLenList)
				else:
					print("EMG file does not exist. Please execute the command first: 'saveDate', get EMG data.")
					continue

			elif actionType == "saveModel":

				self.saveModelRunFlag = True
				
				if self.saveDataRunFlag == True:

					self.emgDataFilePath = self.mSaveMyoData.emgDataFilePath
					self.fileName = self.mSaveMyoData.fileName
					self.emgDataFileExitFlag = True
					self.emgDataExitFlag = True

				else:
					self.getFileExit(dataType = ["emg"])

				if  self.emgDataFileExitFlag == True and self.emgDataExitFlag == True :
					if self.selectModelRunFlag == True:
						pass
					else:
						self.mModel = myModel(model = LDA()  , dataFileName = self.fileName ,  dataFilePath = self.emgDataFilePath , featureList = ['MAV' , 'RMS' , "ZC" ])
						self.myClf = analysisData(self.mModel)
					self.myClf.fit()
					self.myClf.evaluateModel()
					self.myClf.saveModel()
				else:
					print("EMG file does not exist. Please execute the command first: 'saveDate', get EMG data.")
					continue

			elif actionType == "loadModel":

				if  self.saveModelRunFlag == True:

					self.modelFilePathExitFlag = True 
					self.modelPathExitFlag = True
				else:
					self.getFileExit(dataType = ["model"])

				if self.modelFilePathExitFlag == True and self.modelPathExitFlag == True:
					if self.saveModelRunFlag == True:
						pass
					else:
						self.getFileExit(dataType = ["emg"])	
						self.mModel = myModel(model = LDA()  , dataFileName = self.fileName ,  dataFilePath = self.emgDataFilePath , featureList = ['MAV' , 'RMS' , "ZC" ])
						self.myClf = analysisData(self.mModel)
					self.myClf.loadModel()
					self.myClf.evaluateModel()
					print("actionNames:" , self.myClf.model.actionNames)
					print("mFeatureList:" , self.myClf.model.mFeatureList)
					print("mWinWidth:" , self.myClf.model.mWinWidth)
					print("mSlidingLen:" , self.myClf.model.mSlidingLen)
				else:
					print("The model file is not saved. Please run the command 'saveModel' first: get the model.")
					continue

			elif actionType == "onlineClf":

				if self.saveModelRunFlag == True:
					self.modelFilePathExitFlag = True 
					self.modelPathExitFlag = True
					self.modelDataFilePath = self.myClf.modelFilePath
				else:
					self.getFileExit(dataType = ["model"])
				if self.modelFilePathExitFlag == True and self.modelPathExitFlag == True:

					self.myOnLineClf = onLineClf(modelFilePath = self.modelDataFilePath , dataMode = ["emg"])
					self.myOnLineClf.start()
					self.myOnLineClf.run()
				else:
					print("The model file is not saved. Please run the command 'saveModel' first: get the model.")
					continue

			elif actionType == "help":
				print("The commnd currently supported is:" + "[saveData] , [selectModel] , [saveModel] , [loadModel] , [onlineClf] , [help] , [exit])")

			elif actionType == "exit":
				print("------------------------------ exit--------------------------------------------")
				break
			else:
				print('>>Your input is wrong, please reenter it! ')
				print('>>The support commands are: [saveData] , [selectModel] , [saveModel] , [loadModel] , [onlineClf] , [help] , [exit]')
				continue

if __name__ == "__main__":

	mMyoMain = myoMain()
	mMyoMain.run()
	
	

