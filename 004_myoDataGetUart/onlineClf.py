#coding:UTF-8
from __future__ import print_function

import os
import time
import platform
import numpy as np
from myThread import myThread
from offlineClf import myModel
from sklearn.externals import joblib
from myo import  Myo ,  VibrationType 
from FeatureSpace import FeatureSpace
from saveData import saveMyoData , PrintPoseListener

'''
* @details          :This class mainly completes an online recognition process of actions, 
*                    including model reading, online data collection, feature calculation, action recognition and voting
* @param dataMode   :The types of data sources used can make emg data or attitude data
* @return           :None
* @see              :None
* @note             :Since the relative path is used, there is no need to consider the data storage path. The environment is 
*                    raspberry pie 3b + python2.7 + sklearn 19
'''
class onLineClf(saveMyoData):

    def __init__(self , modelFilePath , dataMode = ["emg" , "imu"]):

        '''data mode'''
        self.dataMode = dataMode #采集的数据的种类，主要有：emg 和 imu  
        '''emg data'''
        self.emgDataLenth = 24  #每次采集多少个数据做一次特征提取和动作的识别
        self.emgDict = dict()
        '''model'''
        self.modelFilePath = modelFilePath  #分类模型保存的路径
        self.numberVoter = 3  #投票成员的个数

    '''初始化myo相关程序'''
    def start(self):

        self.listener = PrintPoseListener(dataType = self.dataMode , trainType = "on")
        self.mMyo = Myo() 
        self.getSystermType()
        # self.setFilePath() #设置路径
        try:
            self.mMyo.connect() 
            self.mMyo.add_listener(self.listener)
            self.mMyo.vibrate(VibrationType.SHORT)

        except ValueError as ex:
            print (ex)

        self.loadModel()#导入模型

        self.mMyo.vibrate(VibrationType.MEDIUM)

    #采集线程主程序
    def myoRun(self):
        try:
            while True:
                self.mMyo.run()
        except KeyboardInterrupt:
            self.mMyo.safely_disconnect()
            self.mThread.delThread("getData")

    #获取在线分类结果函数(分类线程函数)
    def onlineClf(self):

        try:
            while True:
                if self.listener.emgDataCount > self.model.mWinWidth  + self.numberVoter - 1:    #投票数为其加1
                    
                    self.listener.emgDataCount = 0
                    self.listener.emgData = np.array(self.listener.emgData , dtype = np.int64)
                    self.listener.emgData = self.listener.emgData.T
                    self.emgDict['one'] = self.listener.emgData
                    
                    self.sample = FeatureSpace(rawDict = self.emgDict, 
                              moveNames = ['one',], #动作类别
                              ChList = [0,1,2,3,4,5,6,7],  #传感器的通道数目
                              features = {'Names': self.model.mFeatureList,  #定义的特征
                                          'LW': self.model.mWinWidth,  #定义的窗宽
                                          'LI': self.model.mSlidingLen},   #定义的窗滑动的步长
                              one_hot = False ,
                              trainPercent=[1, 0, 0]    #是否进行onehot处理
                             )

                    self.getTrainData()
                    actionList = self.model.mModel.predict(self.trainX)
                    print("the action is :" , self.getTheAction(actionList))
                    self.listener.emgData = []
                    self.emgDict.clear()
                else:
                    time.sleep(0.1)
                    pass

        except KeyboardInterrupt:

            self.mThread.delThread("onlineClf")
   
    #The feature space is divided into data sets
    def getTrainData(self):

        nDimensions = self.sample.trainImageX.shape[1]
        #训练集
        self.trainX = np.reshape(self.sample.trainImageX, (-1, nDimensions))
        self.trainY = np.squeeze(self.sample.trainImageY)
        #测试集
        self.testX = np.reshape(self.sample.testImageX, (-1, nDimensions))
        self.testY = np.squeeze(self.sample.testImageY)
        #评估集
        self.validateX = np.reshape(self.sample.validateImageX, (-1, nDimensions))
        self.validateY = np.squeeze(self.sample.validateImageY)

    '''导入已经保存的模型'''
    def loadModel(self):

        print("path:" , self.modelFilePath)
        self.model = joblib.load(self.modelFilePath)
        self.actionNames = self.model.actionNames
            
        
    '''设置模型文件的名称'''
    def setModelFileName(self , fileName):

        self.modelFileName = fileName

    def getSystermType(self):

        self.systermType = platform.system()

    '''设置模型文件的路径'''
    def setFilePath(self):

        currentFilePath = os.getcwd()
        if self.systermType == "Windows":
            self.modelFilePath = currentFilePath + '//data//model//'
            self.filePathName = currentFilePath + '//data//filePathFile.txt'
        elif self.systermType == "Linux":
            self.modelFilePath = currentFilePath + '/data/model/'
            self.filePathName = currentFilePath + '/data/filePathFile.txt'
        with open(self.filePathName , "r") as fp:
            self.modelFileName = str(fp.readlines()[1]) + "-model.pkl"
            fp.close()
        #判断文件夹是否存在，不存在则创建
        if os.path.exists(self.modelFilePath) == True:
            self.modelFileExitFlag = True
        else:
            print("The model is not saved, please save the model before using the model!!")

    #投票函数
    def getTheAction(self , actionList):

        tempData = np.array(actionList)
        counts = np.bincount(tempData)
        actionNumber = np.argmax(counts)
        return self.actionNames[actionNumber]#返回定义的动作类别字符串

    '''myo主程序的入口'''
    def run(self):

        try:
            self.mThread = myThread()
            self.mThread.addThread('onlineClf' , self.onlineClf , 0) #加入在线识别动作线程
            self.mThread.addThread('getData' , self.myoRun , 0) #加入数据采集的线程
            self.mThread.runThread()

        except KeyboardInterrupt:
            
            self.mMyo.safely_disconnect()
            self.mThread.stopThread()
            print ('Finished.')


if __name__ == "__main__":
    pass

    # myOnLineClf = onLineClf(dataMode = ["emg"])
    # myOnLineClf.start()
    # myOnLineClf.run()

