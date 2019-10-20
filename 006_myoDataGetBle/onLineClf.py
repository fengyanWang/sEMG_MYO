#coding:UTF-8
from __future__ import print_function

import open_myo as myo
import numpy as np
from FeatureSpace import FeatureSpace
import os
from sklearn.externals import joblib
import time

from myClf import myModel

'''
* @details          :This class mainly completes an online recognition process of actions, 
*                    including model reading, online data collection, feature calculation, action recognition and voting
* @param dataMode   :The types of data sources used can make emg data or attitude data
* @return           :None
* @see              :None
* @note             :Since the relative path is used, there is no need to consider the data storage path. The environment is 
*                    raspberry pie 3b + python2.7 + sklearn 16
'''
class onLineClf(object):

    def __init__(self , dataMode = ["emg" , "imu"]):

        '''data mode'''
        self.dataMode = dataMode #采集的数据的种类，主要有：emg 和 imu
        self.emgDataMode = "raw" #肌电数据的类型，主要为原始和滤波后
        self.imuDataMode = "raw" #姿态数据的类型，主要为原始和滤波后   
        '''emg data'''
        self.emgData = []   #用于保存肌电数据
        self.emgDataLenth = 80  #每次采集多少个数据做一次特征提取和动作的识别
        self.emgDataCounter = 0 #对肌电数据个数进行计数
        '''model'''
        self.modelFilePath = ""  #分类模型保存的路径
        self.modelFileName = "emg_data_20180625-022909-model" #分类模型保存的文件的名字
        self.modelFileExitFlag = False #分类模型是否存在
        self.numberVoter = 3   #投票成员的个数

        '''init myo'''
        self.myo_mac_addr = myo.get_myo()  
        print("MAC address: %s" % self.myo_mac_addr)
        self.myo_device = myo.Device()  #myo硬件对象
        self.myo_device.services.sleep_mode(1) # never sleep

    '''初始化myo相关程序'''
    def start(self):
        self.myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)
        self.myo_device.services.vibrate(1) # short vibration
        fw = self.myo_device.services.firmware()
        print("Firmware version: %d.%d.%d.%d" % (fw[0], fw[1], fw[2], fw[3]))
        self.myo_device.services.battery_notifications()
        self.showBatt() #显示当前手环的电量

        if "emg" in self.dataMode:
            if self.emgDataMode == "raw":
                self.myo_device.services.emg_raw_notifications()
                self.myo_device.services.set_mode(myo.EmgMode.RAW, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
            elif self.emgDataMode == "filt":
                self.myo_device.services.emg_filt_notifications()
                self.myo_device.services.set_mode(myo.EmgMode.FILT, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
            self.myo_device.add_emg_event_handler(self.process_emg)
            # self.myo_device.add_emg_event_handler(self.led_emg)
        if "imu" in self.dataMode:
            if self.imuDataMode == "raw":
                self.myo_device.services.imu_notifications()
                self.myo_device.services.set_mode(myo.EmgMode.OFF, myo.ImuMode.RAW, myo.ClassifierMode.OFF)
            elif self.imuDataMode == "filt":
                self.myo_device.services.imu_notifications()
                self.myo_device.services.set_mode(myo.EmgMode.OFF, myo.ImuMode.RAW, myo.ClassifierMode.OFF)
            self.myo_device.add_imu_event_handler(self.process_imu)
        if "clf" in self.dataMode:
            self.myo_device.services.classifier_notifications()
            self.myo_device.services.set_mode(myo.EmgMode.OFF, myo.ImuMode.OFF, myo.ClassifierMode.ON)
            self.myo_device.add_classifier_event_hanlder(self.process_classifier)
        self.myo_device.add_sync_event_handler(self.process_sync) #同步手环
        self.setFilePath() #设置路径
        # self.setModelFileName("emg_data_20180615-021238-model.pkl")
        self.loadModel()#导入模型
        self.myo_device.services.vibrate(1) # 震动提醒导入完成
        
    '''myo主程序的入口'''
    def run(self):
        
        while True:
            if self.myo_device.services.waitForNotifications(1.0): #监听myo事件
                continue
            else:
                print("Waiting...")
                
    '''得到手环的电量'''
    def showBatt(self):
        self.batt = self.myo_device.services.battery()
        print("Battery level: %d" % self.batt)

    '''肌电数据处理程序'''
    def process_emg(self , emg):

        emgDict = dict()
        self.emgData.append(list(emg[0]))
        self.emgData.append(list(emg[1]))
        self.emgDataCounter += 1
    
        if self.emgDataCounter == self.model.mFeatures['LW'] / 2 + self.numberVoter - 1: #投票数为其加1
            # print("a sample is over!!!!")
            self.emgDataCounter = 0
            self.emgData = np.array(self.emgData , dtype = np.int64)
            self.emgData = self.emgData.T
            emgDict['one'] = self.emgData

            self.sample = FeatureSpace(rawDict = emgDict, 
                      moveNames = ['one',], #动作类别
                      ChList = [0,1,2,3,4,5,6,7],  #传感器的通道数目
                      features = self.model.mFeatures,   
                      one_hot = False ,
                      trainPercent=[1, 0, 0]    #是否进行onehot处理
                     )
            self.getTrainData()
            actionList = self.model.mModel.predict(self.trainX)

            print("the action is :" , self.getTheAction(actionList))
            self.emgData = []
            emgDict.clear()

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

    def getModelFilePath(self , modelFilePath):
        self.modelFilePath = modelFilePath


    '''导入已经保存的模型'''
    def loadModel(self):
        
        if self.modelFileExitFlag == True:
            self.model = joblib.load(self.modelFilePath)
            self.actionNames = self.model.actionNames
            print("load model done!!!")
        
    '''设置模型文件的名称'''
    def setModelFileName(self , fileName):

        self.modelFileName = fileName

    '''设置模型文件的路径'''
    def setFilePath(self):

        currentFilePath = os.getcwd()
        self.modelFilePath = currentFilePath + '/data/model/'
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

    '''姿态数据处理函数'''
    def process_imu(self , quat , acc , gyro):
        print("quat:" , quat)
        print("acc:" , acc)
        print("gyro:" , gyro)
    '''手环同步函数'''
    def process_sync(self , arm , x_direction):
        print("arm:" , arm)
        print("x_direction:" , x_direction)
    '''得到手环自带的分类结果'''
    def process_classifier(self , pose):
        print("pose:" , pose)
    '''手环的电量处理函数'''
    def process_battery(self , batt):
        print("Battery level: %d" % batt)
    '''控制LED显示效果'''
    def led_emg(self , emg):
        
        if(self.emgDataCounter == self.emgDataLenth / 2   ):
            self.myo_device.services.set_leds([255, 0, 0], [128, 128, 255])
        else:
            self.myo_device.services.set_leds([128, 128, 255], [128, 128, 255])

if __name__ == "__main__":

    myOnLineClf = onLineClf(dataMode = ["emg"])
    myOnLineClf.start()
    myOnLineClf.run()

