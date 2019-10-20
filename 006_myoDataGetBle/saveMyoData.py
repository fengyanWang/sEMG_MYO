#coding:UTF-8
from __future__ import print_function

import open_myo as myo
from open_myo import EmgMode ,  ImuMode
import numpy as np
import timeit
import time
import pickle
import os
import sys
import pandas as pd
import csv

class saveMyoData(object):

    def __init__(self , mEmgMode = myo.EmgMode.RAW , mImuMode = myo.ImuMode.RAW ):

        self.emgGestures = dict() #保存肌电数据的字典
        self.accGestures = dict() #保存姿态数据的字典
        self.gyroGestures = dict() #保存姿态数据的字典
        self.quatGestures = dict() #保存姿态数据的字典
        #读取emg和imu数据的格式
        self.emgMode = mEmgMode
        self.imuMode = mImuMode

        self.fileMode = "csv"  #csv or pkl

        #计算采样率
        self.setTimeStampFlag = False
        self.emgSampleCounter = 0
        self.imuSampleCounter = 0

        #数据打标签
        self.get_reading = False #判断是否开始读取数据
        self.n_gestures = 0     #动作类别的个数
        self.n_iterations = 0   #每个动作类别重复的个数
        self.runtime = 0    #每个动作每次维持的时间
        self.gesturesName = 0 #动作的名称
        self.iteration = 0  #当前重复动作的个数

        #数据保存的路径
        self.emgDataPath = ''
        self.imuDataPath = ''

        self.pythonVersion = 0
        print("1")
        self.myo_device = myo.Device()
        print("2")
        self.myo_device.services.sleep_mode(1) # never sleep
        print("3")
        self.myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)
        
    
    def start(self):
        self.myo_device.services.vibrate(1) # short vibration
        if self.emgMode == myo.EmgMode.RAW:
            self.myo_device.services.emg_raw_notifications()
        elif self.emgMode == myo.EmgMode.FILT:
            self.myo_device.services.emg_filt_notifications()

        if self.imuMode != myo.ImuMode.OFF:
            self.myo_device.services.imu_notifications()

        self.myo_device.services.set_mode(self.emgMode , self.imuMode, myo.ClassifierMode.OFF)
        time.sleep(1)
        self.getPythonVersion()

        if self.emgMode != myo.EmgMode.OFF:
            self.myo_device.add_emg_event_handler(self.process_emg)
        if self.imuMode != myo.ImuMode.OFF:
            self.myo_device.add_imu_event_handler(self.process_imu)

    #肌电数据处理函数
    def process_emg(self,emg):
        if self.get_reading:
            self.emgSampleCounter += 2
            if self.setTimeStampFlag == True:
                timeStamp = time.time()
                emg[0] = list(emg[0])
                emg[0].append(timeStamp)
            self.emgGestures[self.gesturesName][self.iteration].append(emg[0])
            if self.setTimeStampFlag == True:
                timeStamp = time.time()
                emg[1] = list(emg[1])
                emg[1].append(timeStamp)
            self.emgGestures[self.gesturesName][self.iteration].append(emg[1])
    #
    def Dict2dataFrame(self ,emgDict , dataMode):

        n_iterations = [len(value) for value in emgDict.values()][0]
        Data = []
        myData = []
        for k in  emgDict.keys():
            for i in range(n_iterations):
                myData = emgDict[k][i]
                for j in myData:
                    tempData = list(j)
                    tempData.append(k)
                    Data.append(tempData)
        if dataMode == "emg":
            if self.setTimeStampFlag == True:
                return  pd.DataFrame(Data , columns = ["ch0"  , "ch1" , 'ch2' , "ch3" , 'ch4' ,"ch5" , 'ch6' ,"ch7" , 'time' , 'action'])
            else:
                return  pd.DataFrame(Data , columns = ["ch0"  , "ch1" , 'ch2' , "ch3" , 'ch4' ,"ch5" , 'ch6' ,"ch7" , 'action'])
        elif dataMode == "acc":
            if self.setTimeStampFlag == True:
                return pd.DataFrame(Data , columns = ["ax"  , "ay" , 'az' , 'time' ,'action'])
            else:
                return pd.DataFrame(Data , columns = ["ax"  , "ay" , 'az' ,'action'])
        elif dataMode == "gyro":
            if self.setTimeStampFlag == True:
                return pd.DataFrame(Data , columns = ["gx"  , "gy" , 'gz' , 'time' ,'action'])
            else:
                return pd.DataFrame(Data , columns = ["gx"  , "gy" , 'gz' ,'action'])
        elif dataMode == "quat":
            if self.setTimeStampFlag == True:
                return pd.DataFrame(Data , columns = ["q1"  , "q2" , 'q3' , 'q4' , 'time' ,'action'])
            else:
                return pd.DataFrame(Data , columns = ["q1"  , "q2" , 'q3' , 'q4' ,'action'])


    #肌电数据保存函数
    def emgDataSave(self,emgData):

        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = "emg_data_"+ timestr
        tempData = self.Dict2dataFrame(emgData , 'emg')

        if self.fileMode == 'csv' :
            self.emgDataFilePath = self.emgDataPath +'/' + filename+".csv"
            # print("self.emgDataFilePath is :" , self.emgDataFilePath)
            with open(self.emgDataPath +'/' + filename+".csv", 'a+') as fp:
                tempData.to_csv(fp , sep=',', header=True, index=True)
        elif self.fileMode == "pkl":
            self.emgDataFilePath = self.emgDataPath +'/' + filename+".pkl"
            with open(self.emgDataPath +'/' + filename+".pkl", 'wb') as fp:
                tempData.to_pickle(fp , sep=',', header=True, index=True)

    #肌电数据处理函数
    def process_imu(self , quat, acc, gyro):

        if self.get_reading:
            self.imuSampleCounter += 1

            if self.setTimeStampFlag == True:
                timeStamp = time.time()
                acc = list(acc)
                acc.append(timeStamp)
            self.accGestures[self.gesturesName][self.iteration].append(acc)
            
            if self.setTimeStampFlag == True:
                timeStamp = time.time()
                gyro = list(gyro)
                gyro.append(timeStamp)
            self.gyroGestures[self.gesturesName][self.iteration].append(gyro)
            
            if self.setTimeStampFlag == True:
                timeStamp = time.time()
                quat = list(quat)
                quat.append(timeStamp)
            self.quatGestures[self.gesturesName][self.iteration].append(quat)
    

    #肌电数据保存程序
    def imuDataSave(self , accData , gyroData , quatData):

        timestr = time.strftime("%Y%m%d-%H%M%S")

        accFileName = "acc_data_" + timestr
        gyroFileName = "gyro_data_" + timestr
        quatFileName = "quat_data_" + timestr

        if self.fileMode == "pkl":
            self.accDataFilePath = self.imuDataPath +'/' + accFileName +".pkl"
            with open(self.imuDataPath +'/' + accFileName +".pkl", 'wb') as fp:
                pickle.dump(accData, fp)
            self.gyroDataFilePath = self.imuDataPath +'/' + accFileName +".pkl"
            with open(self.imuDataPath +'/' + gyroFileName +".pkl", 'wb') as fp:
                pickle.dump(gyroData, fp)
            self.quatDataFilePath = self.imuDataPath +'/' + accFileName +".pkl"
            with open(self.imuDataPath +'/' + quatFileName +".pkl", 'wb') as fp:
                pickle.dump(quatData, fp)

        elif self.fileMode == "csv" :
            self.accDataFilePath = self.imuDataPath +'/' + accFileName+".csv"
            tempAccData = self.Dict2dataFrame(accData , 'acc')
            with open(self.imuDataPath +'/' + accFileName+".csv", 'a+') as fp:
                tempAccData.to_csv(fp , sep=',', header=True, index=True)

            self.gyroDataFilePath = self.imuDataPath +'/' + gyroFileName+".csv"
            tempGyroData = self.Dict2dataFrame(gyroData , 'gyro')
            with open(self.imuDataPath +'/' + gyroFileName+".csv", 'a+') as fp:
                tempGyroData.to_csv(fp , sep=',', header=True, index=True)

            self.quatDataFilePath = self.imuDataPath +'/' + quatFileName+".csv"
            tempQuatData = self.Dict2dataFrame(quatData , 'quat')  
            with open(self.imuDataPath +'/' + quatFileName+".csv", 'a+') as fp:
                tempQuatData.to_csv(fp , sep=',', header=True, index=True)
        

    def setFilePath(self):

        currentFilePath = os.getcwd()
        self.emgDataPath = currentFilePath + '/data/emgData'
        self.imuDataPath = currentFilePath + '/data/imuData'
        #判断文件夹是否存在，不存在则创建
        if os.path.exists(self.emgDataPath) == False:
            os.makedirs(self.emgDataPath)
            
        if os.path.exists(self.imuDataPath) == False:
            os.makedirs(self.imuDataPath)

    #在数据中增加时间戳
    def addTimeStamp(self):
        self.setTimeStampFlag = True
    #获得采集的肌电数据的个数
    def getEmgSampleCounter(self):
        if self.emgMode != myo.EmgMode.OFF:
            return self.emgSampleCounter
    #获得采集的姿态数据的个数
    def getImuSampleCounter(self):
        if self.imuMode != myo.ImuMode.OFF:
            return self.imuSampleCounter


    #获取用户输入的信息
    def getUserInput(self):

        if self.pythonVersion == "2":

            self.n_gestures = int(raw_input("How many gestures do you want to perform?: "))
            self.n_iterations = int(raw_input("How many times do you want to repeat each gesture?: "))
            self.runtime = int(raw_input("How many seconds do you want each gesture to last?: "))
        
        elif self.pythonVersion == "3":
            self.n_gestures = int(input("How many gestures do you want to perform?: "))
            self.n_iterations = int(input("How many times do you want to repeat each gesture?: "))
            self.runtime = int(input("How many seconds do you want each gesture to last?: "))
    
    def getPythonVersion(self):

        verisonStr = sys.version
        self.pythonVersion = verisonStr.split("|")[0].split(".")[0]



    #执行函数
    def run(self):

        self.getUserInput()

        for g in range(self.n_gestures):

            if self.pythonVersion == "2":
                self.gesturesName = raw_input("Enter the name of gesture number {}: ".format(g+1))
            elif self.pythonVersion == "3":
                self.gesturesName = input("Enter the name of gesture number {}: ".format(g+1))
            
            if self.emgMode != myo.EmgMode.OFF:
                self.emgGestures[self.gesturesName] = list()
            if self.imuMode != myo.ImuMode.OFF:
                self.accGestures[self.gesturesName] = list()
                self.gyroGestures[self.gesturesName] = list()
                self.quatGestures[self.gesturesName] = list()

            for self.iteration in range(self.n_iterations):

                if self.emgMode != myo.EmgMode.OFF:
                    self.emgGestures[self.gesturesName].append(list())
                if self.imuMode != myo.ImuMode.OFF:
                    self.accGestures[self.gesturesName].append(list())
                    self.gyroGestures[self.gesturesName].append(list())
                    self.quatGestures[self.gesturesName].append(list())

                if self.pythonVersion == "2":
                    raw_input("Iteration {}. Press enter to begin recording.".format(self.iteration+1))
                elif self.pythonVersion == "3":
                    input("Iteration {}. Press enter to begin recording.".format(self.iteration+1))
                
                
                self.myo_device.services.vibrate(1) # short vibration
                time.sleep(0)
                start_time = timeit.default_timer()
                tick = start_time
                self.get_reading = True
                startTime = time.time()
                while round(tick - start_time, 1) <= self.runtime:
                    if self.myo_device.services.waitForNotifications(1):
                        tick = timeit.default_timer()
                    else:
                        print("Waiting...")
                        continue
                            
                self.get_reading = False
                self.myo_device.services.vibrate(1) # short vibration
                time.sleep(0)

        self.setFilePath()
        if self.emgMode != myo.EmgMode.OFF:
            self.emgDataSave(self.emgGestures)
        if self.imuMode != myo.ImuMode.OFF:
            self.imuDataSave(self.accGestures , self.gyroGestures , self.quatGestures)

if __name__ == "__main__":

    mySaveMyoData = saveMyoData(mEmgMode = myo.EmgMode.RAW , mImuMode = myo.ImuMode.RAW)
    mySaveMyoData.addTimeStamp()
    mySaveMyoData.start()
    mySaveMyoData.run()

        