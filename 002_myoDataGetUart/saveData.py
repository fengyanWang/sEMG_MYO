#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from myo_raw import MyoRaw

import sys
import time
import uuid

import numpy as np
import csv 
import os
import platform

import time

class EMGHandler(object):
	
	def __init__(self, m):

		self.recording = -1
		self.m = m
		
	def __call__(self, emg, moving):

		if self.recording >= 0:
			if len(self.m.emg_data) >= self.m.emgDataSize:
				self.m.createEmgFile(self.m.emg_data , self.m.actionCategry)
				self.m.creatImuFile(self.m.quat_data , self.m.actionCategry , 'quat')
				self.m.creatImuFile(self.m.acc_data , self.m.actionCategry , 'acc')
				self.m.creatImuFile(self.m.gyr_data , self.m.actionCategry , 'gyr')
				if self.m.dataCounter == 0:
					print("the length of unit data is :" + str(self.m.displayLength))
				if len(self.m.emg_data) == self.m.displayLength:
					self.m.dataCounter += 1
					print ('actionCategry is :' + str(self.m.actionCategry) + '-->>'+ str(self.m.dataCounter) + '-->>' + str(self.m.displayLength * self.m.dataCounter))
				self.m.emg_data = []
				self.m.quat_data = []
				self.m.acc_data = []
				self.m.gyr_data = []
		else:
			self.m.dataCounter = 0
			self.m.emg_data = []
			self.m.quat_data = []
			self.m.acc_data = []
			self.m.gyr_data = []


class myDevice(object):

	def __init__(self):

		self.emgDataPath = ''
		self.imuDataPath = ''
		self.emgDataSize = 100
		self.emg_data = []
		self.quat_data = []
		self.acc_data = []
		self.gyr_data = []
		self.actionCategry = -1 #动作的类别
		self.displayLength = 100 #the length of data to display
		self.dataCounter = 0
		self.userName = 'wfy' #eg:wangfengyan -> wfy
		self.userPosture = 'stand' # stand or sit ..stc
		self.userSex = 'male' #male or female

		self.device = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
		self.recording = -1

		self.setFilePath()
	'''
	emg data process
	'''
	def proc_emg(self,emg, moving):
		# emg = np.append(emg , self.userName)
		# emg = np.append(emg , self.userSex)
		# emg = np.append(emg , self.userPosture)
		# emg = np.append(emg,self.actionCategry) 
		# print("the emg data is :" , emg) 
		self.emg_data.append(emg)
	'''
	imu data process
	'''
	def proc_imu(self,quat, acc, gyr ):

		# quat = np.append(quat , self.userName )
		# acc = np.append(acc , self.userName )
		# gyr = np.append(gyr , self.userName )

		# quat = np.append(quat , self.userSex )
		# acc = np.append(acc , self.userSex )
		# gyr = np.append(gyr , self.userSex )

		# quat = np.append(quat , self.userPosture)
		# acc = np.append(acc , self.userPosture)
		# gyr = np.append(gyr ,  self.userPosture)

		# quat = np.append(quat ,self.actionCategry)
		# acc = np.append(acc , self.actionCategry)
		# gyr = np.append(gyr , self.actionCategry)

		self.quat_data.append( quat )
		self.acc_data.append(  acc  )
		self.gyr_data.append(  gyr  )

	def setFilePath(self):
		if platform.system() == 'Windows':
			splitStr = "\\"
		elif platform.system() == 'Linux': 
			splitStr = "/"
		currentFilePath = os.getcwd()
		self.emgDataPath = currentFilePath + splitStr +'data'+ splitStr +'emgData'
		self.imuDataPath = currentFilePath + splitStr +'data'+ splitStr +'imuData'
		#判断文件夹是否存在，不存在则创建
		if os.path.exists(self.emgDataPath) == False:
		    os.makedirs(self.emgDataPath)
		    
		if os.path.exists(self.imuDataPath) == False:
		    os.makedirs(self.imuDataPath)
	'''
	creat emg files
	'''
	def createEmgFile(self , data , actionCategry = 1 ):

		os.chdir(self.emgDataPath)   
		fileName = 'emg' + '_' + str(actionCategry)
		fileExistFlag = os.path.isfile(fileName + '.csv')
		with open(fileName + '.csv' ,'a+') as f:
			writer = csv.writer(f)
			if fileExistFlag == False:
				writer.writerow(["Channel 0","Channel 1",'Channel 2','Channel 3',
								'Channel 4','Channel 5','Channel 6','Channel 7',
								'userName','userSex','userPosture','actionCategry'])
			writer.writerows(data)
	'''
	creat imu data
	'''
	def creatImuFile(self , data , actionCategry  =1 ,dataFlag = 'acc'):
		os.chdir(self.imuDataPath)
		fileName = str(dataFlag) + '_' + str(actionCategry)
		fileExistFlag = os.path.isfile(fileName + '.csv')
		with open(fileName + '.csv' ,'a+') as f:
			writer = csv.writer(f)
			if dataFlag == 'quat':
				if fileExistFlag == False:
					writer.writerow(['q1','q2','q3','q4','userName','userSex','userPosture','actionCategry'])
			elif dataFlag == 'acc':
				if fileExistFlag == False:
					writer.writerow(['ax','ay','az','userName','userSex','userPosture','actionCategry'])
			elif dataFlag == 'gyr':
				if fileExistFlag == False:
					writer.writerow(['gx','gy','gz','userName','userSex','userPosture','actionCategry'])
			writer.writerows(data)
	

	def getUserInformation(self):

		self.userName = raw_input('please input your name:')
		self.userSex = raw_input('please input your sex-->>(male or female):')
		self.userPosture = raw_input('please input your posture-->>(stand or sit):')
		
class myMyo(object):

	def __init__(self):
		
		self.mDevice = myDevice()
		self.hnd = EMGHandler(self.mDevice) 

	def start(self):

		self.mDevice.device.add_emg_handler(self.hnd)
		self.mDevice.device.add_emg_handler(self.mDevice.proc_emg)
		self.mDevice.device.add_imu_handler(self.mDevice.proc_imu)

		self.mDevice.device.connect()

	def run(self):
		try:
			while True:
				if self.mDevice.device.run() is None:
					print("disconnect , connect again!")
					self.mDevice.device.connect()
				self.mDevice.device.run()
		except KeyboardInterrupt:
			self.mDevice.device.disconnect()
			print ("\nDone.\n")


if __name__ == "__main__":

	mMyo = myMyo()
	mMyo.start()
	mMyo.run()












