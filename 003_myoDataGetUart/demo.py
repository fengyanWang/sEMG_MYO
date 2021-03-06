#! /usr/bin/env python
#coding:UTF-8
from __future__ import print_function

import sys
import os
import csv 
import pygame
import time
import numpy as np
from myThread import myThread
from myo import Myo , PoseType , VibrationType , DeviceListener


actionCategry = 0  #记录动作类别

#监听myo数据的类
class PrintPoseListener(DeviceListener):

	def __init__(self , dataType = ["emg" , "imu" , "pose"]):

		self.dataType = dataType #数据的类型
		self.emgData = [] #用于存储肌电数据
		
	#姿态处理函数
	def on_pose(self, pose):

		if "pose" in self.dataType:
			pose_type = PoseType(pose)
			if(pose_type.name=="FIST"):
				print ("im fist!!!!")
			if(pose_type.name=="REST"):
				print ("im REST!!!!" )          
			if(pose_type.name=="WAVE_OUT"):
				print ("im WAVE_OUT!!!!")
			if(pose_type.name=="WAVE_IN"):
				print ("im WAVE_IN!!!!")
			if(pose_type.name=="DOUBLE_TAP"):
				print ("im DOUBLE_TAP!!!!")
		else:
			pass
	#肌电数据处理函数
	def on_emg(self, emg, moving):

		global actionCategry

		if "emg" in self.dataType:
			tempEmgData = np.append(emg , actionCategry) 
			self.emgData.append(tempEmgData)
		else:
			pass
	#姿态数据处理函数
	def on_imu(self , quat, acc, gyro):
		if "imu" in self.dataType:
			print ("the quat is :" + str(quat))
			print ("the acc is :" + str(acc))
			print ("the gyro is :" + str(gyro))
		else:
			pass

#myo数据保存函数
class saveMyoData(object):

	def __init__(self):

		self.actionNum = 0  #用于对动作类别进行计数
		self.actionList = ["one" , "two" , "three" , "four" , "five" , "open_hand" , "ok"
							"low" , "love" , "good" , "fist" , "eight" , "crasp"] #保存的动作列表
		self.selectActionList = [] #永户选用的动作列表			
		self.stopFlag = False #停止采集数据标志

		self.actionTime = 0 #动作持续的时间（s）
		self.restTime = 0 #修息状态持续的时间(s)
	
	#开始函数 	
	def start(self):

		self.listener = PrintPoseListener(dataType = ["emg"]) 
		self.mMyo = Myo()
		self.setFilePath() #设置肌电数据的存储路径
		self.getUserInput() #获得用户输入的信息

		pygame.init()  #初始化显示窗口
		self.screen = pygame.display.set_mode((200,250)) #设置显示窗口的大小

		try:
			self.mMyo.connect() 
			self.mMyo.add_listener(self.listener)
			self.mMyo.vibrate(VibrationType.SHORT)

		except ValueError as ex:
			print (ex)
	#获取用户的输入信息
	def getUserInput(self):

		self.actionTime = int(raw_input("请输入动作时间："))
		self.restTime = int(raw_input("请输入休息时间："))

		print("--------------------------------------------------------")
		print("目前支持的动作为：" + str(self.actionList) )
		print("--------------------------------------------------------")
		print("请输入您想要训练的动作，结束请按“ctrl+c” : ")
		try:
			while True: 
				tempAction = raw_input(">>:") #获取用户的动作选择输入
				if tempAction in self.actionList:
					self.selectActionList.append(tempAction)
				else:
					print("输入的动作有误，请确认后重新输入！")
					print("--------------------------------------------------------")
					print("目前支持的动作为：" + str(self.actionList) )
					print("--------------------------------------------------------")

		except KeyboardInterrupt:
			print("")
			print ('Finished.')

	#设置图片的显示，图片显示线程主函数
	def setActionState(self , actionTime , restTime):

		global actionCategry
		currentFilePath = os.getcwd()

		while True:
			actionPicPath = currentFilePath + "/pose/" + str(self.selectActionList[self.actionNum]) + ".JPG" 
			restPicPath = currentFilePath + "/pose/" + "rest.JPG"
			#action
			img = pygame.image.load(actionPicPath).convert_alpha()
			img = pygame.transform.scale(img, (200,250))
			actionCategry = self.selectActionList[self.actionNum]
			pygame.display.set_caption(self.selectActionList[self.actionNum])
			font = pygame.font.Font(None, 26 )
			self.screen.blit(img, (0,0))
			pygame.display.update()
			time.sleep(actionTime)
			# #rest
			if self.restTime == 0:
				pass
			else:
				img = pygame.image.load(restPicPath).convert_alpha()
				img = pygame.transform.scale(img, (200,250))
				actionCategry = "rest"
				pygame.display.set_caption("rest")
				font = pygame.font.Font(None, 26 )
				self.screen.blit(img, (0,0))
				pygame.display.update()
				time.sleep(restTime)

			self.actionNum += 1
			if self.actionNum >= len(self.selectActionList):
				self.stopFlag = True
				break
	#设置肌电数据的保存路径	
	def setFilePath(self):

		currentFilePath = os.getcwd()
		timeStr = str(time.time())
		self.emgDataPath = currentFilePath  +'/data/emgData/' + timeStr 
		#判断文件夹是否存在，不存在则创建
		if os.path.exists(self.emgDataPath) == False:
		    os.makedirs(self.emgDataPath)
	#保存肌电数据
	def saveEmgData(self , data):

		os.chdir(self.emgDataPath)
		timeStr = str(time.time())
		fileName = 'emg_' +  timeStr
		fileExistFlag = os.path.isfile(fileName + '.csv')
		with open(fileName + '.csv' ,'a+') as f:
			writer = csv.writer(f)
			if fileExistFlag == False:
				writer.writerow(["Channel 0","Channel 1",'Channel 2','Channel 3',
								'Channel 4','Channel 5','Channel 6','Channel 7' , 'label']) #为肌电数据设置列索引
			writer.writerows(data) #保存数据为csv格式
	#数据采集线程主函数   
	def myoRun(self):

		while True:
			self.mMyo.run()
			if self.stopFlag == True:
				self.mMyo.vibrate(VibrationType.SHORT)
				self.mMyo.safely_disconnect()
				self.saveEmgData(self.listener.emgData)
				print("save data finish!!!!!")
				break
	#运行程序
	def run(self):

		try:
			self.mThread = myThread()
			self.mThread.addThread('setPic' , targetFunc = self.setActionState , haveArgsFlag = 1 , args = (self.actionTime,self.restTime))
			self.mThread.addThread('getData' , self.myoRun , 0) #加入数据采集的线程
			self.mThread.runThread()

		except KeyboardInterrupt:
			
			self.mMyo.safely_disconnect()
			print ('Finished.')
	  

if __name__ == '__main__':

	mSaveMyoData = saveMyoData()
	mSaveMyoData.start()
	try:
		mSaveMyoData.run()
	except KeyboardInterrupt:
		mSaveMyoData.mThread.stopThread()

		

	



		 

