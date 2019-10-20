#coding:UTF-8

import time
import threading

class myThread(object):

	def __init__(self):

		self.threadList = []
		self.threadNameList = []

	def addThread(self, threadName , targetFunc , haveArgsFlag , args = (1,2)):
		if haveArgsFlag == 0:
			threadName = threading.Thread(target = targetFunc )
		else:
			threadName = threading.Thread(target = targetFunc , args = args)
		self.threadList.append(threadName)
		self.threadNameList.append(threadName)

	def delThread(self , threadName):
		if threadName in self.threadNameList:
			self.threadNameList.remove(threadName)
			self.threadList.remove(threadName)
		else:
			print(threadName + ' not be creat!!!!!!')
			
	def runThread(self):

		for t in self.threadList:
			t.start()

	def stopThread(self):
		for t in self.threadList:
			t.join()