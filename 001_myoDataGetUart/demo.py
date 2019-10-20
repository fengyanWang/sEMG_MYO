#! /usr/bin/env python
#coding:UTF-8
from __future__ import print_function

import sys
import time
from myo import Myo , PoseType , VibrationType , DeviceListener

class PrintPoseListener(DeviceListener):

    def __init__(self , dataType = ["emg" , "imu" , "pose"]):
        self.dataType = dataType

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

    def on_emg(self, emg, moving):
        if "emg" in self.dataType:
            print ("the emg is :" + str(emg))
        else:
            pass
    def on_imu(self , quat, acc, gyro):
        if "imu" in self.dataType:
            print ("the quat is :" + str(quat))
            print ("the acc is :" + str(acc))
            print ("the gyro is :" + str(gyro))
        else:
            pass
        
if __name__ == '__main__':

    print ('Start Myo for Linux' )

    listener = PrintPoseListener(dataType = ["emg"])
    myo = Myo()
    try:
        myo.connect()
        myo.add_listener(listener)
        myo.vibrate(VibrationType.SHORT)
        while True:
            myo.run()
    except KeyboardInterrupt:
        pass
    except ValueError as ex:
        print (ex)

    finally:
        myo.safely_disconnect()
        print ('Finished.')


        

    



         

