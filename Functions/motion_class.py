#!/usr/bin/python3
# coding=utf8

import sys
import cv2
import time
import threading
import math
import numpy as np

sys.path.append('/home/pi/ArmPi/')

from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *
from perception_class import ColorTracking

class Motion:
    def __init__(self):
        self.rect = None
        self.track = False
        self._stop = False
        self.get_roi = False
        self.unreachable = False
        self.__isRunning = True
        self.detect_color = 'None'
        self.action_finish = False
        self.rotation_angle = 0
        self.world_X, self.world_Y = 0, 0
        self.world_x, self.world_y = 0, 0
        self.center_list = []
        self.count = 0
        self.start_pick_up = True
        self.first_move = True
        self.servo1 = 500
        self.AK = ArmIK()

        # Target placement coordinates for different colors
        self.coordinate = {
            'red':   (-15 + 0.5, 12 - 0.5, 1.5),
            'green': (-15 + 0.5, 6 - 0.5,  1.5),
            'blue':  (-15 + 0.5, 0 - 0.5,  1.5),
    }

    def set_rgb(self, color):
        """Controls LED RGB lights based on detected color."""
        colors = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255)}
        rgb_value = colors.get(color, (0, 0, 0))
        for i in range(2):
            Board.RGB.setPixelColor(i, Board.PixelColor(*rgb_value))
        Board.RGB.show()

    def initMove(self):
        """Moves the arm to its initial position."""
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def setBuzzer(self, timer):
        """Activates the buzzer for a short duration."""
        Board.setBuzzer(1)
        time.sleep(timer)
        Board.setBuzzer(0)

    def move(self):
        """Controls the motion of the robotic arm based on detected objects."""
        while True:
            if self.__isRunning:
                if self.first_move and self.start_pick_up:
                    self.action_finish = False
                    self.set_rgb(self.detect_color)
                    self.setBuzzer(0.1)
                    
                    result = self.AK.setPitchRangeMoving((self.world_X, self.world_Y - 2, 5), -90, -90, 0)
                    self.unreachable = not result
                    
                    self.start_pick_up = False
                    self.first_move = False
                    self.action_finish = True
                
                elif not self.first_move and not self.unreachable:
                    self.set_rgb(self.detect_color)
                    if self.track:
                        if self.__isRunning:
                            self.AK.setPitchRangeMoving((self.world_x, self.world_y - 2, 5), -90, -90, 0, 20)
                            time.sleep(0.02)
                            self.track = False
                    
                    if self.start_pick_up:
                        self.action_finish = False
                        if not self.__isRunning:
                            continue

                        # Open gripper
                        Board.setBusServoPulse(1, self.servo1 - 280, 500)
                        servo2_angle = getAngle(self.world_X, self.world_Y, self.rotation_angle)
                        Board.setBusServoPulse(2, servo2_angle, 500)
                        time.sleep(0.8)
                        
                        if not self.__isRunning:
                            continue
                        self.AK.setPitchRangeMoving((self.world_X, self.world_Y, 2), -90, -90, 0, 1000)
                        time.sleep(2)

                        # Close gripper
                        if not self.__isRunning:
                            continue
                        Board.setBusServoPulse(1, self.servo1, 500)
                        time.sleep(1)

                        # Lift object
                        if not self.__isRunning:
                            continue
                        Board.setBusServoPulse(2, 500, 500)
                        self.AK.setPitchRangeMoving((self.world_X, self.world_Y, 12), -90, -90, 0, 1000)
                        time.sleep(1)

                        # Move to placement area
                        target = self.coordinate[self.detect_color]
                        result = self.AK.setPitchRangeMoving((target[0], target[1], 12), -90, -90, 0)
                        time.sleep(result[2] / 1000)

                        if not self.__isRunning:
                            continue
                        servo2_angle = getAngle(target[0], target[1], -90)
                        Board.setBusServoPulse(2, servo2_angle, 500)
                        time.sleep(0.5)

                        if not self.__isRunning:
                            continue
                        self.AK.setPitchRangeMoving((target[0], target[1], target[2] + 3), -90, -90, 0, 500)
                        time.sleep(0.5)

                        if not self.__isRunning:
                            continue
                        self.AK.setPitchRangeMoving(target, -90, -90, 0, 1000)
                        time.sleep(0.8)

                        # Release object
                        if not self.__isRunning:
                            continue
                        Board.setBusServoPulse(1, self.servo1 - 200, 500)
                        time.sleep(0.8)

                        if not self.__isRunning:
                            continue
                        self.AK.setPitchRangeMoving((target[0], target[1], 12), -90, -90, 0, 800)
                        time.sleep(0.8)

                        self.initMove()
                        time.sleep(1.5)

                        self.detect_color = 'None'
                        self.first_move = True
                        self.get_roi = False
                        self.action_finish = True
                        self.start_pick_up = False
                        self.set_rgb(self.detect_color)
                    else:
                        time.sleep(0.01)
            else:
                if self._stop:
                    self._stop = False
                    self.initMove()
                    time.sleep(1.5)
                time.sleep(0.01)


def run_motion():
    """Starts the color tracking and motion control threads."""
    motion = Motion()
    color_tracking = ColorTracking()
    
    color_tracking_thread = threading.Thread(target=color_tracking.main, daemon=True)
    color_tracking_thread.start()

    while True:
        if color_tracking.detect_color != 'None':
            motion.start_pick_up = True  
            motion.move()
            break  
        time.sleep(0.1)

if __name__ == '__main__':
    run_motion()
