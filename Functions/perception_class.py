#!/usr/bin/python3
# coding=utf8

import sys
import cv2
import time
import threading
import math
import numpy as np

sys.path.append('/home/pi/ArmPi/')

import Camera
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *

class ColorTracking:
    def __init__(self): 
        self.roi = ()
        self.rect = []
        self.count = 0
        self.track = False 
        self.get_roi = False
        self.center_list = []
        self.__isRunning = True
        self.unreachable = False
        self.detect_color = 'None'
        self.action_finish = False
        self.rotation_angle = 0
        self.last_x, self.last_y = 0, 0
        self.world_x, self.world_y = 0, 0
        self.world_X, self.world_Y = 0, 0
        self.start_count_t1 = True
        self.t1 = 0
        self.start_pick_up = False
        self.first_move = True
        self.AK = ArmIK()
        self.__target_color = ('red', 'blue', 'green')
        self.color_area_max = None
        
        # RGB color mapping
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
        }

    def getAreaMaxContour(self, contours):
        """Finds the largest contour based on area."""
        area_max_contour = max(contours, key=cv2.contourArea, default=None)
        contour_area_max = cv2.contourArea(area_max_contour) if area_max_contour is not None else 0
        return (area_max_contour, contour_area_max) if contour_area_max > 300 else (None, 0)

    def target_color_fun(self, frame_lab): 
        """Identifies and returns bounding boxes for all detected colors."""
        detected_boxes = []
        
        if not self.start_pick_up:
            for color in color_range:
                if color in self.__target_color:
                    mask = cv2.inRange(frame_lab, color_range[color][0], color_range[color][1])
                    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
                    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))
                    
                    contours = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 300:
                            rect = cv2.minAreaRect(contour)
                            box = np.int0(cv2.boxPoints(rect))
                            detected_boxes.append((color, box, area))
        
        return detected_boxes

    def find_distance(self): 
        """Tracks movement based on position changes."""
        if self.distance < 0.3:
            self.center_list.append((self.world_x, self.world_y))
            self.count += 1
            if self.start_count_t1:
                self.start_count_t1 = False
                self.t1 = time.time()
        return self.t1
    
    def time_fun(self):
        """Updates position tracking after a delay."""
        if time.time() - self.t1 > 1.5:
            self.rotation_angle = self.rect[2]
            self.start_count_t1 = True
            self.world_X, self.world_Y = np.mean(np.array(self.center_list).reshape(self.count, 2), axis=0)
            self.count = 0
            self.center_list = []
            self.start_pick_up = True
    
    def run(self, img): 
        """Processes the image frame to track color objects."""
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        cv2.line(img, (0, img_h // 2), (img_w, img_h // 2), (0, 0, 200), 1)
        cv2.line(img, (img_w // 2, 0), (img_w // 2, img_h), (0, 0, 200), 1)
        
        if not self.__isRunning:
            return img
        
        frame_resized = cv2.resize(img_copy, (640, 480), interpolation=cv2.INTER_NEAREST)
        frame_blurred = cv2.GaussianBlur(frame_resized, (11, 11), 11)
        
        if self.get_roi and self.start_pick_up:
            self.get_roi = False
            frame_blurred = getMaskROI(frame_blurred, self.roi, (640, 480))
        
        frame_lab = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2LAB)
        found, box, area_max = self.target_color_fun(frame_lab)
        if not found:
            return img
        
        self.roi = getROI(box)
        self.get_roi = True
        img_centerx, img_centery = getCenter(self.rect, self.roi, (640, 480), 10)
        self.world_x, self.world_y = convertCoordinate(img_centerx, img_centery, (640, 480))
        
        if area_max > 2500:
            cv2.drawContours(img, [box], -1, self.range_rgb[self.color_area_max], 2)
            cv2.putText(img, f'({self.world_x},{self.world_y})', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[self.color_area_max], 1)
            
            self.distance = math.hypot(self.world_x - self.last_x, self.world_y - self.last_y)
            self.last_x, self.last_y = self.world_x, self.world_y
            self.track = True
            
            if self.action_finish:
                self.t1 = self.find_distance()
                self.time_fun()
            else:
                self.t1 = time.time()
                self.start_count_t1 = True
                self.count = 0
                self.center_list = []
        
        return img
    
    def run2(self, img): 
        """Processes the image frame to track multiple color objects simultaneously."""
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        cv2.line(img, (0, img_h // 2), (img_w, img_h // 2), (0, 0, 200), 1)
        cv2.line(img, (img_w // 2, 0), (img_w // 2, img_h), (0, 0, 200), 1)
        
        if not self.__isRunning:
            return img
        
        frame_resized = cv2.resize(img_copy, (640, 480), interpolation=cv2.INTER_NEAREST)
        frame_blurred = cv2.GaussianBlur(frame_resized, (11, 11), 11)
        
        if self.get_roi and self.start_pick_up:
            self.get_roi = False
            frame_blurred = getMaskROI(frame_blurred, self.roi, (640, 480))
        
        frame_lab = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2LAB)
        detected_boxes = self.target_color_fun(frame_lab)

        # Reset tracking variables
        self.track = False  

        for color, box, area in detected_boxes:
            if area > 2500:
                self.track = True  # At least one object is detected
                cv2.drawContours(img, [box], -1, self.range_rgb[color], 2)
                img_centerx, img_centery = getCenter(cv2.minAreaRect(box), None, (640, 480), 10)
                world_x, world_y = convertCoordinate(img_centerx, img_centery, (640, 480))

                # Display the coordinates
                cv2.putText(img, f'({world_x},{world_y})', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[color], 1)
                
                # Compute movement tracking variables
                self.distance = math.hypot(world_x - self.last_x, world_y - self.last_y)
                self.last_x, self.last_y = world_x, world_y

                # Time-based movement tracking
                if self.action_finish:
                    self.t1 = self.find_distance()
                    self.time_fun()
                else:
                    self.t1 = time.time()
                    self.start_count_t1 = True
                    self.count = 0
                    self.center_list = []

        return img
    
    def main(self): 
        my_camera = Camera.Camera()
        my_camera.camera_open()

        while True:
            img = my_camera.frame
            if img is not None:
                frame = img.copy()
                Frame = self.run2(frame)   
                cv2.imshow('Frame', Frame)
                if cv2.waitKey(1) == 27:
                    break

        my_camera.camera_close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    color_tracking = ColorTracking()
    color_tracking.main()
