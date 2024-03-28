# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#configurations_options
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math

class PoseDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.landmarks_t = np.zeros((33, 2))

    def euclidean(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def find_angle(self,Ax, Ay, Ox, Oy, Bx, By):
        OA_x = Ax - Ox
        OA_y = Ay - Oy
        OB_x = Bx - Ox
        OB_y = By - Oy
        
        dp = OA_x * OB_x + OA_y * OB_y
        mag_OA = math.sqrt(OA_x**2 + OA_y**2)
        mag_OB = math.sqrt(OB_x**2 + OB_y**2)
        try:
            theta = math.acos(dp / (mag_OA * mag_OB))
        except:
            theta = 100
        
        return theta
    
    def draw_landmarks(self, frame,classified_pose='warrior1'):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frameRGB)
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id_results, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # lm is the ratio of the image. So we multiply it by the width and height to get the pixel value
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks_t[id_results] = [cx, cy]
                if classified_pose == 'tree':
                    flag = self.check_bad_tree(frame)
                elif classified_pose == 'warrior1':
                    flag = self.check_bad_warrior1(frame)
                elif classified_pose == 'downdog':
                    flag = self.check_bad_downdog(frame)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, f'{id_results}', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return frame
    
    def check_sideways(self, frame, right_shoulder, left_shoulder,good=False):
        if not good:
            right = (0, 255, 0)
            wrong = (0,0,255)
        else:
            right = (0,0,255)
            wrong = (0, 255, 0)
        sideways_check = self.euclidean(right_shoulder[0], right_shoulder[1], left_shoulder[0], left_shoulder[1]) < 80
        if sideways_check:
            cv2.putText(frame, f'You are facing sideways',(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, right, 2)
        else:
            cv2.putText(frame,f'You are facing front',(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, wrong, 2)
        return sideways_check
    
    def check_leg_angle(self,frame,angle_val,threshold_low=1.2,threshold_high=1.6):
        right = (0, 255, 0)
        wrong = (0,0,255)
        if threshold_low < angle_val < threshold_high:
            cv2.putText(frame, f'Your leg is bent',(10,90),cv2.FONT_HERSHEY_SIMPLEX, 1, right, 2)   
        else:
            cv2.putText(frame,f'Your leg is straight',(10,90),cv2.FONT_HERSHEY_SIMPLEX, 1, wrong, 2)

    def check_bad_warrior1(self,frame):
        flag = False
        # to check bad warrior pose I will simply be checking if:
        # 1. The subject is standing sideways
        # 2. The subject's leg is bent at the knee/ at some angle
        right_shoulder = self.landmarks_t[12]
        left_shoulder = self.landmarks_t[11]
        # O = knee, A = hip, B = ankle
        right_O = self.landmarks_t[26]
        right_A = self.landmarks_t[24]
        right_B = self.landmarks_t[28]
        
        if self.check_sideways(frame, right_shoulder, left_shoulder,good=False):
            angle_val = self.find_angle(right_A[0], right_A[1], right_O[0], right_O[1], right_B[0], right_B[1])
            flag = self.check_leg_angle(frame,angle_val,threshold_low=1.2,threshold_high=1.6)
        
        return flag and self.check_sideways(frame, right_shoulder, left_shoulder,good=False)

    def check_bad_tree(self,frame):
        flag = False
        # to check bad tree pose I will be checking if:
        # 1. The subject is standing straight/facing the camera
        # 2. The subject's leg is bent at the knee/ at some angle away from the other leg
        right_shoulder = self.landmarks_t[12]
        left_shoulder = self.landmarks_t[11]

        # O = knee, A = hip, B = ankle
        right_O = self.landmarks_t[26]
        right_A = self.landmarks_t[24]
        right_B = self.landmarks_t[28]
        
        if not self.check_sideways(frame, right_shoulder, left_shoulder,good=True):
            angle_val = self.find_angle(right_A[0], right_A[1], right_O[0], right_O[1], right_B[0], right_B[1])
            flag = self.check_leg_angle(frame,angle_val,threshold_low=0.9,threshold_high=1.1)
        return flag and self.check_sideways(frame, right_shoulder, left_shoulder,good=True)
    
    def check_bad_downdog(self,frame):
        # to check bad downdog pose I will be checking if:
        # 1. The subject is standing sideways
        # 2. The subject's body is arched
        pass

    

    

if __name__=='__main__':
    vid = cv2.VideoCapture(1)
    detector = PoseDetector()

    while True:
        ret,frame = vid.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = detector.draw_landmarks(frame)
            
        fps = vid.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break