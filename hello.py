print("hello")



import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
cap=cv2.VideoCapture(0)
#setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret,frame=cap.read()
#         #Recolor image to RGB
#         image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         image.flags.writeable=False
#         #Make dtection
#         results=pose.process(image)
#         # print(results)
#         #recolor back to BGR
#         image.flags.writeable=True
#         image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#         #Render detection
#         mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

#         cv2.imshow("Media pipe",image)
#         if cv2.waitKey(10) & 0xFF==ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

cap=cv2.VideoCapture(0)
#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        #Recolor image to RGB
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        #Make dtection
        results=pose.process(image)
        # print(results)
        #recolor back to BGR
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        #Render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,177,66),thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )

        cv2.imshow("Media pipe",image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()