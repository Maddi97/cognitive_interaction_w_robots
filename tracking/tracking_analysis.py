from tracking.tracking_algorithms import emotion_detection
from tracking.tracking_algorithms.handTracker import HandTracker
from tracking.tracking_algorithms.bodyTracker import BodyTracker
import cv2
from sys import path
path.append('tracking/tracking_algorithms')  # for module import
import tracking_algorithms.emotion_detection


handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker()
emDec = emotion_detection.EmotionDetection()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    emotion, pred = emDec.predict_emotion(img)
    img = handTracker.track_hand(img)
    # img = bodyTracker.track_body(img)
    gesture, prediction = handTracker.predict_gesture(img)
    print(prediction)
    # cv2.startWindowThread()

    # cv2.putText(img, gesture, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(img, emotion, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow("Image", img)


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
