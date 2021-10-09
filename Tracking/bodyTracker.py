import cv2
import mediapipe as mp


class BodyTracker:

    def __init__(self, mode=False, upBody=0, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def track_body(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img):
        landmarks = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_obj = {'id': id, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'cx': cx, 'cy': cy}
                landmarks.append(landmark_obj)

        return landmarks
