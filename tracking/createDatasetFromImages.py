from tracking.tracking_algorithms.handTracker import HandTracker
from tracking.tracking_algorithms.bodyTracker import BodyTracker
from CONSTANTS import CLASSES
import pandas as pd
import numpy as np
import cv2
import os

PATH = '../assets/train_img/'

handTracker = HandTracker(mode=True)
bodyTracker = BodyTracker(mode=True, trackCon=0.8, detectionCon=0.8)

results = []
for c in CLASSES:
    for filename in os.listdir(PATH + c ):
        if filename.endswith(".jpg"):
            file = os.path.join(PATH, c, filename)
            img = cv2.flip(cv2.imread(file), 1)
            img = handTracker.track_hand(img)
            result = handTracker.get_landmark_data(img)

            img = bodyTracker.track_body(img)
            result2 = bodyTracker.get_landmark_data(img)[11:17]

            results.append(result + result2 + [c])
            cv2.imwrite(os.path.join(PATH, 'w_landmarks/' + str(len(results) - 1) + '.jpg'), img)
        else:
            continue


results = np.array(results)
df = pd.DataFrame(results)
df.columns = [*df.columns[:-1], 'y']


df.to_csv(PATH + 'df.csv')
