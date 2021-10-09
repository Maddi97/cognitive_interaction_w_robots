from Tracking.handTracker import HandTracker
from Tracking.bodyTracker import BodyTracker
import cv2
import matplotlib.pyplot as plt

handTracker = HandTracker()
bodyTracker = BodyTracker()

cap = cv2.VideoCapture(0)
z_data = []

for i in range(1000):
    print(i)
    success, img = cap.read()
    # img = handTracker.track_hand(img)
    # img = bodyTracker.track_body(img)

    # mark_data = handTracker.get_landmark_data(img)

    # if len(mark_data) != 0:
    #     for e in mark_data:
    #         if e['id'] == 3:
    #             print(e['z']*1000)
    #             z_data.append(e['z']*1000)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.plot(z_data)
plt.show()
