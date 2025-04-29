import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import math
import csv

# --------------------- People Detector using HOG ---------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --------------------- Kalman Filter Tracker ---------------------
class PersonTracker:
    def __init__(self, id, initial_position):
        self.id = id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 5.0
        self.kf.P *= 50.0
        self.kf.Q *= 0.01

        x, y = initial_position
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.last_position = (x, y)
        self.missing_frames = 0

    def update(self, measurement):
        self.kf.predict()
        self.kf.update(measurement)
        self.last_position = self.kf.x[:2].reshape(2)
        self.missing_frames = 0

    def predict(self):
        self.kf.predict()
        self.last_position = self.kf.x[:2].reshape(2)
        self.missing_frames += 1

    def get_velocity(self):
        vx, vy = self.kf.x[2][0], self.kf.x[3][0]
        return vx, vy

    def get_position(self):
        x, y = self.kf.x[0][0], self.kf.x[1][0]
        return int(x), int(y)

# --------------------- Helper ---------------------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --------------------- Main Video Loop ---------------------
cap = cv2.VideoCapture(r"C:\Users\JAYA SREE\c tutorial\proxy_and_me_eye\input\test.mp4")
trackers = []
person_id = 0
max_distance = 100
max_missing_frames = 10

total_detections = 0
correct_matches = 0

csv_file = open('output_tracking_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Person_ID', 'X', 'Y', 'Speed_m/s', 'Speed_km/h'])

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    frame = cv2.resize(frame, (800, 600))
    boxes, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
    centers = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes]
    total_detections += len(centers)

    predictions = [tracker.get_position() for tracker in trackers]
    cost_matrix = []

    for pred in predictions:
        row = []
        for center in centers:
            row.append(euclidean(pred, center))
        cost_matrix.append(row)

    matched_trackers = set()
    matched_detections = set()

    if cost_matrix:
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < max_distance:
                trackers[r].update(np.array(centers[c]))
                matched_trackers.add(r)
                matched_detections.add(c)
                correct_matches += 1
            else:
                trackers[r].predict()

    for i, tracker in enumerate(trackers):
        if i not in matched_trackers:
            tracker.predict()

    for i, center in enumerate(centers):
        if i not in matched_detections:
            trackers.append(PersonTracker(person_id, center))
            person_id += 1

    trackers = [t for t in trackers if t.missing_frames < max_missing_frames]

    for tracker in trackers:
        x, y = tracker.get_position()
        vx, vy = tracker.get_velocity()
        speed_m_per_s = math.sqrt(vx**2 + vy**2)
        speed_km_per_h = speed_m_per_s * 3.6

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID:{tracker.id} V:{speed_m_per_s:.1f}m/s", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.arrowedLine(frame, (x, y), (int(x+vx*5), int(y+vy*5)), (0, 0, 255), 2)

        csv_writer.writerow([frame_number, tracker.id, x, y, f"{speed_m_per_s:.2f}", f"{speed_km_per_h:.2f}"])

    cv2.imshow("Improved Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

# --------------------- Accuracy Calculation ---------------------
print(f"\nTotal Detections: {total_detections}")
print(f"Correct Matches: {correct_matches}")

if total_detections > 0:
    accuracy = (correct_matches / total_detections) * 100
    print(f"Tracking Accuracy: {accuracy:.2f}%")
else:
    print("No detections found, accuracy cannot be calculated.")
