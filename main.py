import os
import sys
import cv2
from ultralytics import YOLO
import features_extractor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = YOLO("detect_webcam.pt")  # TODO: Name of model
cap = cv2.VideoCapture("Khanh_Minh.mp4")  # TODO: Name of video
frame_no = 2800  # TODO frame_start
threshold = 0.7
amount_of_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
image = cv2.imread("2157_3.png")  # TODO
# TODO tracking object with camera and coordinate be changed at the same time
feature_without_webcam = features_extractor.embedding(image)

feature_map = [None, None, None, None, None]
coordinate = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
time_appear = [0, 0, 0, 0, 0]


# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
# ret, frame = cap.read()
# results = model(frame)

# init features
# for result in results:
#     k = 0
#     for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             if class_id == 15:
#                 img = frame[y1:y2, x1:x2]
#                 feature_map[k] = features_extractor.embedding(img)
#                 coordinate[k] = [x1, y1]
#                 time_appear[k] += 1
#                 k += 1

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def index(detection, check_object):
    img, x, y = detection
    feature_img = features_extractor.embedding(img)
    cnt = -1
    for i in range(len(feature_map)):
        if check_object[i] == 0:
            cosine = cosine_similarity(feature_img, feature_map[i])
            if abs(cosine) > threshold:
                return i

            else:
                # cosine1 = cosine_similarity(feature_img, feature_without_webcam)
                x_old, y_old = coordinate[i]
                x_min = x_old - 35  # 35 is oscillate range
                x_max = x_old + 35
                y_min = y_old - 35
                y_max = y_old + 35

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return i
            cnt = i
    return cnt + 1


def main():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    while True:
        check_object = [0, 0, 0, 0, 0]
        results = model(frame)
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                img = frame[y1:y2, x1:x2]
                if class_id == 15 and score >= 0.75:  # 15 is class_id of webcam
                    detections.append([img, x1, y1])
            if len(detections) == 0:
                continue
            for detection in detections:
                i = index(detection, check_object)
                feature_map[i] = features_extractor.embedding(detection[0])
                coordinate[i] = (detection[1], detection[2])
                check_object[i] = 1
                time_appear[i] += 1
                frame = cv2.putText(
                    frame,
                    str(i),
                    coordinate[i],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    # print(time_appear)


if __name__ == "__main__":
    option = sys.argv[1]
    if option != "ON":
        sys.exit(0)
    sys.exit(main())
