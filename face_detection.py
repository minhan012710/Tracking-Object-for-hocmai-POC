import cv2
from ultralytics import YOLO

model = YOLO("detect_webcam.pt")


def detect_face(image):
    imgs = []
    xmins = []
    ymins = []
    results = model(image, verbose=False)[0]
    for result in results:
        boxes = result.boxes
        xmin, ymin, xmax, ymax = boxes.xyxy[0]
        xmax = int(xmax.item())
        ymax = int(ymax.item())
        xmin = int(xmin.item())
        ymin = int(ymin.item())
        # print(xmax.item(), ymax.item(), xmin.item(), ymin.item())
        img = image[ymin:ymax, xmin:xmax, :]
        # img = Image.fromarray(img)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        xmins.append(xmin)
        ymins.append(ymin)
    return imgs
