from ultralytics import YOLO
import cv2


vid_capture = cv2.VideoCapture("C:/Users/2nith/Downloads/1.mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = vid_capture.read()

    if not ret:
        break

    results = model.predict(frame)[0]
    # print(results.boxes.data.tolist())
    for data in results.boxes.data.tolist():
        confidence = data[4]
        if confidence<.50:
            continue
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),2)
        cv2.putText(frame,results.names[data[5]],(xmin, ymin+10),cv2.FONT_HERSHEY_SIMPLEX , .5, (0,0,255),2)

    cv2.imshow("video",frame)

    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()