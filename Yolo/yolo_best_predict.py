from ultralytics import YOLO
import cv2

img_path = "C:/Users/2nith/Downloads/960x0.jpg"
img = cv2.imread(img_path)

model = YOLO('D:/MentorNow/Projects/Image Processing/yolo/runs/detect/train4/weights/best.pt')

results = model.predict(img_path)[0]




for data in results.boxes.data.tolist():
    class_id = results.names[data[-1]]
    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    print(class_id)
    cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0,0,255),2)
    cv2.putText(img,class_id,(xmin, ymin-5),cv2.FONT_HERSHEY_SIMPLEX , .5, (0,0,255),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()