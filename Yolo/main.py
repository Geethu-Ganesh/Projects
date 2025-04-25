from ultralytics import YOLO
import cv2

img_path = "C:/Users/2nith/Downloads/UYYqo.jpg"
img = cv2.imread(img_path)
model = YOLO('yolov8m.pt')

results = model.predict(img_path)

result = results[0]

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(),2)
    print(class_id)
    print(cords)
    print(conf)
    cv2.rectangle(img,cords[0:2],cords[2:4],(0,0,255),2)
    cv2.putText(img,class_id,(cords[0],cords[1]-5),cv2.FONT_HERSHEY_SIMPLEX , .5, (0,0,255),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()