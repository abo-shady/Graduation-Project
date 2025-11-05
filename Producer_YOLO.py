from ultralytics import YOLO
from kafka import KafkaProducer
import cv2
import json
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test.mp4") 

producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'], 
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = "camera_detections"

print("YOLO tracking producer started...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        results = model.track(source=frame, tracker="bytetrack.yaml", persist=True, conf=0.5, show=False)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                coords = box.xyxy[0].tolist()
                track_id = int(box.id[0]) 

                detections.append({
                    "id": track_id,
                    "class": name,
                    "confidence": conf,
                    "bbox": coords
                })

        if detections:
            message = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": detections
            }
            producer.send(topic_name, value=message)
            print(f"Sent message: {len(detections)} detections.")

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Stopping producer...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    producer.flush() 
    producer.close()
    print("YOLO producer stopped safely.")