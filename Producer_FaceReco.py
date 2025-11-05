# --- Imports ---
from kafka import KafkaProducer
import json
import cv2
import pickle
import os
import face_recognition
import time

# --- Kafka Configuration ---
producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],  # Kafka broker address
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Serialize messages to JSON
)
topic_name = "face_detections"  # The topic to send data to

print("Face recognition producer started...")

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# --- Load Known Faces ---
# Load the pre-computed encodings and names from the pickle file
print("Loading known face encodings...")
file = open('EncodeFile.p', 'rb')
encodeListKnown, Names = pickle.load(file)
file.close()
print("Encodings loaded successfully.")

# --- Cooldown & Performance Settings ---
last_sent_time = {}  # Dictionary to track when we last sent a message for each person
COOLDOWN_SECONDS = 10  # Send a message for the same person only once every 10 seconds

frame_counter = 0  # A counter for frames
PROCESS_EVERY_N_FRAMES = 3  # Process (detect faces) only every 3rd frame for performance

# --- Persistent Drawing ---
# Store the last known locations and names to prevent flickering
last_known_boxes = []
last_known_names = []

# --- Main Loop ---
try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_counter += 1

        # --- Process This Frame (Heavy Lifting) ---
        # Only run the heavy face recognition logic every N frames
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            
            # Reset lists for this processing cycle
            current_boxes = []
            current_names = []

            # 1. Resize frame for faster processing
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            # 2. Convert from BGR (OpenCV default) to RGB (face_recognition default)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # 3. Find all face locations and encodings in the small frame
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            # --- Loop through found faces ---
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                # 4. Compare the found face with our known list
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                # 5. Find the best match (lowest distance)
                matchIndex = faceDis.argmin()

                # --- If a match is found ---
                if matches[matchIndex]:
                    # 6. Get raw data (for Kafka message)
                    name_raw = Names[matchIndex].upper()
                    confidence_score = int((1 - faceDis[matchIndex]) * 100)
                    
                    # 7. Create display text (for video overlay)
                    display_name = f'{name_raw} {confidence_score}%'
                    
                    # 8. Get (and scale back up) face location
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    
                    # 9. Store for persistent drawing
                    current_boxes.append((x1, y1, x2, y2))
                    current_names.append(display_name)

                    # --- Cooldown Logic ---
                    # Check if we should send a Kafka message
                    current_time = time.time()
                    time_since_last_sent = current_time - last_sent_time.get(name_raw, 0)

                    if time_since_last_sent > COOLDOWN_SECONDS:
                        # 10. Prepare Kafka message with clean, raw data
                        message = {
                            "name": name_raw,
                            "confidence": confidence_score,
                            "bbox": [x1, y1, x2, y2]
                        }
                        # 11. Send data to Kafka
                        producer.send(topic_name, value=message)
                        print(f"*** SENT TRIGGER: {message} ***")

                        # 12. Update the last sent time for this person
                        last_sent_time[name_raw] = current_time
            
            # Update the persistent drawings with the results from this frame
            last_known_boxes = current_boxes
            last_known_names = current_names

        # --- Drawing Block (Runs every frame) ---
        # Draw the last known boxes and names onto the *current* frame.
        # This prevents flickering, as drawings persist across skipped frames.
        for (x1, y1, x2, y2), display_name in zip(last_known_boxes, last_known_names):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, display_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

        # --- Display frame and check for quit ---
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Stopping producer...")
            break

# --- Cleanup ---
finally:
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    producer.flush()  # Ensure all pending messages are sent
    producer.close()
    print("Face recognition producer stopped safely.")