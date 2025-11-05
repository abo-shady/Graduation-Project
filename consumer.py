from kafka import KafkaConsumer
from pymongo import MongoClient
import json

print("Connecting to Kafka...")

consumer = KafkaConsumer(
    'camera_detections', 'face_detections',
    bootstrap_servers=['localhost:29092'], 
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

print("Connecting to MongoDB...")

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['car_system']      
collection_detection = db['detections']   
collection_Face_reco = db['face_recognition']        

print("📡 Listening for messages...")

for message in consumer:
    data = message.value
    print(f"Received: {data}")
    
    if message.topic == 'camera_detections':
            collection_detection.insert_one(data)
            print("YOLO Data inserted into MongoDB.")
    elif message.topic == 'face_detections':
            collection_Face_reco.insert_one(data)
            print("Face Recognition Data inserted into MongoDB.")