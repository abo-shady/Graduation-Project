import face_recognition
import cv2
import pickle
import os

# importing images 
FolderPath = 'Images'
PathList = os.listdir(FolderPath)
Names = []
ImgList = []
for path in PathList:
    ImgList.append(cv2.imread(os.path.join(FolderPath, path)))
    # removing file extension from name
    Names.append(os.path.splitext(path)[0])
print(Names)

# encoding images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(ImgList)
print("Encoding Complete")

# saving encodings and names
with open('EncodeFile.p', 'wb') as f:
    pickle.dump((encodeListKnown, Names), f)

print("File Saved")