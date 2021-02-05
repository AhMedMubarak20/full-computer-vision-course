import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date
import playsound
import random
import time
path = 'Images/face_data'
unknown_path = 'Images/unknown'
n = random.randint(999, 999999)

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
       # while name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            today = date.today()
            ddString = today.strftime("%d/%m/%Y")
            f.writelines(f'\n{name},{dtString},{ddString}')
            print("d1 =", ddString)
            break
    return



encodeListKnown = findEncodings(images)
print('Encoding Complete')
while True:
    cap = cv2.VideoCapture(0)

    fc = 1
    while fc < 2:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
               # markAttendance(name)
                # time.sleep(5)
                playsound.playsound("speech.mp3")
                print("done")
                fc += 1
                template = cv2.imread("c.png", cv2.IMREAD_GRAYSCALE)
                w, h = template.shape[::-1]

                loop_1 = 1
                while loop_1 < 2:
                    _, frame = cap.read()
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= 0.7)

                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
                        print("something")
                        loop_1 += 1

                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1)
                    if key == 1:
                        break
                print("logo detected successfully")
                playsound.playsound("logo.mp3")
                markAttendance(name)
            else:
                for i in range(5):
                    return_value, image = cap.read()
                    cv2.imwrite(os.path.join(unknown_path, 'unknown' + str(i) + str(n) + '.png'), image)
                playsound.playsound("denied.mp3")

        cv2.imshow('face recognation', img)
        cv2.waitKey(1)

