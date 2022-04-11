import cv2
import sys
import Status as st
import numpy as np
import Alarm

def ContrastHist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def Camera(CamNum):
    return cv2.VideoCapture(int(CamNum))

def writeText(frame, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (20, 20), font, 1, color, 2, cv2.LINE_AA)

def countblink(list_detect, C1):
    A = np.array(list_detect[1:])
    B = np.array(list_detect[:-1])
    C = B - A
    C[C<0] = 0 
    C1 = [*C1[1:],sum(C)]

    A = np.array(C1[1:])
    B = np.array(C1[:-1])
    D = B - A
    D[D < 0] =0
    return C1 , D

def detect():
        alarm = Alarm.Alarm()
        Time = 0
        cv2.namedWindow('frame')
        list_detect = [1] * 40
        C1 = [0] * (len(list_detect) -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while (cap.isOpened() and cv2.getWindowProperty('frame', 0) >= 0):
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            frame = frame[h //4: h * 3//4, w // 4: w * 3 //4]

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = ContrastHist(gray)
                state = st.main(gray)

                if state == -1:
                    list_detect = [*list_detect[1:],list_detect[-1]]
                    writeText(frame, '', (0, 0, 0))
                    Time = Time + 1
                    if Time == 10:
                        if not alarm.Status():
                            alarm.Start()
                elif state:
                    list_detect = [*list_detect[1:],1]
                    writeText(frame, 'Opened', (0, 255, 0))
                    Time = 0
                    if alarm.Status():
                        alarm.Stop()
                else:
                    list_detect = [*list_detect[1:],0]
                    writeText(frame, 'Closed', (0, 0, 255))
                    Time = Time + 1
                    if Time == 10:
                        if not alarm.Status():
                            alarm.Start()
                A = np.array(list_detect[1:])
                B = np.array(list_detect[:-1])
                C = B - A
                C[C<0] = 0 
                C1 = [*C1[1:],sum(C)]

                A = np.array(C1[1:])
                B = np.array(C1[:-1])
                D = B - A
                D[D < 0] =0

                cv2.putText(frame, '{}'.format(np.max(C1)), (w // 2 - 40,20), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if sum(D) > 0:
                    print('number of blink', np.max(C1))
                    list_detect = [list_detect[-1]] * len(list_detect)
                    C1 = [0] * (len(list_detect) -1)


                cv2.imshow('frame', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()