import cv2
import sys
import Status as st
import numpy as np
import Alarm
import serial
import time
from detect import Camera, writeText, ContrastHist, countblink


class EyeBlink(object):
    def __init__(self, camera):
        self.camera = camera
        # self.ser = serial.Serial('COM5', baudrate=9600, parity='N', stopbits=1)
        # if self.ser.is_open:
        #     self.ser.close()
        # self.ser.open()

    def detect(self):
        alarm = Alarm.Alarm()
        Time = 0
        cv2.namedWindow('frame')
        list_detect = [1] * 40
        C1 = [0] * (len(list_detect) - 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cap = Camera(self.camera)
        while (cap.isOpened() and cv2.getWindowProperty('frame', 0) >= 0):
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            frame = frame[h // 4: h * 3 // 4, w // 4: w * 3 // 4]

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = ContrastHist(gray)
                state = st.main(gray)

                if state == -1:
                    list_detect = [*list_detect[1:], list_detect[-1]]
                    writeText(frame, '', (0, 0, 0))
                    Time = Time + 1
                    if Time == 10:
                        if not alarm.Status():
                            pass
                            # alarm.Start(self.ser)  # gui 9 sang cong com
                elif state:
                    list_detect = [*list_detect[1:], 1]
                    writeText(frame, 'Opened', (0, 255, 0))
                    Time = 0
                    if alarm.Status():
                        alarm.Stop()
                else:
                    list_detect = [*list_detect[1:], 0]
                    writeText(frame, 'Closed', (0, 0, 255))
                    Time = Time + 1
                    if Time == 10:
                        if not alarm.Status():
                            pass
                            # alarm.Start(self.ser)

                C1, D = countblink(list_detect, C1)
                cv2.putText(frame, '{}'.format(np.max(C1)), (w // 2 - 40, 20), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if sum(D) > 0:
                    print('number of blink', np.max(C1))
                    # self.ser.write(str(np.max(C1)).encode('utf-8'))
                    list_detect = [list_detect[-1]] * len(list_detect)
                    C1 = [0] * (len(list_detect) - 1)

                cv2.imshow('frame', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run = EyeBlink(camera=0)
    run.detect()
