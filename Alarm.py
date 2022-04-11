from pygame import *

class Alarm:
    def Start(self, ser):
        ser.write('9'.encode('utf-8'))
        # mixer.init()
        # mixer.music.load('alarm.mp3')
        # mixer.music.play()
    def Stop(self):
        # mixer.music.stop()
        return
    def Status(self):
        try:
            return mixer.music.get_busy()
        except Exception as e:
            return 0
