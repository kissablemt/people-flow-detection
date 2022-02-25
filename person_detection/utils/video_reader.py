import cv2
from threading import Thread


class VideoReader:

    def __init__(self, src=None, step_frames=1):
        if src:
            self.__cap = cv2.VideoCapture(src)
        else:
            self.__cap = None
        
        self.is_local_video = False
        if src != None and src != 0 and isinstance(src, str) and len(src) > 4 and "rtsp" != src[:4]:
            self.is_local_video = True
        self.step_frames = step_frames

    def init(self, src=0, step_frames=1):
        if self.__cap and self.__cap.isOpened():
            self.__cap.release()
        self.__cap = cv2.VideoCapture(src)

        self.is_local_video = False
        if src != None and src != 0 and isinstance(src, str) and len(src) > 4 and "rtsp" != src[:4]:
            self.is_local_video = True
        self.step_frames = step_frames
        return self

    def start(self):
        self.is_reading = True
        self.is_pausing = False
        (self.ret, self.frame) = self.__cap.read()
        if self.is_local_video:
            self.fps = self.__cap.get(cv2.CAP_PROP_FPS)
            self.cur_frame = self.step_frames
        else:
            self.__thd = Thread(target=self.__thd_reader, args=(), daemon=True)
            self.__thd.start()
        return self

    def stop(self):
        self.is_reading = False
        return self

    def pause(self):
        self.is_pausing = True

    def resume(self):
        self.is_pausing = False
        return self

    def release(self):
        self.is_reading = False
        self.is_pausing = False
        if not self.is_local_video:
            self.__thd.join()
        self.__cap.release()

    def read(self):
        if self.is_local_video:
            self.cur_frame += self.step_frames
            self.__cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
            self.ret, self.frame = self.__cap.read()
        
        return self.ret, self.frame

    def restart(self):
        self.stop()
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        return self.start()

    def __thd_reader(self):
        while self.is_reading:
            (self.ret, self.frame) = self.__cap.read()
            if not self.ret:
                self.stop()
            while self.is_pausing and self.is_reading:
                pass
        self.stop()
        print("thr_reader end.")


if __name__ == '__main__':
    # url = '/Users/wzt/Movies/Object Detection/bst.mp4'
    url = '/Users/wzt/Movies/bilibili/辅导孩子作业的崩溃现场，哈哈哈哈哈哈嗝.mp4'
    # url = 'rtsp://192.168.15.184:8554/live'

    reader = VideoReader(url).start()

    while True:
        ret, frame = reader.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.stop()
    reader.release()
    # origin
    # cap = cv2.VideoCapture(url)

    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     frame = putIterationsPerSec(frame, cps.countsPerSec())
    #     cps.increment()
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
