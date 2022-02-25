import os
import sys
from threading import Lock
from queue import Queue
from person_detection.utils.video_reader import VideoReader


class ReaderTask:
    def __init__(self, max_reader=2):
        self.__reader_num = max_reader
        self.__pool = [VideoReader() for _ in range(max_reader)]  # list[VideoReader()]
        self.__id_lock = [Lock() for _ in range(max_reader)]  # list[Lock()]
        self.__id_queue = Queue(maxsize=max_reader)
        for i in range(max_reader):
            self.__id_queue.put(i)

    def get_free_id(self, block=True):
        return self.__id_queue.get(block=block)

    def put_free_id(self, id_, block=True):
        self.__id_queue.put(id_, block=block)

    def get_reader(self, id_: int):
        with self.__id_lock[id_]:
            return self.__pool[id_]

    def put_reader(self, id_: int, video_reader: VideoReader):
        with self.__id_lock[id_]:
            self.__pool[id_] = video_reader

    def stop_reader(self, id_: int):
        with self.__id_lock[id_]:
            self.__pool[id_].release()
            self.put_free_id(id_)
            
