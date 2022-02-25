import cv2
from queue import Queue
from person_detection.utils.video_reader import VideoReader
from person_detection.utils.counts_per_sec import CountsPerSec, put_iterations_per_sec
from threading import Lock

READER_NUM = 2
READER_POOL = [None for _ in range(READER_NUM)]  # list[VideoReader()]
READER_POOL_LOCK = [Lock() for _ in range(READER_NUM)]  # list[Lock()]

READER_IS_READING = [False for _ in range(READER_NUM)]  # list[bool()]
READER_IS_READING_LOCK = [Lock() for _ in range(READER_NUM)]  # list[Lock()]

READER_QUEUE = Queue(maxsize=READER_NUM)  # Queue(int())
for i in range(READER_NUM):
    READER_QUEUE.put(i)


def video_feed_gen(available_id, video_source):
    global READER_POOL, READER_QUEUE, READER_POOL_LOCK, READER_IS_READING_LOCK

    src = video_source
    # src = "/Users/wzt/Movies/Object Detection/bst.mp4"
    # src = 'rtsp://192.168.15.184:8554/live'

    available_id = int(available_id)
    print("&" * 1000)
    rd = READER_POOL[available_id] = VideoReader(src, 10).start()
    cps = CountsPerSec().start()
    with READER_IS_READING_LOCK[available_id]:
        READER_IS_READING[available_id] = True

    while True:
        ret, frame = rd.read()
        if not ret:
            break
        frame = put_iterations_per_sec(frame, cps.counts_per_sec())
        ret, jpeg = cv2.imencode('.jpg', frame)

        with READER_IS_READING_LOCK[available_id]:
            if not READER_IS_READING[available_id]:
                break
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        cps.increment()
    rd.release()
    
    with READER_IS_READING_LOCK[available_id]:
        if READER_IS_READING[available_id]:
            READER_QUEUE.put(available_id, block=True)
            READER_IS_READING[available_id] = False
            READER_POOL_LOCK[available_id].release()
            print("Pool[{}] is released.".format(available_id))
