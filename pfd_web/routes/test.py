from flask import Blueprint, render_template, request, Response, redirect, url_for
import cv2
from queue import Queue
from person_detection.utils.video_reader import VideoReader
from person_detection.utils.counts_per_sec import CountsPerSec, put_iterations_per_sec
from threading import Lock

test_bp = Blueprint('test', __name__)

"""****************************************** index ******************************************"""


@test_bp.route('/')
def index():
    return render_template('test/index.html')


"""****************************************** via ******************************************"""


@test_bp.route('/via')
def via():
    return render_template('test/via.html')


"""****************************************** roi ******************************************"""


@test_bp.route('/roi')
def select_roi():
    return render_template('test/selectROI.html')


"""****************************************** video ******************************************"""
READER_NUM = 2
READER_POOL = [None for _ in range(READER_NUM)]  # list[VideoReader()]
READER_POOL_LOCK = [Lock() for _ in range(READER_NUM)]  # list[Lock()]

READER_IS_READING = [False for _ in range(READER_NUM)]  # list[bool()]
READER_IS_READING_LOCK = [Lock() for _ in range(READER_NUM)]  # list[Lock()]

READER_QUEUE = Queue(maxsize=READER_NUM)  # Queue(int())
for i in range(READER_NUM):
    READER_QUEUE.put(i)


@test_bp.route('/video')
def origin_video():
    global READER_POOL, READER_QUEUE, READER_POOL_LOCK
    available_id = READER_QUEUE.get(block=True)
    READER_QUEUE.task_done()

    """***** Pool Locked!!! *****"""
    READER_POOL_LOCK[available_id].acquire()
    print("Pool[{}] is locked.".format(available_id))

    print("video_feed(available_id={})".format(available_id))
    return render_template('test/originVideo.html', video_id=available_id)


def video_feed_gen(available_id):
    global READER_POOL, READER_QUEUE, READER_POOL_LOCK, READER_IS_READING_LOCK

    src = "/Users/wzt/Movies/bilibili/通厕神器.mp4"
    # src = "/Users/wzt/Movies/Object Detection/bst.mp4"
    # src = 'rtsp://192.168.15.184:8554/live'

    available_id = int(available_id)

    rd = READER_POOL[available_id] = VideoReader(src).start()
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
    rd.stop()
    rd.release()

    with READER_IS_READING_LOCK[available_id]:
        if READER_IS_READING[available_id]:
            READER_QUEUE.put(available_id, block=True)
            READER_IS_READING[available_id] = False
            READER_POOL_LOCK[available_id].release()
            print("Pool[{}] is released.".format(available_id))


@test_bp.route('/video_feed/<video_id>')
def video_feed(video_id):
    return Response(video_feed_gen(video_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@test_bp.route('/stop', methods=['POST'])
def stop():
    video_id = int(request.form.get('video_id'))
    with READER_IS_READING_LOCK[video_id]:
        if READER_IS_READING[video_id]:
            READER_QUEUE.put(video_id, block=True)
            READER_IS_READING[video_id] = False
            READER_POOL_LOCK[video_id].release()
            print("Pool[{}] is released.".format(video_id))
    print("video[{}] is killed.".format(video_id))
    return '0'


@test_bp.route('/pause', methods=['POST'])
def pause():
    return '0'
