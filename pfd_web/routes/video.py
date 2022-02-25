import os
from flask import Blueprint, render_template, request, Response
import time
import urllib
from queue import Queue
import cv2
from cv2 import VideoCapture
from threading import Lock, Thread
from multiprocessing import Process, Value
from pfd_web.utils import task
import json
import pandas as pd
import datetime

from detectron2.engine import DefaultPredictor
from person_detection.utils.video_reader import VideoReader
from person_detection.utils.counts_per_sec import CountsPerSec, put_iterations_per_sec
from person_detection.detector import Detector
from person_detection.optimizer import Optimizer
from person_detection.tracker import Tracker
from person_detection.utils import mycv
from person_detection.utils import visualizer as vis
from person_detection.config import videos, models

from . import data as D


video_bp = Blueprint('video', __name__)
reader_task = task.ReaderTask()

file_path = os.path.abspath(__file__)  # /home/wzt/PFD/pfd_web/routes/video.py
current_path = os.path.abspath(os.path.dirname(
    file_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web/routes
father_path = os.path.abspath(os.path.dirname(
    current_path) + os.path.sep + ".")  # /home/wzt/PFD/pfd_web
root_path = os.path.abspath(father_path + os.path.sep + "..")  # /home/wzt/PFD

data_path = os.path.join(root_path, 'data')  # /home/wzt/PFD/data
image_path = os.path.join(data_path, 'images')  # /home/wzt/PFD/data/images
opt_path = os.path.join(data_path, 'opts')  # /home/wzt/PFD/data/opts


class VideoChecker:
    def __init__(self):
        self.thds = {}
        self.last_time = {}
        self.data = {}

    def add_target(self, video_id):
        print("*******\nadd_target %d \n*******\n" % video_id)
        self.last_time[video_id] = time.time()
        self.thds[video_id] = Thread(
            target=self.__thd_checker, args=(video_id,), daemon=True)
        self.thds[video_id].start()

    def keep_alive(self, video_id):
        if not self.last_time.get(video_id):
            self.add_target(video_id)
        else:
            self.last_time[video_id] = time.time()
    
    def upload_data(self, video_id, scene_id, data):
        self.data[video_id] = [scene_id, data]

    def del_target(self, video_id):
        global reader_task
        if self.last_time.get(video_id):
            del self.last_time[video_id]
        if self.thds.get(video_id):
            del self.thds[video_id]
        reader_task.stop_reader(video_id)
        D.add(self.data[video_id][0], pd.DataFrame(self.data[video_id][1]))
        print("*******\ndel_target %d \n*******\n" % video_id)

    def __thd_checker(self, video_id):
        while True:
#             print("*******\nchecking %d \n*******\n" % video_id)

            if not self.last_time.get(video_id):
                break
            now = time.time()
            diff = now - self.last_time[video_id]
            if diff > 10:
                print("*******\nkill VideoChecker %s\n*******\n" % video_id)
                break
            time.sleep(2)
        self.del_target(video_id)


reader_checker = VideoChecker()


@video_bp.route('/show', methods=['GET'])
def show():
    global reader_task
    scene = request.args.get("scene")
    scene_id = request.args.get("scene_id")
    free_id = reader_task.get_free_id()
    print("video_feed(free_id={})".format(free_id))
    return render_template('video/show.html', video_id=free_id, scene=scene, scene_id=scene_id)


def video_feed_gen(free_id, scene, scene_id):
    global reader_task
    global reader_checker
    
    scene = json.loads(scene.replace("\'", "\""))
    src = scene["source"]
    
    free_id = int(free_id)

    rd = reader_task.get_reader(free_id)
    rd.init(src=src, step_frames=4)
    rd.start()
    
    cfg = models.select_model(0)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)

    detector = Detector(predictor, predict_score=scene["predict_score"],
                        confidence_score=scene["confidence_score"], max_overlap=scene["max_overlap"])
    optimizer = Optimizer(json_file=os.path.join(opt_path, "%s.json" % scene_id))
    tracker = Tracker(max_dist=scene["max_dist"], min_confidence=scene["min_confidence"],
                      nms_max_overlap=scene["nms_max_overlap"], max_iou_distance=scene["max_iou_distance"],
                      max_age=scene["max_age"], n_init=scene["n_init"], nn_budget=scene["nn_budget"])
    
    data = {
        "time": [],
        "people_count": [], 
        "people_total": [],
        "mask_count": [], 
    }
    p_tot = -1

    
    while True:
        ret, frame = rd.read()
        if not ret:
            break
        
        width = frame.shape[1]
        frame = mycv.scale_by_width(frame, 800)
        scale = frame.shape[1] / width

        # Predict Box
        start_time = time.time()
        boxes, classes, scores = detector.detect(frame)

        # Optimize Boxes
        boxes, classes, scores = optimizer.optmize(boxes, classes, scores, scale=scale)

        # Update Tracker
        person_boxes, person_ids = tracker.update(
            frame, boxes, classes, scores)

        person_count = classes.count(0)
        mask_count = len(classes) - person_count

        # Visualize Box
        frame = vis.draw_text(frame, " person count: {}\n mask count: {}\n predict time: {:.2f} sec".format(
            person_count, mask_count, time.time() - start_time), (5, 5))
        frame = vis.draw_person_and_mask(frame, boxes, classes, scores)
        frame = vis.draw_person_ids(frame, person_boxes, person_ids)   

        # Write to local
        data["time"].append(pd.Timestamp(datetime.datetime.now()))

        if person_ids == []:
            if p_tot == -1:
                p_tot = person_count
        else:
            m_id = max(person_ids)
            p_tot = max(p_tot, m_id, person_count)
          
        data["people_count"].append(person_count)
        data["people_total"].append(p_tot)

        data["mask_count"].append(mask_count)
        
        print(data["time"][-1], data["people_count"][-1], data["people_total"][-1], data["mask_count"][-1])
        reader_checker.upload_data(free_id, scene_id, data) 
        
#         frame = mycv.scale_by_width(frame, 500)
        ret, jpeg = cv2.imencode('.jpg', frame)

        if not rd.is_reading:
            break
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'

    rd.release()
    # reader_task.put_free_id(free_id, data)


@video_bp.route('/video_feed', methods=['GET'])
def video_feed():
    video_id = request.args.get("video_id")
    scene = request.args.get("scene")
    scene_id = request.args.get("scene_id")
    return Response(video_feed_gen(video_id, scene, scene_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@video_bp.route('/check', methods=['POST'])
def check():
    global reader_checker
    video_id = int(request.form.get("video_id"))
    reader_checker.keep_alive(video_id)
    # print("*******\nchecking %d \n*******\n" %  video_id)
    return "check %s" % video_id
