import cv2
import os
import time
import datetime
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from detector import Detector
import pandas as pd
from utils.video_reader import VideoReader
from utils import mycv
from utils import visualizer as vis
from config import videos, models


class Tracker:
    def __init__(self, max_dist=0.2, min_confidence=0.9, nms_max_overlap=0.5, max_iou_distance=0.9, max_age=50, n_init=3, nn_budget=100):
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(
            os.path.dirname(current_path) + os.path.sep + ".")

        cfg = get_config()
        cfg.merge_from_file(os.path.join(
            father_path, "deep_sort/configs/deep_sort.yaml"))
        # self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
        #                          max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        #                          nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        #                          max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
        #                          use_cuda=True)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=max_dist, min_confidence=min_confidence,
                                 nms_max_overlap=nms_max_overlap, max_iou_distance=max_iou_distance,
                                 max_age=int(max_age), n_init=int(n_init), nn_budget=int(nn_budget),
                                 use_cuda=True)

    def update(self, img, boxes, classes, scores):
        person_boxes = []
        person_ids = []
        person_scores = []
        ctwhs = []

        for box, class_, score in zip(boxes, classes, scores):

            if class_ == 0:
                x1, y1, x2, y2 = box
                ctwhs.append(
                    (int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1))
                person_scores.append(score)
        if len(ctwhs):
            ctwhs = torch.Tensor(ctwhs)
            person_scores = torch.Tensor(person_scores)
            outputs = self.deepsort.update(ctwhs, person_scores, img)

            for x1, y1, x2, y2, idx in list(outputs):
                person_boxes.append((x1, y1, x2, y2))
                person_ids.append(idx)
        return person_boxes, person_ids
    
    # def __del__(self):
    #     del self.deepsort

def test():
    src = videos.select_local(1)

    # Detectron2 DefaultPredictor Init
    cfg = models.select_model(5)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    # My Detector
    detector = Detector(predictor, predict_score=0.0,
                        confidence_score=0.95, max_overlap=0.8)
    tracker = Tracker()
    reader = VideoReader(src, step_frames=1).start()
    
    data = {
        "time": [],
        "count": [], 
        "total": [],
    }
    tot = -1
    i_frame = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break

        frame = mycv.scale_by_width(frame, 1000)
        
        bak = frame.copy()
        frame[0:250, 0:200] = 0

        # Predict Box
        start_time = time.time()
        boxes, classes, scores = detector.detect(frame)
        person_count = classes.count(0)
        mask_count = len(classes) - person_count

        # Update Tracker
        person_boxes, person_ids = tracker.update(
            frame, boxes, classes, scores)

        # Visualize Box
        frame = bak
        frame = vis.draw_text(frame, " person count: {}\n mask count: {}\n predict time: {:.2f} sec".format(
            person_count, mask_count, time.time() - start_time), (5, 5))
        frame = vis.draw_person_and_mask(frame, boxes, classes, scores)
        frame = vis.draw_person_ids(frame, person_boxes, person_ids)
        
        # Write to local
        if person_ids == []:
            if tot == -1:
                tot = person_count
            else:
                continue
        else:
            tot = max(person_ids)
            
#         data["time"].append(pd.Timestamp(datetime.datetime.now()))
        data["time"].append(i_frame)
        data["count"].append(person_count)
        data["total"].append(tot)
        i_frame += 4
        
        print(data["time"][-1], data["count"][-1], data["total"][-1])
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()
    df = pd.DataFrame(data)
    df.to_csv("log/video4_kp_dsort.csv", mode='w', index=False)


if __name__ == '__main__':
    test()
