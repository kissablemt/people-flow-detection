import time
import cv2
import os
from data import via
from shapely.geometry import Polygon
from detectron2.engine import DefaultPredictor
from detector import Detector
import pandas as pd
from utils.video_reader import VideoReader
from utils import mycv
from utils import visualizer as vis
from config import videos, models

class Optimizer:
    def __init__(self, json_file, default_max_overlap=0.5):
        if os.path.exists(json_file):
            via_dataset = via.ViaDataset(json_file=json_file)
            self.polygons, self.max_overlaps, self.types = via_dataset.get_opt_polygons(default_max_overlap)
            self.active = True
        else:
            self.active = False
    
    def optmize(self, boxes, classes, scores, scale=1.0):
        '''
        boxes: List[XYXY]
        '''
        if not self.active:
            return boxes, classes, scores
        opt_boxes = []
        opt_classes = []
        opt_scores = []
        for box, class_, score in zip(boxes, classes, scores):
            if class_ == 0:
                box_scale = list(map(lambda x: x / scale, box))
                x1, y1, x2, y2 = box_scale
                box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                keep = True
                for poly, max_overlap, type_ in zip(self.polygons, self.max_overlaps, self.types):
                    if type_ == 0:
                        iou = box_poly.intersection(poly).area / poly.area
                    else:
                        iou = box_poly.intersection(poly).area / box_poly.area
                    if iou >= max_overlap:
                        keep = False
                        break
                if not keep:
                    continue  
            
            opt_boxes.append(box)
            opt_classes.append(class_)
            opt_scores.append(score)
        return opt_boxes, opt_classes, opt_scores
    

def test():
    
    src = videos.select_local(4)

    # Detectron2 DefaultPredictor Init
    cfg = models.select_model(0)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)

    # My Detector
    detector = Detector(predictor, predict_score=0.0,
                        confidence_score=0.95, max_overlap=0.8)
    reader = VideoReader(src, step_frames=4).start()
    
    optimizer = Optimizer(json_file="/home/wzt/PFD/data/opts/3.json")

    while True:
        ret, frame = reader.read()
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
        
        person_count = classes.count(0)
        mask_count = len(classes) - person_count

        # Visualize Box
        frame = vis.draw_person_and_mask(frame, boxes, classes, scores)
        frame = vis.draw_text(frame, " person count: {}\n mask count: {}\n predict time: {:.2f} sec".format(
            person_count, mask_count, time.time() - start_time), (5, 5))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()


if __name__ == '__main__':
    test()
            