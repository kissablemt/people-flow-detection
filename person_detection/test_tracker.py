import os
from typing import List, Tuple
import time
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.catalog import Metadata
from utils.video_reader import VideoReader
from utils import mycv
from utils import visualizer as vis
from config import videos, models
import pandas as pd
import datetime

from detector import Detector
from optimizer import Optimizer
from tracker import Tracker


def main():
    src = videos.select_local(4)
    
    # Video Reader
    rd = VideoReader(src, step_frames=1).start()
    
    # Detector
    cfg = models.select_model(5)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    detector = Detector(predictor, predict_score=0.8, confidence_score=1.0, max_overlap=0.8)
    
    # Optimizer
    optimizer = Optimizer("/home/wzt/PFD/data/opts/3.json")
    
    # Tracker
    tracker = Tracker()
    
    data = {
        "time": [],
        "people_count": [], 
        "people_total": [],
        "mask_count": [], 
    }
    p_tot = -1
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/opt2.avi', fourcc, 24.0, (800, 450), True)
    
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
        out.write(frame)
#         cv2.imshow("", frame)
        
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

        if not rd.is_reading:
            break
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

    rd.release()
    log_filepath = os.path.join("output", "opt2.csv")
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(log_filepath, mode='w', index=False)
    print("Done.")
            
if __name__ == '__main__':
    main()
