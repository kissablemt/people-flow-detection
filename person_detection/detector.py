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


class Detector:

    def __init__(self, predictor: DefaultPredictor, predict_score=0.75, confidence_score=1.0, max_overlap=0.8):
        """
        Args:
            predictor: DefaultPredictor.
            predict_score: Model SCORE_THRESH_TEST.
            confidence_score: If the score is higher than the confidence score, select directly.
            max_overlap: Maximum similarity tolerance.
        """
        self._predict_score = predict_score
        self._confidence_score = confidence_score
        self._max_overlap = max_overlap
        self.predictor = predictor

    def detect(self, image: np.ndarray) -> Tuple[List, List, List]:
        start_time = time.time()
        outputs = self.predictor(image)

        # Boxes are already sorted by score (scores[i] > scores[i+1])
        person_idx = np.argwhere(
            outputs["instances"].pred_classes.cpu().numpy() == 0).flatten()
        person_boxes = [box.cpu().tolist() for box in list(
            outputs["instances"].pred_boxes[person_idx])]
        person_scores = outputs["instances"].scores[person_idx].tolist()

        mask_idx = np.argwhere(
            outputs["instances"].pred_classes.cpu().numpy() == 1).flatten()
        mask_boxes = [box.cpu().tolist() for box in list(
            outputs["instances"].pred_boxes[mask_idx])]
        mask_scores = outputs["instances"].scores[mask_idx].tolist()

        # Remove repeated person boxes
        boxes, scores = self._nms(boxes=person_boxes, scores=person_scores)
        classes = np.concatenate((np.zeros(len(boxes), dtype=int), np.ones(
            len(mask_boxes), dtype=int)), axis=0).tolist()
        boxes.extend(mask_boxes)
        scores.extend(mask_scores)

        end_time = time.time()
#         print("time: ", time.time() - start_time)

        return boxes, classes, scores

    def _nms(self, boxes: List[tuple], scores: List[float]) -> Tuple[List, List]:
        """
        Remove repeated person boxes.

        Args:
            boxes: Unoptimized person boxes.
            scores: Scores of unoptimized person boxes.

        Returns:
            list[list]:
                The maximum overlap ratio between the overlapping part and the two rectangles.

        """
        def overlap_ratio(rect1, rect2):
            """
            Calculate the overlap of two rectangles.

            Args:
                rect1: optimized person box.
                rect2: Unoptimized person box.

            Returns:
                float:
                    The maximum overlap ratio between the overlapping part and the two rectangles.
            """
            (w1, h1) = (rect1[2] - rect1[0], rect1[3] - rect1[1])
            (w2, h2) = (rect2[2] - rect2[0], rect2[3] - rect2[1])
            (startx, endx) = (min(rect1[0], rect2[0]), max(rect1[2], rect2[2]))
            (starty, endy) = (min(rect1[1], rect2[1]), max(rect1[3], rect2[3]))
            width = w1 + w2 - (endx - startx)
            height = h1 + h2 - (endy - starty)

            (ratio, area) = (0.0, 0.0)
            if width > 0 and height > 0:
                (area, area1, area2) = (width * height, w1 * h1, w2 * h2)
                ratio = max(area / area1, area / area2)

            return ratio

        threshold = self._max_overlap
        confidence_score = self._confidence_score

        opt_boxes = []
        opt_scores = []
        for (box, score) in zip(boxes, scores):
            if score < self._predict_score:
                continue
            selected = True
            # If it is lower than the confidence score, then calculate the overlap.
            if score < confidence_score:
                for cmp in opt_boxes:
                    # If the similarity is greater than $threshold, it is considered that two boxes are detecting the same person.
                    if overlap_ratio(cmp, box) > threshold:
                        selected = False
                        break
            if selected:
                opt_boxes.append(box)
                opt_scores.append(score)
        return opt_boxes, opt_scores


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

    while True:
        ret, frame = reader.read()
        assert ret

        frame = mycv.scale_by_width(frame, 800)

        # Predict Box
        start_time = time.time()
        boxes, classes, scores = detector.detect(frame)
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
