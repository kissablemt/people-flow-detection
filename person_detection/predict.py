import os
import random
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from utils import visualizer as vis
from utils import mycv
from detector import Detector


def predict_person_and_mask():
    # Detectron2 DefaultPredictor Init
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(
        "FRCNN-R50-FPN_person_and_mask", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    predictor = DefaultPredictor(cfg)

    # My Detector
    detector = Detector(predictor, predict_score=0.75,
                        confidence_score=0.95, max_overlap=0.8)

    im = cv2.imread("./data/samples/person_and_mask/maksssksksss3.png")
#     im = cv2.imread("/home/wzt/PFD/COCO/val2017/000000000139.jpg")

#     im = mycv.scale(im, 2)
    boxes, classes, scores = detector.detect(im)
    person_count = classes.count(0)
    mask_count = len(classes) - person_count
    
    im = vis.draw_text(im, " person count: {}\n mask count: {}\n".format(
            person_count, mask_count), (5, 5))

    out = vis.draw_person_and_mask(im, boxes, classes, scores)
    cv2.imshow("", out)
    cv2.waitKey()


def main():
    return 0


if __name__ == '__main__':
    predict_person_and_mask()
