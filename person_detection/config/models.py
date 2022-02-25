from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
father_path = os.path.abspath(current_path + os.path.sep + "..")

cfgs = [
    "my_model()", # 0
    "faster_rcnn_R_50_FPN_3x()", # 1
    "faster_rcnn_R_50_C4_3x()", # 2
    "faster_rcnn_R_50_DC5_3x()", # 3
    "mask_rcnn_R_50_FPN_3x()", # 4
    "keypoint_rcnn_R_50_FPN_3x()", # 5
    "retinanet_R_50_FPN_3x()", # 6
]

def select_model(idx: int):
    return eval(cfgs[idx])

"""
Faster R-CNN ResNet-50
    FPN_3x coco(2 class)
"""
def my_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "FRCNN-R50-FPN_person_and_mask", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    return cfg


"""
Faster R-CNN ResNet-50
    FPN_3x coco(80 classes)
    FPN_3x coco(1 class)
    
    C4_3x coco(80 classes)
    C4_3x coco(1 class)
    
    DC5_3x coco(80 classes)
    DC5_3x coco(1 class)
"""
def faster_rcnn_R_50_FPN_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "faster_rcnn_R_50_FPN_3x.pkl")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    return cfg

def faster_rcnn_R_50_C4_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "faster_rcnn_R_50_C4_3x.pkl")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    return cfg

def faster_rcnn_R_50_DC5_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "faster_rcnn_R_50_DC5_3x.pkl")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    return cfg


"""
Mask R-CNN ResNet-50
    FPN_3x coco(80 classes)
    FPN_3x coco(1 class)
"""
def mask_rcnn_R_50_FPN_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "mask_rcnn_R_50_FPN_3x.pkl")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    return cfg


"""
Keypoint R-CNN ResNet-50
    FPN_3x coco(1 class)
"""
def keypoint_rcnn_R_50_FPN_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "keypoint_rcnn_R_50_FPN_3x.pkl")
    return cfg


"""
Retina-Net ResNet-50
    FPN_3x coco(80 classes)
    FPN_3x coco(1 class)
"""
def retinanet_R_50_FPN_3x():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(father_path, "checkpoint", "retinanet_R_50_FPN_3x.pkl")
    cfg.MODEL.RETINANET.NUM_CLASSES = 80
    return cfg