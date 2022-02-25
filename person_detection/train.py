import os
import random
import torch, torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
# from data.via_dataset import register_via_dataset, get_via_dicts
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from data import via

cfg = get_cfg()
models = {
    "fpn": (
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"), 
        os.path.join("checkpoint", "faster_rcnn_R_50_FPN_3x.pkl"),
        "faster_rcnn_R_50_FPN_3x"
    ),
    "c4": (
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"), 
        os.path.join("checkpoint", "faster_rcnn_R_50_C4_3x.pkl"),
        "faster_rcnn_R_50_C4_3x"
    ),
    "dc5": (
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"), 
        os.path.join("checkpoint", "faster_rcnn_R_50_DC5_3x.pkl"),
        "faster_rcnn_R_50_DC5_3x"
    ),
    "mrcnn": (
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"), 
        os.path.join("checkpoint", "mask_rcnn_R_50_FPN_3x.pkl"),
        "mask_rcnn_R_50_FPN_3x"
    ),
    "kp": (
        model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"), 
        os.path.join("checkpoint", "keypoint_rcnn_R_50_FPN_3x.pkl"),
        "keypoint_rcnn_R_50_FPN_3x"
    ),
}
model_select = "c4"
cfg.merge_from_file(models[model_select][0])
cfg.MODEL.WEIGHTS = models[model_select][1]
cfg.OUTPUT_DIR = models[model_select][2]
# cfg.OUTPUT_DIR = "FRCNN-R50-FPN_person_and_mask"
cfg.OUTPUT_DIR = "c4-512-50000"

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = os.path.join("checkpoint", "retinanet_R_101_FPN_3x.pkl")
# cfg.OUTPUT_DIR = 'retinanet_R_101_FPN_3x_output'

# cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")

cfg.SOLVER.MAX_ITER = 50000
cfg.SOLVER.BASE_LR = 0.0015


# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration).
cfg.SOLVER.IMS_PER_BATCH = 2

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)



def train_person_and_mask():
    dataset_name = "person_and_mask"
    img_root = "./data/datasets/VIA-%s/" % dataset_name
    class_to_category_id = {
        "person": 0,
        "face_with_mask": 1,
    }
    
    for d in ["train", "val"]:
        img_dir = os.path.join(img_root, d)
        via_dataset = via.ViaDataset(json_file=os.path.join(img_dir, 'via_region_data.json'))
        
        name = "%s_%s" % (dataset_name, d)
        func = lambda: via_dataset.get_standard_dataset_dicts(img_dir, class_to_category_id)

        DatasetCatalog.register(name, func)
        MetadataCatalog.get(name).set(thing_classes=["person", "mask"])
    
    cfg.DATASETS.TRAIN = ("%s_train" % dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
    print("\nSuccessfully trained!\n")
    
def train_person():
#     dataset_name = "person"
#     img_root = "./data/datasets/VIA-%s/" % dataset_name
#     class_to_category_id = {
#         "person": 0,
#     }
    
#     for d in ["train", "val"]:
#         img_dir = os.path.join(img_root, d)
#         via_dataset = via.ViaDataset(json_file=os.path.join(img_dir, 'via_region_data.json'))
        
#         name = "%s_%s" % (dataset_name, d)
#         func = lambda: via_dataset.get_standard_dataset_dicts(img_dir, class_to_category_id)

#         DatasetCatalog.register(name, func)
#         MetadataCatalog.get(name).set(thing_classes=["person"])
    
    dataset_name = "person_coco"
    for d in ["train", "val"]:
        coco_img_dir = "/home/wzt/PFD/COCO/{}2017".format(d)
        coco_json_file = "/home/wzt/PFD/COCO/{}2017_person.json".format(d)
        register_coco_instances("{}_{}".format(dataset_name, d), {}, coco_json_file, coco_img_dir)
        MetadataCatalog.get("{}_{}".format(dataset_name, d)).set(thing_classes=["person"])
        

    cfg.DATASETS.TRAIN = ("%s_train" % dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
    print("\nSuccessfully trained!\n")
    
def main():
    
    return 0


if __name__ == '__main__':
#     main()
#     train_n_classes()
#     train()
#     train_person_and_mask()
    train_person()
    pass
    
