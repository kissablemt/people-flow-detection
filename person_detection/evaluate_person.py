from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import os
from data import via
from detectron2.engine import DefaultTrainer

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
        "MRCNN-512-50000",
        "mask_rcnn_R_50_FPN_3x"
    ),
}
model_select = "mrcnn"
cfg.merge_from_file(models[model_select][0])
cfg.MODEL.WEIGHTS = models[model_select][1]
cfg.OUTPUT_DIR = models[model_select][2]
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = os.path.join("FRCNN-R50-FPN_person_and_mask", "model_final.pth")
cfg.DATALOADER.NUM_WORKERS = 4

dataset_name = "person_coco"
for d in ["train", "val"]:
    coco_img_dir = "/home/wzt/PFD/COCO/{}2017".format(d)
    coco_json_file = "/home/wzt/PFD/COCO/{}2017_person.json".format(d)
    register_coco_instances("{}_{}".format(dataset_name, d), {}, coco_json_file, coco_img_dir)
    MetadataCatalog.get("{}_{}".format(dataset_name, d)).set(thing_classes=["person"])

# dataset_name = "person_and_mask"
# img_root = "./data/datasets/VIA-%s/" % dataset_name
# class_to_category_id = {
#     "person": 0,
#     "face_with_mask": 1,
# }

# for d in ["train", "val"]:
#     img_dir = os.path.join(img_root, d)
#     via_dataset = via.ViaDataset(json_file=os.path.join(img_dir, 'via_region_data.json'))

#     name = "%s_%s" % (dataset_name, d)
#     func = lambda: via_dataset.get_standard_dataset_dicts(img_dir, class_to_category_id)

#     DatasetCatalog.register(name, func)
#     MetadataCatalog.get(name).set(thing_classes=["person", "mask"])

cfg.DATASETS.TRAIN = ("%s_train" % dataset_name,)
cfg.DATASETS.TEST = ()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
trainer = DefaultTrainer(cfg) 

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("%s_val" % dataset_name, cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "%s_val" % dataset_name)
inference_on_dataset(predictor.model, val_loader, evaluator)