import os
import random
import torch, torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
# from data.via_dataset import register_via_dataset, get_via_dicts
from data import via

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("checkpoint", "retinanet_R_50_FPN_3x.pkl")
cfg.OUTPUT_DIR = 'retinanet_R_50_FPN_3x'
cfg.MODEL.RETINANET.NUM_CLASSES = 1


cfg.SOLVER.MAX_ITER = 50000
cfg.SOLVER.BASE_LR = 0.0015


# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration).
cfg.SOLVER.IMS_PER_BATCH = 2

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)



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
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
    print("\nSuccessfully trained!\n")
    
def train_person():
    dataset_name = "person"
    img_root = "./data/datasets/VIA-%s/" % dataset_name
    class_to_category_id = {
        "person": 0,
    }
    
    for d in ["train", "val"]:
        img_dir = os.path.join(img_root, d)
        via_dataset = via.ViaDataset(json_file=os.path.join(img_dir, 'via_region_data.json'))
        
        name = "%s_%s" % (dataset_name, d)
        func = lambda: via_dataset.get_standard_dataset_dicts(img_dir, class_to_category_id)

        DatasetCatalog.register(name, func)
        MetadataCatalog.get(name).set(thing_classes=["person"])
    
    cfg.DATASETS.TRAIN = ("%s_train" % dataset_name,)
    cfg.DATASETS.TEST = ()
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
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
    
