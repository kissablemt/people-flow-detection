import json
import os
import random
import numpy as np
from shutil import copy
import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
import shapely
from shapely.geometry import Polygon

class ViaAnnotation:
    """
    e.g.
    filename: '16335852991_f55de7958d_k.jpg'
    size: 1767935
    regions: [
        {
            'shape_attributes': {
                name: 'polygon',
                all_points_x: [588, 617, 649, 673, 692, 708, 722, 730, 737, 718, 706, 699, 697, 676, 650, 613, 580, 552, 534, 520, 513, 513, 521, 526, 541, 560, 588],
                all_points_y: [173, 168, 172, 182, 197, 216, 237, 260, 283, 312, 341, 367, 390, 369, 349, 337, 337, 347, 361, 332, 296, 266, 243, 225, 205, 187, 173]
            },
            'region_attributes': {'class': 'balloon'}
        },
        {
            'shape_attributes': {
                name: 'rect',
                x: 314,
                y: 49,
                width: 192,
                height: 250
            },
            'region_attributes': {'class': 'mask'}
        }
    ]

    """
    filename: str
    size: int
    regions: list

    def __init__(self, filename: str, size: int, regions: list):
        self.filename = filename
        self.size = size
        if isinstance(regions, list):
            self.regions = regions
        elif isinstance(regions, dict):
            self.regions = list(regions.values())
        else:
            raise TypeError

    def dict(self):
        if not self.filename or not self.size or not self.regions:
            raise ValueError
        anno_dict = {
            'filename': self.filename,
            'size': self.size,
            'regions': self.regions
        }
        #         print(type(json.dumps(anno_dict)))
        #         return json.dumps(anno_dict)
        return anno_dict


class ViaDataset:
    annotations: list

    def __init__(self, json_file=None):
        if json_file:
            self.load(json_file)
        else:
            self.annotations = []
        pass

    def add(self, img_annotation: ViaAnnotation):
        if isinstance(img_annotation, ViaAnnotation):
            self.annotations.append(img_annotation)
        else:
            raise TypeError

    def merge(self, via_dataset, deduplicate=False):
        if isinstance(via_dataset, ViaDataset):
            self.annotations.extend(via_dataset.annotations)
            if deduplicate:
                self.annotations = list(set(self.annotations))
        return self

    def load(self, json_file):
        with open(json_file, 'r') as f:
            imgs_anns = json.load(f)
        self.annotations = [ViaAnnotation(anno['filename'], anno['size'], anno['regions'])
                            for anno in imgs_anns.values()]
        return self

    def load_dicts(self, dataset_dicts, img_dir, category_to_class: dict):
        annotations = []
        for d in dataset_dicts:
            img_annos = d['annotations']

            filename = d['file_name']
            size = os.path.getsize(os.path.join(img_dir, filename))
            regions = []

            for anno in img_annos:
                category_id = anno['category_id']
                segmentation = anno.get('segmentation')

                # Ploygon but not rectangle
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    r = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [],
                            "all_points_y": [],
                        },
                        'region_attributes': {
                            'class': category_to_class[category_id]
                        }
                    }
                    try:
                        poly = segmentation[0]
                    except KeyError:
                        global rle
                        rle = anno.copy()
                        print(anno)
                    all_points_x = []
                    all_points_y = []
                    for i in range(0, len(poly), 2):
                        all_points_x.append(poly[i])
                        all_points_y.append(poly[i + 1])
                    r["shape_attributes"]["all_points_x"] = all_points_x
                    r["shape_attributes"]["all_points_y"] = all_points_y

                # RLE or Rectangle
                else:
                    r = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 0,
                            "y": 0,
                            "width": 0,
                            "height": 0,
                        },
                        'region_attributes': {
                            'class': category_to_class[category_id]
                        }
                    }
                    bbox = anno['bbox']
                    bbox_mode = anno['bbox_mode']

                    r["shape_attributes"]["x"] = bbox[0]
                    r["shape_attributes"]["y"] = bbox[1]
                    if bbox_mode == BoxMode.XYWH_ABS:
                        r["shape_attributes"]["width"] = bbox[2]
                        r["shape_attributes"]["height"] = bbox[3]
                    elif bbox_mode == BoxMode.XYXY_ABS:
                        r["shape_attributes"]["width"] = bbox[2] - bbox[0]
                        r["shape_attributes"]["height"] = bbox[3] - bbox[1]
                    else:
                        raise TypeError
                regions.append(r)
            annotations.append(ViaAnnotation(filename, size, regions))
        self.annotations = annotations
        return self

    def map_class(self, categories_map: dict):
        new_annotations = []
        for anno in self.annotations:
            regions = anno.regions
            new_regions = []

            for r in regions:
                region_class = r['region_attributes']['class']
                if categories_map.get(region_class):
                    r['region_attributes']['class'] = categories_map[region_class]
                    new_regions.append(r)

            if len(new_regions) > 0:
                anno.regions = new_regions
                new_annotations.append(anno)
        self.annotations = new_annotations

    def load_coco(self, coco_json_file: str, img_dir: str, category_to_class: dict):
        tmp = str(random.randint(0, 1000))
        register_coco_instances(tmp, {}, coco_json_file, "")
        dataset_dicts = DatasetCatalog.get(tmp)
        self.load_dicts(dataset_dicts, img_dir, category_to_class)
        return self

    def dict(self):
        imgs_dict = {}
        for anno in self.annotations:
            k = anno.filename + str(anno.size)
            v = anno.dict()
            imgs_dict[k] = v
        return imgs_dict
    
    def get_standard_dataset_dicts(self, img_dir: str, class_to_category_id: dict):
        annotations = self.annotations
        dataset_dicts = []
        for idx, anno in enumerate(annotations):
            record = {}
            
            filename = os.path.join(img_dir, anno.filename)
            try:
                height, width = cv2.imread(filename).shape[:2]
            except Exception as e:
                print(filename)
                print(e)
                continue

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            
            objs = []
            for r in anno.regions:
                category_id = 0
                if r.get("region_attributes") and r["region_attributes"].get("class"):
                    category_id = class_to_category_id[r["region_attributes"]["class"]]
                
                shape_attr = r["shape_attributes"]
                if shape_attr["name"] == "polygon":
                    px = shape_attr["all_points_x"]
                    py = shape_attr["all_points_y"]
                    poly = [(x, y) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category_id,
                    }
                        
                elif shape_attr["name"] == "rect":
                    x = shape_attr["x"]
                    y = shape_attr["y"]
                    w = shape_attr["width"]
                    h = shape_attr["height"]

                    obj = {
                        "bbox": [x, y, w, h],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": category_id,
                    }
                else:
                    pass
                
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    def save(self, json_file):
        with open(json_file, 'w') as f:
            f.write(json.dumps(self.dict(), indent=4))
        return self
    
    def get_opt_polygons(self, default_max_overlap=0.5, default_type=0): 
        polys = []
        max_overlaps = []
        types = []
        annotations = self.annotations
        for idx, anno in enumerate(annotations):
            for r in anno.regions:
                try:
                    shape_attr = r.get("shape_attributes", None)
                    if shape_attr == None:
                        continue

                    if shape_attr["name"] == "polygon":
                        px = shape_attr["all_points_x"]
                        py = shape_attr["all_points_y"]
                        poly = [(x, y) for x, y in zip(px, py)]
                    elif shape_attr["name"] == "rect":
                        x = shape_attr["x"]
                        y = shape_attr["y"]
                        w = shape_attr["width"]
                        h = shape_attr["height"]
                        poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    else:
                        pass

                    poly = Polygon(poly)
                    polys.append(poly)
                    
                    try:
                        max_overlap = float(r["region_attributes"]["max_overlap"])
                    except:
                        max_overlap = default_max_overlap
                    
                    try:
                        type_ = int(r["region_attributes"]["type"])
                    except:
                        type_ = default_type
                    
                    max_overlaps.append(max_overlap)
                    types.append(type_)
                except Exception as e:
                    print("error in ", anno, e)
        
        return polys, max_overlaps, types


def copy_images(src_img_dir: str, src_via_dataset: ViaDataset, dst_img_dir: str):
    os.makedirs(dst_img_dir, exist_ok=True)
    annos = src_via_dataset.annotations
    for anno in annos:
        try:
            src = os.path.join(src_img_dir, anno.filename)
            copy(src, dst_img_dir)
        except Exception as e:
            print(e)
    src_via_dataset.save(os.path.join(dst_img_dir, 'via_region_data.json'))


def train_val_split(src_img_dir: str, src_via_dataset: ViaDataset,
                    dst_root: str, train_size: float = 0.75):
    random.seed(0)

    annos = src_via_dataset.annotations
    rand_idx = [_ for _ in range(len(annos))]
    random.shuffle(rand_idx)
    order = {
        "train": (0, int(train_size * len(annos))),
        "val": (int(train_size * len(annos)), len(annos)),
    }
    for train_or_val, range_ in order.items():
        img_dir = os.path.join(dst_root, train_or_val)
        os.makedirs(img_dir, exist_ok=True)
        via_ds = ViaDataset()
        for idx in rand_idx[range_[0]: range_[1]]:
            filename = annos[idx].filename
            try:
                src = os.path.join(src_img_dir, filename)
                copy(src, img_dir)
                via_ds.add(annos[idx])
            except Exception as e:
                print(e)
        via_ds.save(os.path.join(img_dir, 'via_region_data.json'))


def merge_images(src_img_dir1: str, src_via_dataset1: ViaDataset,
                 src_img_dir2: str, src_via_dataset2: ViaDataset,
                 dst_img_dir: str, dst_json_file: str = None, deduplicate=False):
    os.makedirs(dst_img_dir, exist_ok=True)

    dst_via_ds = ViaDataset()
    dst_via_ds.merge(src_via_dataset1, deduplicate)
    dst_via_ds.merge(src_via_dataset2, deduplicate)

    order = [
        (src_img_dir1, src_via_dataset1),
        (src_img_dir2, src_via_dataset2),
    ]
    for img_dir, dset in order:
        for anno in dset.annotations:
            filename = anno.filename
            try:
                copy(os.path.join(img_dir, filename), dst_img_dir)
            except Exception as e:
                print(e)
    if not dst_json_file:
        dst_json_file = os.path.join(dst_img_dir, 'via_region_data.json')
    dst_via_ds.save(dst_json_file)
    return dst_via_ds


if __name__ == '__main__':
    ds = ViaDataset('./datasets/VIA-person_and_mask/val/via_region_data.json')
    ds.map_class(categories_map={
        "face_with_mask": "face_with_mask",
        "with_mask": "face_with_mask",
        "mask_weared_incorrect": "face_with_mask",
        "person": "person",
    })
    ds.save('./datasets/VIA-person_and_mask/val/via_region_data.json')
    pass
