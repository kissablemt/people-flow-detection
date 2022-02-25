from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
import numpy as np
from typing import Tuple

thing_classes=['person', 'mask']
pm_metadata = Metadata().set(thing_classes=thing_classes)
# pm_metadata = None

def random_color():
    return (np.random.random(), np.random.random(), np.random.random())
    

def draw_person_and_mask_(img: np.ndarray, predict_outputs: dict, scale=1.0):
    v = Visualizer(img[:, :, ::-1], metadata=pm_metadata, scale=scale)
    out = v.draw_instance_predictions(predict_outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def draw_person_and_mask(img: np.ndarray, boxes: list, classes: list, scores: list, scale=1.0):
    '''
    boxes: List[XYXY]
    '''
    visualizer = Visualizer(img[:, :, ::-1], scale=scale)
    for box, class_, score in zip(boxes, classes, scores):
        color = random_color()
        out = visualizer.draw_box(box, edge_color=color)
        out = visualizer.draw_text("{} {:.2f}%".format(thing_classes[class_], score * 100), (box[0], box[1]), color=color, horizontal_alignment='left')
        img = out.get_image()[:, :, ::-1]
    return img

def draw_boxes(img: np.ndarray, boxes: list, scale=1.0):
    '''
    boxes: List[XYXY]
    '''
    visualizer = Visualizer(img[:, :, ::-1], scale=scale)
    for box in boxes:
        out = visualizer.draw_box(box, edge_color=random_color())
        img = out.get_image()[:, :, ::-1]
    return img

def draw_person_ids_centroid(img: np.ndarray, centroids: list, texts: list, scale=1.0):
    visualizer = Visualizer(img[:, :, ::-1], scale=scale)
    for ct, text in zip(centroids, texts):
        x, y = ct
        out = visualizer.draw_text(str(text), (x, y))
        img = out.get_image()[:, :, ::-1]
    return img
        
def draw_person_ids(img: np.ndarray, boxes: list,  texts: list, scale=1.0):
    visualizer = Visualizer(img[:, :, ::-1], scale=scale)
    for box, text in zip(boxes, texts):
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        out = visualizer.draw_text(str(text), (x, y))
        img = out.get_image()[:, :, ::-1]
    return img

def draw_text(img: np.ndarray, text: str, location: Tuple[float, float]):
    visualizer = Visualizer(img[:, :, ::-1], scale=1.0)
    out = visualizer.draw_text(str(text), location, horizontal_alignment='left')
    img = out.get_image()[:, :, ::-1]
    return img